/*
 * Copyright (C) 2026 The PenguinOS Project
 * SPDX-License-Identifier: Apache-2.0
 */

#define LOG_TAG "Laminar"

#include "Laminar.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <sstream>
#include <string>
#include <sys/syscall.h>
#include <thread>
#include <unordered_map>
#include <unistd.h>

#include <android-base/file.h>
#include <android-base/logging.h>


namespace {

constexpr int kMaxFloor = 1024;
constexpr int kCeiling = 640;
constexpr int kFutileFrames = 8;
// A frame over target raises the floor by this much of the full range per unit of overrun.
constexpr double kRaiseGain = 1.5;
// Frames with room lower it more gently, so one quick frame does not drop a busy session.
constexpr double kLowerGain = 0.4;
// Frames under this share of the target count as having room.
constexpr double kRoomRatio = 0.75;
constexpr int kQuietFramesBeforeLowering = 2;
constexpr int kMinStep = 24;
constexpr int kApplyThreshold = 16;
constexpr int kLoadUpFloor = 512;
// A session that stops reporting frames is idle; its floor drops back after this long.
constexpr auto kIdleReset = std::chrono::milliseconds(150);
// Floor for the first frames after CPU_LOAD_RESET, which HWUI sends when it starts drawing again
// after 100 ms without frames (a new touch, fling or animation). Those frames have no history to
// react to, so starting them at 0 made every gesture miss its first frames.
constexpr int kResetFloor = 384;
// A burst that starts this soon after the previous one ended likely needs a similar floor.
constexpr auto kBurstMemory = std::chrono::milliseconds(1000);

// Start GPU votes at a share of this device's maximum clock, then climb its supported levels.
// Frequencies themselves are discovered from the driver, never copied from another SoC.
constexpr int kGpuMinFloorPercent = 45;
// Frames with GPU time under this share of the target count as having room.
constexpr double kGpuRoomRatio = 0.5;
// GPU limited: SurfaceFlinger composes on the GPU after its CPU work, so both add up.
constexpr double kGpuLateShare = 0.5;
// Lower one step only after about 200 ms of easy frames at 120 Hz
constexpr int kGpuQuietFramesBeforeLowering = 24;
// Each GPU vote expires independently, including when CPU-only reports keep a session alive.
constexpr int kGpuLeaseMs = 250;
constexpr const char* kGpuMinFreqPath = "/sys/class/kgsl/kgsl-3d0/devfreq/min_freq";
// The vote lasts only while frames keep being reported
constexpr int kExpensiveRenderingLeaseMs = 100;

// Discover supported levels once; only floor transitions write sysfs afterward.
const std::vector<int64_t>& gpuStepsHz() {
    static const std::vector<int64_t> steps = [] {
        constexpr const char* paths[] = {
                "/sys/class/kgsl/kgsl-3d0/gpu_available_frequencies",
                "/sys/class/kgsl/kgsl-3d0/devfreq/available_frequencies",
        };
        for (const char* path : paths) {
            std::string contents;
            if (!android::base::ReadFileToString(path, &contents)) continue;

            std::istringstream input(contents);
            std::vector<int64_t> frequencies;
            int64_t frequencyHz;
            while (input >> frequencyHz) {
                // Keep exact driver frequencies, including fractional MHz steps.
                if (frequencyHz > 0 && frequencyHz <= 10'000'000'000LL) {
                    frequencies.push_back(frequencyHz);
                }
            }
            if (frequencies.empty()) continue;
            std::sort(frequencies.begin(), frequencies.end());
            frequencies.erase(std::unique(frequencies.begin(), frequencies.end()),
                              frequencies.end());

            const int64_t minimum =
                    (static_cast<int64_t>(frequencies.back()) * kGpuMinFloorPercent + 99) / 100;
            const auto first = std::lower_bound(frequencies.begin(), frequencies.end(), minimum);
            std::vector<int64_t> levels = {0};  // No vote: leave the clock to the governor.
            levels.insert(levels.end(), first, frequencies.end());
            LOG(INFO) << "GPU floor uses " << levels.size() - 1
                      << " driver levels up to " << levels.back() << " Hz";
            return levels;
        }
        LOG(WARNING) << "No readable GPU frequency table; keeping CPU hints without GPU votes";
        return std::vector<int64_t>{0};
    }();
    return steps;
}

constexpr uint64_t kKeepAll = 0x18;
constexpr uint64_t kUtilClampMin = 0x20;

struct SchedAttr {
    uint32_t size;
    uint32_t sched_policy;
    uint64_t sched_flags;
    int32_t sched_nice;
    uint32_t sched_priority;
    uint64_t sched_runtime;
    uint64_t sched_deadline;
    uint64_t sched_period;
    uint32_t sched_util_min;
    uint32_t sched_util_max;
};

bool setUtilMin(int32_t tid, int value) {
    SchedAttr attr = {};
    attr.size = sizeof(attr);
    attr.sched_flags = kKeepAll | kUtilClampMin;
    attr.sched_util_min = static_cast<uint32_t>(value);
    attr.sched_util_max = kMaxFloor;
    return syscall(__NR_sched_setattr, tid, &attr, 0) == 0;
}

std::mutex gIdleLock;
std::condition_variable gIdleCv;
std::unordered_map<LaminarController*, std::chrono::steady_clock::time_point> gIdleDeadlines;
std::once_flag gIdleThreadOnce;

// The GPU is shared, so the floor is the highest step any session votes for.
struct GpuVote {
    int step;
    std::chrono::steady_clock::time_point deadline;
};
std::mutex gGpuLock;
std::condition_variable gGpuCv;
std::unordered_map<const void*, GpuVote> gGpuVotes;
std::once_flag gGpuThreadOnce;
int gGpuApplied = 0;
std::string gGpuBaseline;
bool gGpuWriteFailed = false;

// Caller holds gGpuLock. Preserve the pre-existing floor when our last vote ends.
void applyGpuVoteLocked() {
    int top = 0;
    for (const auto& [_, vote] : gGpuVotes) top = std::max(top, vote.step);
    if (top == gGpuApplied) {
        if (top == 0) gGpuBaseline.clear();
        return;
    }
    if (gGpuApplied == 0 && gGpuBaseline.empty() &&
        !android::base::ReadFileToString(kGpuMinFreqPath, &gGpuBaseline)) {
        if (!gGpuWriteFailed) PLOG(WARNING) << "Cannot read GPU minimum frequency";
        gGpuWriteFailed = true;
        return;
    }
    // Do not lower a higher floor that another owner had already requested.
    const std::string value = top == 0 ? gGpuBaseline : std::to_string(std::max<int64_t>(
            gpuStepsHz()[top], std::strtoll(gGpuBaseline.c_str(), nullptr, 10)));
    if (!android::base::WriteStringToFile(value, kGpuMinFreqPath)) {
        if (!gGpuWriteFailed) PLOG(WARNING) << "Cannot write GPU minimum frequency";
        gGpuWriteFailed = true;
        return;
    }
    gGpuWriteFailed = false;
    gGpuApplied = top;
    if (top == 0) gGpuBaseline.clear();
}

void gpuLeaseLoop() {
    std::unique_lock<std::mutex> lock(gGpuLock);
    while (true) {
        const auto now = std::chrono::steady_clock::now();
        for (auto it = gGpuVotes.begin(); it != gGpuVotes.end();) {
            if (it->second.deadline <= now) it = gGpuVotes.erase(it);
            else ++it;
        }
        applyGpuVoteLocked();
        if (gGpuVotes.empty() && gGpuApplied == 0) {
            gGpuCv.wait(lock);
        } else {
            // Also retry a failed release; an I/O error must not strand a high floor.
            auto next = now + std::chrono::milliseconds(kGpuLeaseMs);
            for (const auto& [_, vote] : gGpuVotes) next = std::min(next, vote.deadline);
            gGpuCv.wait_until(lock, next);
        }
    }
}

void setGpuVote(const void* owner, int step, int leaseMs = kGpuLeaseMs) {
    std::call_once(gGpuThreadOnce, [] { std::thread(gpuLeaseLoop).detach(); });
    std::lock_guard<std::mutex> lock(gGpuLock);
    if (step > 0 && step < static_cast<int>(gpuStepsHz().size())) {
        gGpuVotes[owner] = {step, std::chrono::steady_clock::now() +
                                std::chrono::milliseconds(leaseMs)};
    } else {
        gGpuVotes.erase(owner);
    }
    applyGpuVoteLocked();
    gGpuCv.notify_one();
}

const int gExpensiveRenderingOwner = 0;
std::atomic<bool> gExpensiveRendering = false;

}  // namespace

void laminarIdleLoop() {
    std::unique_lock<std::mutex> lock(gIdleLock);
    while (true) {
        if (gIdleDeadlines.empty()) {
            gIdleCv.wait(lock);
            continue;
        }
        auto next = std::chrono::steady_clock::time_point::max();
        for (const auto& [controller, deadline] : gIdleDeadlines) next = std::min(next, deadline);
        gIdleCv.wait_until(lock, next);
        auto now = std::chrono::steady_clock::now();
        for (auto it = gIdleDeadlines.begin(); it != gIdleDeadlines.end();) {
            if (it->second <= now) {
                // Reset under gIdleLock so a controller being destroyed waits for it.
                it->first->resetIfIdle();
                it = gIdleDeadlines.erase(it);
            } else {
                ++it;
            }
        }
    }
}

LaminarController::LaminarController(const std::vector<int32_t>& threads) : mThreads(threads) {}

LaminarController::~LaminarController() {
    {
        std::lock_guard<std::mutex> lock(gIdleLock);
        gIdleDeadlines.erase(this);
    }
    reset();
}

bool LaminarController::enabled() {
    return true;
}

// Sent just before SurfaceFlinger composes blur or a color conversion on the GPU
void LaminarController::setExpensiveRendering(bool expected) {
    gExpensiveRendering = expected;
    setGpuVote(&gExpensiveRenderingOwner, expected ? 1 : 0, kExpensiveRenderingLeaseMs);
}

void LaminarController::setThreads(const std::vector<int32_t>& threads) {
    std::lock_guard<std::mutex> lock(mLock);
    int floor = mApplied;
    applyLocked(0);
    mThreads = threads;
    mApplied = -1;
    applyLocked(std::max(floor, 0));
}

void LaminarController::setTarget(int64_t targetNanos) {
    std::lock_guard<std::mutex> lock(mLock);
    if (targetNanos > 0) mTargetNanos = targetNanos;
}

void LaminarController::report(int64_t actualNanos, int64_t cpuNanos, int64_t gpuNanos) {
    if (gExpensiveRendering) {
        setGpuVote(&gExpensiveRenderingOwner, 1, kExpensiveRenderingLeaseMs);
    }
    {
        std::lock_guard<std::mutex> lock(mLock);
        // The utilization floor only helps the CPU part of the work. When the session reports the
        // split, GPU bound frames must not raise it.
        const int64_t cpuWork = cpuNanos > 0 ? cpuNanos : actualNanos;
        double ratio = static_cast<double>(cpuWork) / static_cast<double>(mTargetNanos);
        if (ratio > 1.0) {
            mQuietFrames = 0;
            if (mFloor >= kCeiling && ++mFutileFrames >= kFutileFrames) {
                mFloor = kCeiling / 2;
                mFutileFrames = 0;
            } else {
                int step = std::max(kMinStep,
                                    static_cast<int>(kRaiseGain * (ratio - 1.0) * kMaxFloor));
                mFloor = std::min(kCeiling, mFloor + step);
            }
        } else if (ratio < kRoomRatio) {
            mFutileFrames = 0;
            if (++mQuietFrames >= kQuietFramesBeforeLowering) {
                int step = std::max(kMinStep / 2,
                                    static_cast<int>(kLowerGain * (kRoomRatio - ratio) * kMaxFloor));
                mFloor = std::max(0, mFloor - step);
            }
        } else {
            mQuietFrames = 0;
            mFutileFrames = 0;
        }
        mBurstPeak = std::max(mBurstPeak, mFloor);
        applyLocked(mFloor);

        if (gpuNanos > 0) {
            const double gpuRatio =
                    static_cast<double>(gpuNanos) / static_cast<double>(mTargetNanos);
            const double actualRatio =
                    static_cast<double>(actualNanos) / static_cast<double>(mTargetNanos);
            if (gpuRatio > 1.0 || (actualRatio > 1.0 && gpuRatio >= kGpuLateShare)) {
                const int maxLevel = static_cast<int>(gpuStepsHz().size()) - 1;
                const bool farBehind = gpuRatio > 1.5 || actualRatio > 1.5;
                mGpuLevel = std::min(maxLevel, mGpuLevel + (farBehind ? 2 : 1));
                mGpuQuietFrames = 0;
            } else if (gpuRatio < kGpuRoomRatio && actualRatio < kRoomRatio) {
                if (++mGpuQuietFrames >= kGpuQuietFramesBeforeLowering) {
                    mGpuLevel = std::max(0, mGpuLevel - 1);
                    mGpuQuietFrames = 0;
                }
            } else {
                mGpuQuietFrames = 0;
            }
            // Serialize against reset()/pause so an older report cannot resurrect a vote.
            setGpuVote(this, mGpuLevel);
        }
    }
    armIdle();
}

void LaminarController::loadUp() {
    {
        std::lock_guard<std::mutex> lock(mLock);
        mFloor = std::max(mFloor, std::min(kLoadUpFloor, kCeiling));
        applyLocked(mFloor);
    }
    armIdle();
}

void LaminarController::loadSpike() {
    {
        std::lock_guard<std::mutex> lock(mLock);
        mFloor = kCeiling;
        applyLocked(mFloor);
    }
    armIdle();
}

void LaminarController::rebaseline() {
    {
        std::lock_guard<std::mutex> lock(mLock);
        int floor = kResetFloor;
        if (std::chrono::steady_clock::now() - mLastBurstEnd < kBurstMemory) {
            floor = std::max(floor, mLastBurstPeak * 3 / 4);
        }
        mFloor = std::min(kCeiling, floor);
        mFutileFrames = 0;
        mQuietFrames = 0;
        mBurstPeak = mFloor;
        applyLocked(mFloor);
    }
    armIdle();
}

void LaminarController::armIdle() {
    std::call_once(gIdleThreadOnce, [] { std::thread(laminarIdleLoop).detach(); });
    std::lock_guard<std::mutex> lock(gIdleLock);
    gIdleDeadlines[this] = std::chrono::steady_clock::now() + kIdleReset;
    gIdleCv.notify_one();
}

void LaminarController::resetIfIdle() {
    std::lock_guard<std::mutex> lock(mLock);
    mLastBurstPeak = mBurstPeak;
    mLastBurstEnd = std::chrono::steady_clock::now();
    mBurstPeak = 0;
    mFloor = 0;
    mQuietFrames = 0;
    applyLocked(0);
    mGpuLevel = 0;
    mGpuQuietFrames = 0;
    setGpuVote(this, 0);
}

void LaminarController::reset() {
    std::lock_guard<std::mutex> lock(mLock);
    mFloor = 0;
    mQuietFrames = 0;
    applyLocked(0);
    mGpuLevel = 0;
    mGpuQuietFrames = 0;
    setGpuVote(this, 0);
}

void LaminarController::applyLocked(int value) {
    if (mApplied >= 0 && std::abs(value - mApplied) < kApplyThreshold && value != 0) return;
    if (value == mApplied) return;
    for (int32_t tid : mThreads) {
        if (tid <= 0) continue;
        if (!setUtilMin(tid, value) && errno != ESRCH) {
            PLOG(WARNING) << "sched_setattr uclamp.min " << value << " on " << tid;
        }
    }
    mApplied = value;
}

/*
 * Copyright (C) 2026 The PenguinOS Project
 * SPDX-License-Identifier: Apache-2.0
 */

#define LOG_TAG "Laminar"

#include "Laminar.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <sys/syscall.h>
#include <thread>
#include <unordered_map>
#include <unistd.h>

#include <android-base/logging.h>

#include "utils.h"

namespace {

constexpr int kMaxFloor = 1024;
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

// GPU floor steps in MHz, all pineapple GPU power levels. Step 0 leaves the GPU to its governor.
constexpr int kGpuStepsMhz[] = {0, 680, 770, 834, 903};
constexpr int kGpuSteps = sizeof(kGpuStepsMhz) / sizeof(kGpuStepsMhz[0]);
// Frames with GPU time under this share of the target count as having room.
constexpr double kGpuRoomRatio = 0.5;
constexpr int kGpuQuietFramesBeforeLowering = 8;
// The floor is a perf lock lease that each GPU heavy frame renews, so it lapses on its own if
// frames stop.
constexpr int kGpuLeaseMs = 250;
// Perf HAL resource: GPU minimum frequency in MHz (the same one the expensive rendering boost uses).
constexpr int kGpuMinFreqResource = 0x4280C000;

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
std::mutex gGpuLock;
std::unordered_map<const LaminarController*, int> gGpuVotes;
int gGpuApplied = 0;
int gGpuHandle = 0;
std::chrono::steady_clock::time_point gGpuRenewed;

void setGpuVote(const LaminarController* controller, int step) {
    std::lock_guard<std::mutex> lock(gGpuLock);
    if (step > 0) {
        gGpuVotes[controller] = step;
    } else {
        gGpuVotes.erase(controller);
    }
    int top = 0;
    for (const auto& [_, vote] : gGpuVotes) top = std::max(top, vote);

    const auto now = std::chrono::steady_clock::now();
    if (top == gGpuApplied &&
        (top == 0 || now - gGpuRenewed < std::chrono::milliseconds(kGpuLeaseMs / 2))) {
        return;
    }
    if (gGpuHandle > 0 && (top == 0 || top != gGpuApplied)) {
        release_request(gGpuHandle);
        gGpuHandle = 0;
    }
    gGpuApplied = top;
    if (top == 0) return;

    int args[] = {kGpuMinFreqResource, kGpuStepsMhz[top]};
    const int handle = interaction_with_handle(gGpuHandle, kGpuLeaseMs, 2, args);
    gGpuHandle = handle > 0 ? handle : 0;
    gGpuRenewed = now;
}

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
    int gpuStep = -1;
    {
        std::lock_guard<std::mutex> lock(mLock);
        // The utilization floor only helps the CPU part of the work. When the session reports the
        // split, GPU bound frames must not raise it.
        const int64_t cpuWork = cpuNanos > 0 ? cpuNanos : actualNanos;
        double ratio = static_cast<double>(cpuWork) / static_cast<double>(mTargetNanos);
        if (ratio > 1.0) {
            int step = std::max(kMinStep, static_cast<int>(kRaiseGain * (ratio - 1.0) * kMaxFloor));
            mFloor = std::min(kMaxFloor, mFloor + step);
            mQuietFrames = 0;
        } else if (ratio < kRoomRatio) {
            if (++mQuietFrames >= kQuietFramesBeforeLowering) {
                int step = std::max(kMinStep / 2,
                                    static_cast<int>(kLowerGain * (kRoomRatio - ratio) * kMaxFloor));
                mFloor = std::max(0, mFloor - step);
            }
        } else {
            mQuietFrames = 0;
        }
        mBurstPeak = std::max(mBurstPeak, mFloor);
        applyLocked(mFloor);

        if (gpuNanos > 0) {
            const double gpuRatio =
                    static_cast<double>(gpuNanos) / static_cast<double>(mTargetNanos);
            if (gpuRatio > 1.0) {
                mGpuLevel = std::min(kGpuSteps - 1, mGpuLevel + (gpuRatio > 1.5 ? 2 : 1));
                mGpuQuietFrames = 0;
            } else if (gpuRatio < kGpuRoomRatio) {
                if (++mGpuQuietFrames >= kGpuQuietFramesBeforeLowering) {
                    mGpuLevel = std::max(0, mGpuLevel - 1);
                    mGpuQuietFrames = 0;
                }
            } else {
                mGpuQuietFrames = 0;
            }
            gpuStep = mGpuLevel;
        }
    }
    if (gpuStep >= 0) setGpuVote(this, gpuStep);
    armIdle();
}

void LaminarController::loadUp() {
    {
        std::lock_guard<std::mutex> lock(mLock);
        mFloor = std::max(mFloor, kLoadUpFloor);
        applyLocked(mFloor);
    }
    armIdle();
}

void LaminarController::loadSpike() {
    {
        std::lock_guard<std::mutex> lock(mLock);
        mFloor = kMaxFloor;
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
        mFloor = std::min(kMaxFloor, floor);
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

/*
 * Copyright (C) 2026 The PenguinOS Project
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <chrono>
#include <cstdint>
#include <mutex>
#include <vector>

/**
 * Keeps a hint session's threads on time by raising their scheduler utilization floor
 * (uclamp.min) when frames run over the target and lowering it again when they have room.
 */
class LaminarController {
  public:
    explicit LaminarController(const std::vector<int32_t>& threads);
    ~LaminarController();

    static bool enabled();

    void setThreads(const std::vector<int32_t>& threads);
    void setTarget(int64_t targetNanos);
    void report(int64_t actualNanos);
    void loadUp();
    void loadSpike();
    // CPU_LOAD_RESET: the coming work is unknown, start from a baseline floor.
    void rebaseline();
    void reset();

  private:
    void applyLocked(int value);
    void armIdle();
    void resetIfIdle();

    friend void laminarIdleLoop();

    std::mutex mLock;
    std::vector<int32_t> mThreads;
    int64_t mTargetNanos = 16'666'666;
    int mFloor = 0;
    int mApplied = -1;
    int mQuietFrames = 0;
    // Highest floor of the current burst, and of the last one once the session went idle.
    int mBurstPeak = 0;
    int mLastBurstPeak = 0;
    std::chrono::steady_clock::time_point mLastBurstEnd;
};

/*
 * Copyright (C) 2026 The PenguinOS Project
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

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
    void reset();

  private:
    void applyLocked(int value);

    std::mutex mLock;
    std::vector<int32_t> mThreads;
    int64_t mTargetNanos = 16'666'666;
    int mFloor = 0;
    int mApplied = -1;
    int mQuietFrames = 0;
};

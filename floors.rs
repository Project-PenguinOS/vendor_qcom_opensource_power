/*
 * Copyright (C) 2026 The PenguinOS Project
 * SPDX-License-Identifier: Apache-2.0
 */

//! Laminar's per-session floor controller: how frame reports move the uclamp.min floor of the
//! session's threads and its step on the shared GPU frequency floor. No side effects here.

use std::time::{Duration, Instant};

pub const MAX_FLOOR: i32 = 1024;
const CEILING: i32 = 640;
const FUTILE_FRAMES: i32 = 8;
// A frame over target raises the floor by this much of the full range per unit of overrun.
const RAISE_GAIN: f64 = 1.5;
// Frames with room lower it more gently, so one quick frame does not drop a busy session.
const LOWER_GAIN: f64 = 0.4;
// Frames under this share of the target count as having room.
const ROOM_RATIO: f64 = 0.75;
const QUIET_FRAMES_BEFORE_LOWERING: i32 = 2;
const MIN_STEP: i32 = 24;
const APPLY_THRESHOLD: i32 = 16;
const LOAD_UP_FLOOR: i32 = 512;
// Floor for the first frames after CPU_LOAD_RESET, which HWUI sends when it starts drawing again
// after 100 ms without frames (a new touch, fling or animation). Those frames have no history to
// react to, so starting them at 0 made every gesture miss its first frames.
const RESET_FLOOR: i32 = 384;
// A burst that starts this soon after the previous one ended likely needs a similar floor.
const BURST_MEMORY: Duration = Duration::from_millis(1000);

// Frames with GPU time under this share of the target count as having room.
const GPU_ROOM_RATIO: f64 = 0.5;
// GPU limited: SurfaceFlinger composes on the GPU after its CPU work, so both add up.
const GPU_LATE_SHARE: f64 = 0.5;
// Lower one step only after about 200 ms of easy frames at 120 Hz.
const GPU_QUIET_FRAMES_BEFORE_LOWERING: i32 = 24;

pub struct Floors {
    target_nanos: i64,
    pub floor: i32,
    quiet_frames: i32,
    futile_frames: i32,
    // Highest floor of the current burst, and of the last one once the session went idle.
    burst_peak: i32,
    last_burst_peak: i32,
    last_burst_end: Option<Instant>,
    pub gpu_level: usize,
    gpu_quiet_frames: i32,
}

impl Default for Floors {
    fn default() -> Self {
        Floors {
            target_nanos: 16_666_666,
            floor: 0,
            quiet_frames: 0,
            futile_frames: 0,
            burst_peak: 0,
            last_burst_peak: 0,
            last_burst_end: None,
            gpu_level: 0,
            gpu_quiet_frames: 0,
        }
    }
}

impl Floors {
    pub fn set_target(&mut self, target_nanos: i64) {
        if target_nanos > 0 {
            self.target_nanos = target_nanos;
        }
    }

    fn recent_burst(&self, now: Instant) -> bool {
        self.last_burst_end.is_some_and(|end| now.saturating_duration_since(end) < BURST_MEMORY)
    }

    /// A frame report. `cpu_nanos` and `gpu_nanos` are the split of `actual_nanos` when the
    /// session reports it, else 0. Returns the session's new GPU step when it reported GPU time.
    pub fn report(
        &mut self,
        now: Instant,
        actual_nanos: i64,
        cpu_nanos: i64,
        gpu_nanos: i64,
        gpu_max_level: usize,
    ) -> Option<usize> {
        // First frame after the session went idle. Sessions that send no CPU_LOAD_RESET, like
        // SurfaceFlinger's, start from the last burst's floor here like rebaseline() does.
        if self.floor == 0 && self.burst_peak == 0 && self.last_burst_peak > 0 && self.recent_burst(now)
        {
            self.floor = CEILING.min(self.last_burst_peak * 3 / 4);
        }

        // The utilization floor only helps the CPU part of the work. When the session reports the
        // split, GPU bound frames must not raise it.
        let cpu_work = if cpu_nanos > 0 { cpu_nanos } else { actual_nanos };
        let ratio = cpu_work as f64 / self.target_nanos as f64;
        if ratio > 1.0 {
            self.quiet_frames = 0;
            self.futile_frames += i32::from(self.floor >= CEILING);
            if self.floor >= CEILING && self.futile_frames >= FUTILE_FRAMES {
                self.floor = CEILING / 2;
                self.futile_frames = 0;
            } else {
                let step = MIN_STEP.max((RAISE_GAIN * (ratio - 1.0) * MAX_FLOOR as f64) as i32);
                self.floor = CEILING.min(self.floor.saturating_add(step));
            }
        } else if ratio < ROOM_RATIO {
            self.futile_frames = 0;
            self.quiet_frames += 1;
            if self.quiet_frames >= QUIET_FRAMES_BEFORE_LOWERING {
                let step = (MIN_STEP / 2)
                    .max((LOWER_GAIN * (ROOM_RATIO - ratio) * MAX_FLOOR as f64) as i32);
                self.floor = 0.max(self.floor - step);
            }
        } else {
            self.quiet_frames = 0;
            self.futile_frames = 0;
        }
        self.burst_peak = self.burst_peak.max(self.floor);

        if gpu_nanos <= 0 {
            return None;
        }
        let gpu_ratio = gpu_nanos as f64 / self.target_nanos as f64;
        let actual_ratio = actual_nanos as f64 / self.target_nanos as f64;
        if gpu_ratio > 1.0 || (actual_ratio > 1.0 && gpu_ratio >= GPU_LATE_SHARE) {
            let far_behind = gpu_ratio > 1.5 || actual_ratio > 1.5;
            self.gpu_level = gpu_max_level.min(self.gpu_level + if far_behind { 2 } else { 1 });
            self.gpu_quiet_frames = 0;
        } else if gpu_ratio < GPU_ROOM_RATIO && actual_ratio < ROOM_RATIO {
            self.gpu_quiet_frames += 1;
            if self.gpu_quiet_frames >= GPU_QUIET_FRAMES_BEFORE_LOWERING {
                self.gpu_level = self.gpu_level.saturating_sub(1);
                self.gpu_quiet_frames = 0;
            }
        } else {
            self.gpu_quiet_frames = 0;
        }
        Some(self.gpu_level)
    }

    pub fn load_up(&mut self) {
        self.floor = self.floor.max(LOAD_UP_FLOOR.min(CEILING));
    }

    pub fn load_spike(&mut self) {
        self.floor = CEILING;
    }

    /// CPU_LOAD_RESET: the coming work is unknown, start from a baseline floor.
    pub fn rebaseline(&mut self, now: Instant) {
        let mut floor = RESET_FLOOR;
        if self.recent_burst(now) {
            floor = floor.max(self.last_burst_peak * 3 / 4);
        }
        self.floor = CEILING.min(floor);
        self.futile_frames = 0;
        self.quiet_frames = 0;
        self.burst_peak = self.floor;
    }

    /// The session stopped reporting frames: remember the burst and drop the floors.
    pub fn end_burst(&mut self, now: Instant) {
        self.last_burst_peak = self.burst_peak;
        self.last_burst_end = Some(now);
        self.burst_peak = 0;
        self.reset();
    }

    pub fn reset(&mut self) {
        self.floor = 0;
        self.quiet_frames = 0;
        self.gpu_level = 0;
        self.gpu_quiet_frames = 0;
    }
}

/// Whether a floor change is worth the sched_setattr calls. `applied` is None before the first.
pub fn should_apply(applied: Option<i32>, value: i32) -> bool {
    match applied {
        None => true,
        Some(applied) if value == applied => false,
        Some(applied) => value == 0 || (value - applied).abs() >= APPLY_THRESHOLD,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TARGET: i64 = 10_000_000;

    fn floors() -> Floors {
        let mut f = Floors::default();
        f.set_target(TARGET);
        f
    }

    #[test]
    fn overrun_raises_by_gain() {
        let mut f = floors();
        f.report(Instant::now(), TARGET * 3 / 2, 0, 0, 0);
        // 1.5 * 0.5 * 1024
        assert_eq!(f.floor, 768.min(CEILING));
        let mut f = floors();
        f.report(Instant::now(), TARGET + TARGET / 100, 0, 0, 0);
        assert_eq!(f.floor, MIN_STEP);
    }

    #[test]
    fn room_lowers_after_two_quiet_frames() {
        let mut f = floors();
        f.load_spike();
        let now = Instant::now();
        f.report(now, TARGET / 4, 0, 0, 0);
        assert_eq!(f.floor, CEILING);
        f.report(now, TARGET / 4, 0, 0, 0);
        // 0.4 * 0.5 * 1024 = 204
        assert_eq!(f.floor, CEILING - 204);
    }

    #[test]
    fn cpu_split_keeps_gpu_bound_frames_from_raising() {
        let mut f = floors();
        f.report(Instant::now(), TARGET * 3 / 2, TARGET / 2, TARGET * 6 / 5, 5);
        assert_eq!(f.floor, 0);
        assert_eq!(f.gpu_level, 1);
    }

    #[test]
    fn futile_frames_at_ceiling_halve_the_floor() {
        let mut f = floors();
        f.load_spike();
        let now = Instant::now();
        for _ in 0..FUTILE_FRAMES - 1 {
            f.report(now, TARGET * 2, 0, 0, 0);
            assert_eq!(f.floor, CEILING);
        }
        f.report(now, TARGET * 2, 0, 0, 0);
        assert_eq!(f.floor, CEILING / 2);
    }

    #[test]
    fn gpu_far_behind_climbs_two_and_caps() {
        let mut f = floors();
        let now = Instant::now();
        assert_eq!(f.report(now, TARGET * 2, TARGET / 2, TARGET * 2, 3), Some(2));
        assert_eq!(f.report(now, TARGET * 2, TARGET / 2, TARGET * 2, 3), Some(3));
        for _ in 0..GPU_QUIET_FRAMES_BEFORE_LOWERING - 1 {
            assert_eq!(f.report(now, TARGET / 4, TARGET / 8, TARGET / 8, 3), Some(3));
        }
        assert_eq!(f.report(now, TARGET / 4, TARGET / 8, TARGET / 8, 3), Some(2));
    }

    #[test]
    fn burst_memory_seeds_the_next_burst() {
        let mut f = floors();
        let now = Instant::now();
        f.load_spike();
        f.report(now, TARGET, 0, 0, 0);
        f.end_burst(now);
        assert_eq!(f.floor, 0);
        f.report(now + Duration::from_millis(200), TARGET * 4 / 5, 0, 0, 0);
        assert_eq!(f.floor, CEILING * 3 / 4);

        let mut f = floors();
        f.load_spike();
        f.report(now, TARGET, 0, 0, 0);
        f.end_burst(now);
        f.report(now + BURST_MEMORY, TARGET * 4 / 5, 0, 0, 0);
        assert_eq!(f.floor, 0);
    }

    #[test]
    fn rebaseline_uses_reset_floor_or_last_burst() {
        let mut f = floors();
        let now = Instant::now();
        f.rebaseline(now);
        assert_eq!(f.floor, RESET_FLOOR);
        f.load_spike();
        f.report(now, TARGET, 0, 0, 0);
        f.end_burst(now);
        f.rebaseline(now);
        assert_eq!(f.floor, RESET_FLOOR.max(CEILING * 3 / 4));
    }

    #[test]
    fn tiny_target_does_not_overflow() {
        let mut f = Floors::default();
        f.set_target(1);
        let now = Instant::now();
        f.report(now, 5_000_000, 0, 5_000_000, 3);
        f.report(now, 5_000_000, 0, 5_000_000, 3);
        assert_eq!(f.floor, CEILING);
        assert_eq!(f.gpu_level, 3);
    }

    #[test]
    fn apply_threshold() {
        assert!(should_apply(None, 0));
        assert!(!should_apply(Some(100), 100));
        assert!(!should_apply(Some(100), 110));
        assert!(should_apply(Some(100), 116));
        assert!(should_apply(Some(100), 0));
    }
}

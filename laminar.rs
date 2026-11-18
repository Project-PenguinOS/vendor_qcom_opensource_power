/*
 * Copyright (C) 2026 The PenguinOS Project
 * SPDX-License-Identifier: Apache-2.0
 */

//! Laminar keeps a hint session's threads on time by raising their scheduler utilization floor
//! (uclamp.min) when frames run over the target and lowering it again when they have room.
//! Sessions that report GPU time (SurfaceFlinger) also vote for a GPU frequency floor.

use crate::floors::{should_apply, Floors, MAX_FLOOR};
use crate::gpu::{self, Owner};
use log::warn;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock, Weak};
use std::thread;
use std::time::{Duration, Instant};

// A session that stops reporting frames is idle; its floor drops back after this long.
const IDLE_RESET: Duration = Duration::from_millis(150);
// The expensive rendering vote lasts only while frames keep being reported.
const EXPENSIVE_RENDERING_LEASE: Duration = Duration::from_millis(100);
// Vote only for a stream of blurred frames. Single composites, like the ones region sampling
// causes a few times a second on an idle screen, are not worth a GPU floor.
const EXPENSIVE_STREAM_GAP: Duration = Duration::from_millis(50);
// Input events renew the touch vote about every 100 ms while a finger is down.
const INTERACTION_GPU_LEASE: Duration = Duration::from_millis(150);

const SCHED_FLAG_KEEP_ALL: u64 = 0x18;
const SCHED_FLAG_UTIL_CLAMP_MIN: u64 = 0x20;

#[repr(C)]
#[derive(Default)]
struct SchedAttr {
    size: u32,
    sched_policy: u32,
    sched_flags: u64,
    sched_nice: i32,
    sched_priority: u32,
    sched_runtime: u64,
    sched_deadline: u64,
    sched_period: u64,
    sched_util_min: u32,
    sched_util_max: u32,
}

fn set_util_min(tid: i32, value: i32) {
    let attr = SchedAttr {
        size: std::mem::size_of::<SchedAttr>() as u32,
        sched_flags: SCHED_FLAG_KEEP_ALL | SCHED_FLAG_UTIL_CLAMP_MIN,
        sched_util_min: value as u32,
        sched_util_max: MAX_FLOOR as u32,
        ..Default::default()
    };
    // SAFETY: `attr` is a valid sched_attr that outlives the call; the kernel only reads it.
    if unsafe { libc::syscall(libc::SYS_sched_setattr, tid, &attr as *const SchedAttr, 0) } != 0 {
        let e = std::io::Error::last_os_error();
        if e.raw_os_error() != Some(libc::ESRCH) {
            warn!("sched_setattr uclamp.min {value} on {tid}: {e}");
        }
    }
}

static EXPENSIVE_RENDERING: AtomicBool = AtomicBool::new(false);
static LAST_EXPENSIVE_FRAME_NS: AtomicI64 = AtomicI64::new(i64::MIN / 2);

fn monotonic_ns() -> i64 {
    static START: OnceLock<Instant> = OnceLock::new();
    START.get_or_init(Instant::now).elapsed().as_nanos() as i64
}

/// Sent just before SurfaceFlinger composes blur or a color conversion on the GPU.
pub fn set_expensive_rendering(expected: bool) {
    EXPENSIVE_RENDERING.store(expected, Ordering::Relaxed);
    if !expected {
        gpu::vote(Owner::ExpensiveRendering, 0, EXPENSIVE_RENDERING_LEASE);
    }
}

/// Touch input: hold a GPU floor briefly so the first frames of a gesture do not wait for the
/// GPU clock to ramp up.
pub fn interaction(duration_ms: i32) {
    let lease = INTERACTION_GPU_LEASE.max(Duration::from_millis(duration_ms.max(0) as u64));
    gpu::vote(Owner::Interaction, 1, lease);
}

struct State {
    threads: Vec<i32>,
    floors: Floors,
    applied: Option<i32>,
    last_activity: Instant,
}

impl State {
    fn apply(&mut self, value: i32) {
        if !should_apply(self.applied, value) {
            return;
        }
        for &tid in self.threads.iter().filter(|&&tid| tid > 0) {
            set_util_min(tid, value);
        }
        self.applied = Some(value);
    }
}

pub struct Laminar {
    owner: Owner,
    state: Mutex<State>,
}

impl Laminar {
    pub fn new(id: u64, threads: &[i32]) -> Arc<Laminar> {
        Arc::new(Laminar {
            owner: Owner::Session(id),
            state: Mutex::new(State {
                threads: threads.to_vec(),
                floors: Floors::default(),
                applied: None,
                last_activity: Instant::now(),
            }),
        })
    }

    pub fn set_threads(&self, threads: &[i32]) {
        let mut state = self.state.lock().unwrap();
        let floor = state.applied.unwrap_or(0);
        state.apply(0);
        state.threads = threads.to_vec();
        state.applied = None;
        state.apply(floor);
    }

    pub fn set_target(&self, target_nanos: i64) {
        self.state.lock().unwrap().floors.set_target(target_nanos);
    }

    pub fn report(self: &Arc<Self>, actual_nanos: i64, cpu_nanos: i64, gpu_nanos: i64) {
        if EXPENSIVE_RENDERING.load(Ordering::Relaxed) {
            let now = monotonic_ns();
            let last = LAST_EXPENSIVE_FRAME_NS.swap(now, Ordering::Relaxed);
            if now - last < EXPENSIVE_STREAM_GAP.as_nanos() as i64 {
                gpu::vote(Owner::ExpensiveRendering, 1, EXPENSIVE_RENDERING_LEASE);
            }
        }
        {
            let mut state = self.state.lock().unwrap();
            let now = Instant::now();
            state.last_activity = now;
            let max_level = gpu::steps_hz().len() - 1;
            let gpu_level = state.floors.report(now, actual_nanos, cpu_nanos, gpu_nanos, max_level);
            let floor = state.floors.floor;
            state.apply(floor);
            // Under the session lock, so an older report cannot resurrect a vote after reset().
            if let Some(level) = gpu_level {
                gpu::vote(self.owner, level, gpu::DEFAULT_LEASE);
            }
        }
        self.arm_idle();
    }

    fn update(self: &Arc<Self>, change: impl FnOnce(&mut Floors, Instant)) {
        {
            let mut state = self.state.lock().unwrap();
            let now = Instant::now();
            state.last_activity = now;
            change(&mut state.floors, now);
            let floor = state.floors.floor;
            state.apply(floor);
        }
        self.arm_idle();
    }

    pub fn load_up(self: &Arc<Self>) {
        self.update(|f, _| f.load_up());
    }

    pub fn load_spike(self: &Arc<Self>) {
        self.update(|f, _| f.load_spike());
    }

    pub fn rebaseline(self: &Arc<Self>) {
        self.update(Floors::rebaseline);
    }

    pub fn reset(&self) {
        let mut state = self.state.lock().unwrap();
        state.floors.reset();
        state.apply(0);
        gpu::vote(self.owner, 0, gpu::DEFAULT_LEASE);
    }

    fn reset_if_idle(&self) {
        let mut state = self.state.lock().unwrap();
        let now = Instant::now();
        // A report that raced the idle timer keeps the session busy.
        if now.saturating_duration_since(state.last_activity) < IDLE_RESET {
            return;
        }
        state.floors.end_burst(now);
        state.apply(0);
        gpu::vote(self.owner, 0, gpu::DEFAULT_LEASE);
    }

    fn arm_idle(self: &Arc<Self>) {
        let timer = idle_timer();
        let mut deadlines = timer.deadlines.lock().unwrap();
        deadlines.insert(self.owner, (Instant::now() + IDLE_RESET, Arc::downgrade(self)));
        timer.changed.notify_one();
    }
}

impl Drop for Laminar {
    fn drop(&mut self) {
        self.reset();
    }
}

struct IdleTimer {
    deadlines: Mutex<HashMap<Owner, (Instant, Weak<Laminar>)>>,
    changed: Condvar,
}

fn idle_timer() -> &'static IdleTimer {
    static TIMER: OnceLock<IdleTimer> = OnceLock::new();
    TIMER.get_or_init(|| {
        thread::Builder::new()
            .name("laminar-idle".to_owned())
            .spawn(idle_loop)
            .expect("Failed to start the Laminar idle thread");
        IdleTimer { deadlines: Mutex::default(), changed: Condvar::new() }
    })
}

fn idle_loop() {
    let timer = idle_timer();
    let mut deadlines = timer.deadlines.lock().unwrap();
    loop {
        let now = Instant::now();
        let mut expired = Vec::new();
        deadlines.retain(|_, (deadline, session)| {
            if *deadline > now {
                return true;
            }
            expired.extend(session.upgrade());
            false
        });
        if !expired.is_empty() {
            drop(deadlines);
            // Session locks are never taken under the timer lock.
            for session in expired {
                session.reset_if_idle();
            }
            deadlines = timer.deadlines.lock().unwrap();
            continue;
        }
        deadlines = match deadlines.values().map(|&(deadline, _)| deadline).min() {
            Some(next) => {
                timer.changed.wait_timeout(deadlines, next.saturating_duration_since(now)).unwrap().0
            }
            None => timer.changed.wait(deadlines).unwrap(),
        };
    }
}

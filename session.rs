/*
 * SPDX-FileCopyrightText: The LineageOS Project
 * SPDX-License-Identifier: Apache-2.0
 */

//! Hint sessions: Laminar for every session, plus the QTI ADPF hints for game sessions.

use crate::laminar::Laminar;
use crate::perf;
use android_hardware_power::aidl::android::hardware::power::{
    IPowerHintSession::IPowerHintSession, SessionConfig::SessionConfig,
    SessionHint::SessionHint, SessionMode::SessionMode, WorkDuration::WorkDuration,
};
use binder::{ExceptionCode, Interface, Status};
use log::{debug, error, info};
use std::sync::atomic::{AtomicI32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

const VENDOR_FEEDBACK_WORKLOAD_TYPE: i32 = 0x1601;
const VENDOR_FEEDBACK_TOPAPP_NAME: i32 = 0x1611;
const VENDOR_HINT_PICARD_TOP_APP: i32 = 0x104A;
const VENDOR_HINT_PICARD_LOW_LAT: i32 = 0x104C;
const VENDOR_HINT_PICARD_HIGH_CPUUTIL: i32 = 0x104D;
const VENDOR_HINT_PICARD_LOAD_CHANGED: i32 = 0x104E;
const SCHED_TASK_LOAD_BOOST: i32 = 0x43C04000;
const GAME_WORKLOAD_TYPE: i32 = 2;

const GPU_BOOST_PCT: i32 = 10;
const MAX_THREADS: usize = 16;
const MAX_TLB_THREADS: usize = 5;
const MAX_TLB_PCT: i32 = 80;
const TLB_STEP_PCT: i32 = 10;

pub const PREFERRED_RATE_NANOS: i64 = 16_666_666;

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

/// The read-only perf config store values ADPF uses.
struct AdpfConfig {
    max_boost_count: u32,
    boost_duration: Duration,
    max_pipeline_threads: usize,
}

fn adpf_config() -> &'static AdpfConfig {
    static CONFIG: OnceLock<AdpfConfig> = OnceLock::new();
    CONFIG.get_or_init(|| AdpfConfig {
        max_boost_count: perf::get_prop_int("ro.vendor.perf.qape.max_boost_count", 3).max(0) as u32,
        boost_duration: Duration::from_secs(
            perf::get_prop_int("ro.vendor.perf.qape.boost_duration", 10).max(0) as u64,
        ),
        max_pipeline_threads: perf::get_prop_int("ro.vendor.perf.qape.max_pipeline_number", 3)
            .max(0) as usize,
    })
}

/// CPU/GPU load spike boosts are rate limited across all sessions.
fn boost_eligible() -> bool {
    struct Window {
        start: Instant,
        count: u32,
    }
    static WINDOW: Mutex<Option<Window>> = Mutex::new(None);

    let config = adpf_config();
    let now = Instant::now();
    let mut window = WINDOW.lock().unwrap();
    match window.as_mut() {
        Some(w) if now.duration_since(w.start) <= config.boost_duration => {
            if w.count >= config.max_boost_count {
                return false;
            }
            w.count += 1;
        }
        _ => *window = Some(Window { start: now, count: 1 }),
    }
    true
}

/// Top app hint failures before the first success, or -1 once it worked. Past the limit the
/// perf HAL is taken to lack the hint (pineapple's perfboostsconfig has none) and it is no
/// longer sent.
static TOP_APP_HINT_FAILURES: AtomicI32 = AtomicI32::new(0);
const TOP_APP_HINT_MAX_FAILURES: i32 = 3;

/// The top app, when the perf HAL classifies it as a game. ADPF hints only apply to those.
fn game_top_app() -> Option<String> {
    if TOP_APP_HINT_FAILURES.load(Ordering::Relaxed) >= TOP_APP_HINT_MAX_FAILURES {
        return None;
    }
    let name = perf::sync_request(VENDOR_FEEDBACK_TOPAPP_NAME)?;
    if perf::hint(VENDOR_HINT_PICARD_TOP_APP, &name, 0, 0) == -1 {
        if TOP_APP_HINT_FAILURES.load(Ordering::Relaxed) >= 0
            && TOP_APP_HINT_FAILURES.fetch_add(1, Ordering::Relaxed) + 1
                == TOP_APP_HINT_MAX_FAILURES
        {
            info!("The perf HAL has no top app hint, game detection is off");
        }
        return None;
    }
    TOP_APP_HINT_FAILURES.store(-1, Ordering::Relaxed);
    (perf::get_feedback(VENDOR_FEEDBACK_WORKLOAD_TYPE, &name) == GAME_WORKLOAD_TYPE).then_some(name)
}

/// Task load boost: a per-thread scheduler load scaling the perf HAL holds as a lock.
#[derive(Default)]
struct TaskLoadBoost {
    handle: i32,
    sum: i32,
}

struct Adpf {
    top_app: String,
    pipeline_threads: usize,
    efficiency_threads: usize,
    boost: TaskLoadBoost,
}

impl Adpf {
    fn release_boost(&mut self) {
        if self.boost.handle > 0 {
            perf::release(self.boost.handle);
        }
        self.boost = TaskLoadBoost { handle: -1, sum: 0 };
    }

    fn change_boost(&mut self, threads: &[i32], step: i32) {
        let sum = (self.boost.sum + step).clamp(-MAX_TLB_PCT, MAX_TLB_PCT);
        if self.boost.handle > 0 {
            if sum == self.boost.sum {
                return;
            }
            perf::release(self.boost.handle);
        }
        self.boost.handle = -1;

        let mut handle = -1;
        if sum != 0 {
            let mut resources: Vec<i32> = threads
                .iter()
                .take(MAX_TLB_THREADS)
                .flat_map(|&tid| {
                    let value = (sum.abs() & 0x7F) | if sum < 0 { 0x80 } else { 0 } | (tid << 8);
                    [SCHED_TASK_LOAD_BOOST, value]
                })
                .collect();
            handle = if resources.is_empty() { 0 } else { perf::lock_acquire(-1, 0, &mut resources) };
            if handle < 0 {
                error!("Unable to apply boost");
                return;
            }
        }
        self.boost = TaskLoadBoost { handle, sum };
    }

    fn release_pipeline(&mut self, threads: &[i32]) {
        for &tid in threads.iter().take(self.pipeline_threads).filter(|&&tid| tid != 0) {
            perf::hint(VENDOR_HINT_PICARD_LOW_LAT, &self.top_app, tid, 0);
        }
        self.pipeline_threads = 0;
    }

    fn release_low_cpu_util(&mut self, threads: &[i32]) {
        for &tid in threads.iter().take(self.efficiency_threads).filter(|&&tid| tid != 0) {
            perf::hint(VENDOR_HINT_PICARD_HIGH_CPUUTIL, &self.top_app, tid, 0);
        }
        self.efficiency_threads = 0;
    }

    /// Sends `hint_id` for up to `max` threads and returns how many took it.
    fn hint_threads(&self, threads: &[i32], hint_id: i32, max: usize) -> usize {
        threads
            .iter()
            .filter(|&&tid| tid != 0)
            .take(max)
            .take_while(|&&tid| perf::hint(hint_id, &self.top_app, tid, 2) != -1)
            .count()
    }

    fn release_all(&mut self, threads: &[i32]) {
        self.release_low_cpu_util(threads);
        self.release_pipeline(threads);
    }

    fn load_changed(&self, value: i32, kind: i32) {
        if !boost_eligible() {
            error!("Boost too frequent");
            return;
        }
        perf::hint(VENDOR_HINT_PICARD_LOAD_CHANGED, &self.top_app, value, kind);
    }
}

struct State {
    threads: Vec<i32>,
    /// pause() makes a session inactive; close() ends it for good.
    closed: bool,
    active: bool,
    adpf: Option<Adpf>,
}

pub struct PowerHintSession {
    id: u64,
    debug: bool,
    laminar: Arc<Laminar>,
    state: Mutex<State>,
}

impl PowerHintSession {
    pub fn new(threads: &[i32], duration_nanos: i64) -> PowerHintSession {
        let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
        let debug = perf::get_prop_int("vendor.debug.enable.adpf", 0) == 1;
        let adpf = game_top_app().map(|top_app| Adpf {
            top_app,
            pipeline_threads: 0,
            efficiency_threads: 0,
            boost: TaskLoadBoost { handle: -1, sum: 0 },
        });
        let session = PowerHintSession {
            id,
            debug,
            laminar: Laminar::new(id, threads),
            state: Mutex::new(State { threads: threads.to_vec(), closed: false, active: true, adpf }),
        };
        session.set_target(duration_nanos);
        session
    }

    pub fn id(&self) -> u64 {
        self.id
    }

    fn set_target(&self, target_nanos: i64) {
        if target_nanos <= 0 {
            return;
        }
        self.laminar.set_target(target_nanos);
    }

    fn hint(&self, hint: SessionHint) {
        if self.debug {
            info!("session {} hint {hint:?}", self.id);
        }
        let mut state = self.state.lock().unwrap();
        if state.closed {
            return;
        }
        // CPU_LOAD_RESET must wake the session up if it is inactive.
        if hint == SessionHint::CPU_LOAD_RESET {
            state.active = true;
        }
        if !state.active {
            return;
        }
        match hint {
            SessionHint::CPU_LOAD_UP => return self.laminar.load_up(),
            SessionHint::CPU_LOAD_SPIKE => return self.laminar.load_spike(),
            SessionHint::CPU_LOAD_RESET => return self.laminar.rebaseline(),
            _ => {}
        }

        let State { threads, adpf, .. } = &mut *state;
        let Some(adpf) = adpf.as_mut() else { return };
        match hint {
            SessionHint::CPU_LOAD_DOWN => adpf.change_boost(threads, -TLB_STEP_PCT),
            SessionHint::CPU_LOAD_RESUME => adpf.release_boost(),
            SessionHint::GPU_LOAD_SPIKE => adpf.load_changed(GPU_BOOST_PCT + 100, 1),
            _ => {}
        }
    }

    fn close_session(&self) {
        let mut state = self.state.lock().unwrap();
        if state.closed {
            return;
        }
        self.laminar.reset();
        let State { threads, adpf, .. } = &mut *state;
        if let Some(adpf) = adpf.as_mut() {
            adpf.release_boost();
            adpf.release_all(threads);
        }
        state.closed = true;
        state.threads.clear();
    }
}

impl Drop for PowerHintSession {
    fn drop(&mut self) {
        self.close_session();
    }
}

impl Interface for PowerHintSession {}

#[allow(non_snake_case)]
impl IPowerHintSession for PowerHintSession {
    fn updateTargetWorkDuration(&self, targetDurationNanos: i64) -> binder::Result<()> {
        self.set_target(targetDurationNanos);
        Ok(())
    }

    fn reportActualWorkDuration(&self, durations: &[WorkDuration]) -> binder::Result<()> {
        // The first of the longest, as the QTI HAL picked it.
        if self.state.lock().unwrap().closed {
            return Ok(());
        }
        let Some(longest) = durations
            .iter()
            .reduce(|max, d| if d.durationNanos > max.durationNanos { d } else { max })
        else {
            return Ok(());
        };
        if self.debug {
            info!(
                "session {} actual {} ns, cpu {} ns, gpu {} ns",
                self.id, longest.durationNanos, longest.cpuDurationNanos, longest.gpuDurationNanos
            );
        }
        self.laminar.report(longest.durationNanos, longest.cpuDurationNanos, longest.gpuDurationNanos);
        Ok(())
    }

    fn pause(&self) -> binder::Result<()> {
        let mut state = self.state.lock().unwrap();
        if !state.closed {
            self.laminar.reset();
            state.active = false;
        }
        Ok(())
    }

    fn resume(&self) -> binder::Result<()> {
        let mut state = self.state.lock().unwrap();
        if !state.closed {
            state.active = true;
        }
        Ok(())
    }

    fn close(&self) -> binder::Result<()> {
        self.close_session();
        Ok(())
    }

    fn sendHint(&self, hint: SessionHint) -> binder::Result<()> {
        self.hint(hint);
        Ok(())
    }

    fn setThreads(&self, threadIds: &[i32]) -> binder::Result<()> {
        debug!("session {} threads {threadIds:?}", self.id);
        if threadIds.is_empty() {
            error!("Threads list is empty");
            return Err(Status::new_exception(ExceptionCode::ILLEGAL_ARGUMENT, None));
        }
        self.hint(SessionHint::CPU_LOAD_RESET);
        let mut state = self.state.lock().unwrap();
        let State { threads, adpf, .. } = &mut *state;
        if let Some(adpf) = adpf.as_mut() {
            adpf.release_all(threads);
        }
        *threads = threadIds.to_vec();
        self.laminar.set_threads(threadIds);
        Ok(())
    }

    fn setMode(&self, mode: SessionMode, enabled: bool) -> binder::Result<()> {
        debug!("session {} mode {mode:?} {enabled}", self.id);
        let mut state = self.state.lock().unwrap();
        let State { threads, adpf, .. } = &mut *state;
        let Some(adpf) = adpf.as_mut() else { return Ok(()) };
        match mode {
            SessionMode::POWER_EFFICIENCY => {
                adpf.release_low_cpu_util(threads);
                if enabled {
                    adpf.efficiency_threads =
                        adpf.hint_threads(threads, VENDOR_HINT_PICARD_HIGH_CPUUTIL, MAX_THREADS);
                }
            }
            SessionMode::GRAPHICS_PIPELINE => {
                adpf.release_pipeline(threads);
                if enabled {
                    let max = adpf_config().max_pipeline_threads;
                    adpf.pipeline_threads =
                        adpf.hint_threads(threads, VENDOR_HINT_PICARD_LOW_LAT, max);
                }
            }
            _ => {}
        }
        Ok(())
    }

    fn getSessionConfig(&self) -> binder::Result<SessionConfig> {
        Ok(SessionConfig { id: self.id as i64 })
    }
}

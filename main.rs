/*
 * SPDX-FileCopyrightText: The LineageOS Project
 * SPDX-License-Identifier: Apache-2.0
 */

//! Power HAL for QTI targets: hint sessions run Laminar frame-aware scheduling, modes and boosts
//! go to the perf HAL, the device extension (power_ext) or both.

// The test build has no caller for main() and the service type.
#![cfg_attr(test, allow(dead_code))]

mod floors;
mod gpu;
mod laminar;
mod perf;
mod session;

use android_hardware_power::aidl::android::hardware::power::{
    Boost::Boost,
    ChannelConfig::ChannelConfig,
    CompositionData::CompositionData,
    CompositionUpdate::CompositionUpdate,
    CpuHeadroomParams::CpuHeadroomParams,
    CpuHeadroomResult::CpuHeadroomResult,
    GpuHeadroomParams::{CalculationType::CalculationType, GpuHeadroomParams},
    GpuHeadroomResult::GpuHeadroomResult,
    IPower::{BnPower, IPower},
    IPowerHintSession::{BnPowerHintSession, IPowerHintSession},
    Mode::Mode,
    SessionConfig::SessionConfig,
    SessionTag::SessionTag,
    SupportInfo::{
        CompositionDataSupportInfo::CompositionDataSupportInfo,
        HeadroomSupportInfo::HeadroomSupportInfo, SupportInfo,
    },
};
use binder::{BinderFeatures, ExceptionCode, Interface, Status, Strong};
use log::{debug, error, info, LevelFilter};
use session::PowerHintSession;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};

const LOG_TAG: &str = "QTI PowerHAL";

const VENDOR_HINT_DISPLAY_OFF: i32 = 0x1040;
const VENDOR_HINT_DISPLAY_ON: i32 = 0x1041;
const PERF_HINT_EXPENSIVE_RENDERING: i32 = 0x10A5;
const VENDOR_FEEDBACK_GPU_HEADROOM: i32 = 0x1603;
const VENDOR_FEEDBACK_CPU_HEADROOM: i32 = 0x1610;

const GPU_HEADROOM_AVG: i32 = 2;
const GPU_HEADROOM_MIN: i32 = 3;
const CPU_HEADROOM_TOTAL: i32 = 6;
const HEADROOM_MIN_WINDOW_MS: i32 = 1000;
const HEADROOM_MAX_WINDOW_MS: i32 = 60000;

/// Extra Binder threads: sessions are independent, so one app's reports need not queue behind
/// another's.
const BINDER_THREADS: u32 = 2;

fn unsupported() -> Status {
    Status::new_exception(ExceptionCode::UNSUPPORTED_OPERATION, None)
}

fn expensive_rendering_supported() -> bool {
    static SUPPORTED: OnceLock<bool> = OnceLock::new();
    *SUPPORTED.get_or_init(|| perf::get_prop_int("vendor.perf.expensive_rendering", 0) == 1)
}

/// A headroom window in whole seconds, or None when it is longer than supported.
fn headroom_window_s(window_ms: i32) -> Option<i32> {
    (window_ms <= HEADROOM_MAX_WINDOW_MS).then(|| window_ms.max(HEADROOM_MIN_WINDOW_MS) / 1000)
}

/// Logs a headroom failure once, so a missing perf HAL feature does not flood the log.
fn headroom(feedback: i32, kind: i32, window_s: i32, failed: &AtomicBool) -> Option<f32> {
    let value = perf::get_feedback_extn(feedback, &mut [kind, window_s]);
    if value < 0 {
        if !failed.swap(true, Ordering::Relaxed) {
            error!("Failed to get headroom {feedback:#X}");
        }
        return None;
    }
    Some(value as f32)
}

struct Power {
    /// The perf lock of the expensive rendering hint while it is held.
    expensive_rendering: Mutex<Option<i32>>,
}

impl Power {
    fn set_expensive_rendering(&self, enabled: bool) {
        let mut handle = self.expensive_rendering.lock().unwrap();
        match (enabled, *handle) {
            (true, None) => {
                *handle = Some(perf::hint_enable(PERF_HINT_EXPENSIVE_RENDERING, 0)).filter(|&h| h > 0)
            }
            (false, Some(h)) => {
                perf::release(h);
                *handle = None;
            }
            _ => {}
        }
    }

    fn create_session(
        &self,
        threads: &[i32],
        duration_nanos: i64,
    ) -> binder::Result<(Strong<dyn IPowerHintSession>, u64)> {
        if threads.is_empty() {
            error!("Error: threadIds.size() shouldn't be 0");
            return Err(Status::new_exception(ExceptionCode::ILLEGAL_ARGUMENT, None));
        }
        let session = PowerHintSession::new(threads, duration_nanos);
        let id = session.id();
        Ok((BnPowerHintSession::new_binder(session, BinderFeatures::default()), id))
    }
}

impl Interface for Power {}

#[allow(non_snake_case)]
impl IPower for Power {
    fn setMode(&self, mode: Mode, enabled: bool) -> binder::Result<()> {
        debug!("setMode {mode:?} {enabled}");
        if power_ext::set_mode(mode, enabled) {
            return Ok(());
        }
        match mode {
            Mode::EXPENSIVE_RENDERING => {
                if expensive_rendering_supported() {
                    self.set_expensive_rendering(enabled);
                }
                laminar::set_expensive_rendering(enabled);
            }
            Mode::INTERACTIVE => {
                perf::hint_enable(
                    if enabled { VENDOR_HINT_DISPLAY_ON } else { VENDOR_HINT_DISPLAY_OFF },
                    0,
                );
            }
            _ => {}
        }
        Ok(())
    }

    fn isModeSupported(&self, mode: Mode) -> binder::Result<bool> {
        if let Some(supported) = power_ext::is_mode_supported(mode) {
            return Ok(supported);
        }
        Ok(matches!(
            mode,
            Mode::EXPENSIVE_RENDERING
                | Mode::INTERACTIVE
                | Mode::SUSTAINED_PERFORMANCE
                | Mode::FIXED_PERFORMANCE
        ))
    }

    fn setBoost(&self, boost: Boost, durationMs: i32) -> binder::Result<()> {
        power_ext::set_boost(boost, durationMs);
        match boost {
            // Sent for every batch of input events, so not logged.
            Boost::INTERACTION => laminar::interaction(durationMs),
            _ => debug!("setBoost {boost:?} {durationMs}"),
        }
        Ok(())
    }

    fn isBoostSupported(&self, boost: Boost) -> binder::Result<bool> {
        if let Some(supported) = power_ext::is_boost_supported(boost) {
            return Ok(supported);
        }
        Ok(boost == Boost::INTERACTION)
    }

    fn createHintSession(
        &self,
        _tgid: i32,
        _uid: i32,
        threadIds: &[i32],
        durationNanos: i64,
    ) -> binder::Result<Strong<dyn IPowerHintSession>> {
        Ok(self.create_session(threadIds, durationNanos)?.0)
    }

    fn getHintSessionPreferredRate(&self) -> binder::Result<i64> {
        Ok(session::PREFERRED_RATE_NANOS)
    }

    fn createHintSessionWithConfig(
        &self,
        _tgid: i32,
        _uid: i32,
        threadIds: &[i32],
        durationNanos: i64,
        tag: SessionTag,
        config: &mut SessionConfig,
    ) -> binder::Result<Strong<dyn IPowerHintSession>> {
        if !matches!(
            tag,
            SessionTag::OTHER | SessionTag::SURFACEFLINGER | SessionTag::HWUI | SessionTag::GAME
        ) {
            return Err(unsupported());
        }
        let (session, id) = self.create_session(threadIds, durationNanos)?;
        config.id = id as i64;
        Ok(session)
    }

    fn getSessionChannel(&self, _tgid: i32, _uid: i32) -> binder::Result<ChannelConfig> {
        // Sessions report work durations through Binder so they reach the hint sessions; a
        // channel nobody reads would drop them.
        Err(unsupported())
    }

    fn closeSessionChannel(&self, _tgid: i32, _uid: i32) -> binder::Result<()> {
        Ok(())
    }

    fn getSupportInfo(&self) -> binder::Result<SupportInfo> {
        // Each bit is the support of that value of the enum, from bit 0.
        Ok(SupportInfo {
            usesSessions: true,
            boosts: 0,
            modes: 0b10001100,
            sessionHints: 0b1100001111,
            sessionModes: 0b0011,
            sessionTags: 0b001111,
            compositionData: CompositionDataSupportInfo {
                isSupported: false,
                disableGpuFences: false,
                maxBatchSize: 1,
                alwaysBatch: false,
            },
            headroom: HeadroomSupportInfo {
                isCpuSupported: true,
                isGpuSupported: true,
                cpuMinIntervalMillis: 1000,
                gpuMinIntervalMillis: 1000,
                cpuMinCalculationWindowMillis: 50,
                cpuMaxCalculationWindowMillis: HEADROOM_MAX_WINDOW_MS,
                gpuMinCalculationWindowMillis: 50,
                gpuMaxCalculationWindowMillis: HEADROOM_MAX_WINDOW_MS,
                cpuMaxTidCount: 5,
            },
        })
    }

    fn getCpuHeadroom(&self, params: &CpuHeadroomParams) -> binder::Result<CpuHeadroomResult> {
        static FAILED: AtomicBool = AtomicBool::new(false);
        let window_s = headroom_window_s(params.calculationWindowMillis).ok_or_else(unsupported)?;
        let value = headroom(VENDOR_FEEDBACK_CPU_HEADROOM, CPU_HEADROOM_TOTAL, window_s, &FAILED);
        // An unset result tells the caller there is no headroom value.
        Ok(value.map_or_else(Default::default, CpuHeadroomResult::GlobalHeadroom))
    }

    fn getGpuHeadroom(&self, params: &GpuHeadroomParams) -> binder::Result<GpuHeadroomResult> {
        static FAILED: AtomicBool = AtomicBool::new(false);
        let kind = match params.calculationType {
            CalculationType::MIN => GPU_HEADROOM_MIN,
            CalculationType::AVERAGE => GPU_HEADROOM_AVG,
            _ => return Err(unsupported()),
        };
        let window_s = headroom_window_s(params.calculationWindowMillis).ok_or_else(unsupported)?;
        let value = headroom(VENDOR_FEEDBACK_GPU_HEADROOM, kind, window_s, &FAILED);
        Ok(value.map_or_else(Default::default, GpuHeadroomResult::GlobalHeadroom))
    }

    fn sendCompositionData(&self, _data: &[CompositionData]) -> binder::Result<()> {
        Ok(())
    }

    fn sendCompositionUpdate(&self, _update: &CompositionUpdate) -> binder::Result<()> {
        Ok(())
    }
}

fn main() {
    logger::init(
        logger::Config::default().with_tag_on_device(LOG_TAG).with_max_level(LevelFilter::Info),
    );

    binder::ProcessState::set_thread_pool_max_thread_count(BINDER_THREADS);
    binder::ProcessState::start_thread_pool();

    let power = Power { expensive_rendering: Mutex::new(None) };
    let service = BnPower::new_binder(power, BinderFeatures::default());
    let instance = format!("{}/default", <Power as IPower>::get_descriptor());
    info!("Instance {instance}");
    binder::add_service(&instance, service.as_binder()).expect("Failed to register service");

    binder::ProcessState::join_thread_pool();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn headroom_window_bounds() {
        assert_eq!(headroom_window_s(50), Some(1));
        assert_eq!(headroom_window_s(2500), Some(2));
        assert_eq!(headroom_window_s(HEADROOM_MAX_WINDOW_MS), Some(60));
        assert_eq!(headroom_window_s(HEADROOM_MAX_WINDOW_MS + 1), None);
    }
}

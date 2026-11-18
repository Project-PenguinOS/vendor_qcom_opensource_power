/*
 * SPDX-FileCopyrightText: The LineageOS Project
 * SPDX-License-Identifier: Apache-2.0
 */

//! Device power extension for targets without one. A device extension is a crate named
//! power_ext with these functions, selected with the qti_power/mode_ext soong config.

use android_hardware_power::aidl::android::hardware::power::{Boost::Boost, Mode::Mode};

/// Some(supported) when the device decides whether `mode` is supported.
pub fn is_mode_supported(_mode: Mode) -> Option<bool> {
    None
}

/// Returns true when the device handled `mode` and the common handling must be skipped.
pub fn set_mode(_mode: Mode, _enabled: bool) -> bool {
    false
}

/// Some(supported) when the device decides whether `boost` is supported.
pub fn is_boost_supported(_boost: Boost) -> Option<bool> {
    None
}

/// Runs before the common boost handling, which always follows.
pub fn set_boost(_boost: Boost, _duration_ms: i32) {}

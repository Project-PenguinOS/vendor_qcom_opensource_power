/*
 * SPDX-FileCopyrightText: The LineageOS Project
 * SPDX-License-Identifier: Apache-2.0
 */

//! The QTI perf HAL client, loaded at runtime like the QTI power HAL did.

use log::error;
use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::sync::{Mutex, MutexGuard, OnceLock};

const PERF_CLIENT: &CStr = c"libqti-perfd-client.so";
pub const PKG: &CStr = c"QTI PowerHAL";
const PROP_VALUE_MAX: usize = 92;

#[repr(C)]
struct PropVal {
    value: [c_char; PROP_VALUE_MAX],
}

struct Client {
    lock_acq: Option<extern "C" fn(c_int, c_int, *mut c_int, c_int) -> c_int>,
    lock_rel: Option<extern "C" fn(c_int) -> c_int>,
    hint: Option<extern "C" fn(c_int, *const c_char, c_int, c_int) -> c_int>,
    get_feedback: Option<extern "C" fn(c_int, *const c_char) -> c_int>,
    get_feedback_extn: Option<extern "C" fn(c_int, *const c_char, c_int, *mut c_int) -> c_int>,
    sync_request: Option<extern "C" fn(c_int) -> *const c_char>,
    get_prop: Option<extern "C" fn(*const c_char, *const c_char) -> PropVal>,
}

fn symbol<T>(handle: *mut c_void, name: &CStr) -> Option<T> {
    // SAFETY: `handle` is a live dlopen handle and `name` is NUL terminated.
    let address = unsafe { libc::dlsym(handle, name.as_ptr()) };
    if address.is_null() {
        error!("Unable to get the {name:?} function handle");
        return None;
    }
    // SAFETY: T is the extern "C" function pointer type the perf client exports for `name`.
    Some(unsafe { std::mem::transmute_copy(&address) })
}

/// The QTI power HAL only ever called the perf client from one thread; keep it that way.
fn client() -> (MutexGuard<'static, ()>, &'static Client) {
    static LOCK: Mutex<()> = Mutex::new(());
    (LOCK.lock().unwrap(), loaded())
}

fn loaded() -> &'static Client {
    static CLIENT: OnceLock<Client> = OnceLock::new();
    CLIENT.get_or_init(|| {
        // SAFETY: plain dlopen of a vendor library; the handle is never closed.
        let handle = unsafe { libc::dlopen(PERF_CLIENT.as_ptr(), libc::RTLD_NOW) };
        if handle.is_null() {
            error!("Unable to open {PERF_CLIENT:?}");
            return Client {
                lock_acq: None,
                lock_rel: None,
                hint: None,
                get_feedback: None,
                get_feedback_extn: None,
                sync_request: None,
                get_prop: None,
            };
        }
        Client {
            lock_acq: symbol(handle, c"perf_lock_acq"),
            lock_rel: symbol(handle, c"perf_lock_rel"),
            hint: symbol(handle, c"perf_hint"),
            get_feedback: symbol(handle, c"perf_get_feedback"),
            get_feedback_extn: symbol(handle, c"perf_get_feedback_extn"),
            sync_request: symbol(handle, c"perf_sync_request"),
            get_prop: symbol(handle, c"perf_get_prop"),
        }
    })
}

fn c_string(value: &str) -> CString {
    CString::new(value).unwrap_or_default()
}

/// A perf hint for `pkg`. Returns the lock handle, or -1.
pub fn hint(hint_id: i32, pkg: &str, duration: i32, kind: i32) -> i32 {
    let (_lock, client) = client();
    let Some(f) = client.hint else { return -1 };
    let handle = f(hint_id, c_string(pkg).as_ptr(), duration, kind);
    if handle < 0 {
        error!("Failed to send perf hint for hint_id: {hint_id:#X}");
    }
    handle
}

/// A perf hint on behalf of the power HAL itself.
pub fn hint_enable(hint_id: i32, duration: i32) -> i32 {
    let (_lock, client) = client();
    let Some(f) = client.hint else { return 0 };
    let handle = f(hint_id, PKG.as_ptr(), duration, -1);
    if handle == -1 {
        error!("Failed to acquire lock for hint_id: {hint_id:#X}");
    }
    handle
}

pub fn get_feedback(hint_id: i32, pkg: &str) -> i32 {
    let (_lock, client) = client();
    match client.get_feedback {
        Some(f) => f(hint_id, c_string(pkg).as_ptr()),
        None => -1,
    }
}

pub fn get_feedback_extn(hint_id: i32, args: &mut [i32]) -> i32 {
    let (_lock, client) = client();
    match client.get_feedback_extn {
        Some(f) => f(hint_id, PKG.as_ptr(), args.len() as c_int, args.as_mut_ptr()),
        None => -1,
    }
}

pub fn sync_request(hint_id: i32) -> Option<String> {
    let (_lock, client) = client();
    let f = client.sync_request?;
    let reply = f(hint_id);
    if reply.is_null() {
        error!("Failed to send perf sync request for hint_id: {hint_id:#X}");
        return None;
    }
    // SAFETY: the perf client returns a malloc'd NUL terminated string the caller frees.
    let value = unsafe { CStr::from_ptr(reply) }.to_string_lossy().into_owned();
    // SAFETY: as above; `reply` is not used after this.
    unsafe { libc::free(reply as *mut c_void) };
    Some(value)
}

/// Acquires (or, with a valid `handle`, renews) a perf lock on `resources`.
pub fn lock_acquire(handle: i32, duration: i32, resources: &mut [i32]) -> i32 {
    let (_lock, client) = client();
    let Some(f) = client.lock_acq else { return handle };
    let handle = f(handle, duration, resources.as_mut_ptr(), resources.len() as c_int);
    if handle == -1 {
        error!("Failed to acquire lock");
    }
    handle
}

pub fn release(handle: i32) {
    let (_lock, client) = client();
    if let Some(f) = client.lock_rel {
        f(handle);
    }
}

/// A property from the perf config store, which overrides the system property.
pub fn get_prop(name: &str, default: &str) -> String {
    let (_lock, client) = client();
    let Some(f) = client.get_prop else { return default.to_owned() };
    let value = f(c_string(name).as_ptr(), c_string(default).as_ptr());
    let bytes: Vec<u8> = value.value.iter().take_while(|&&c| c != 0).map(|&c| c as u8).collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

pub fn get_prop_int(name: &str, default: i32) -> i32 {
    get_prop(name, &default.to_string()).trim().parse().unwrap_or(0)
}

/*
 * Copyright (C) 2026 The PenguinOS Project
 * SPDX-License-Identifier: Apache-2.0
 */

//! The shared GPU frequency floor. Each owner votes for a step that expires on its own; the
//! floor is the highest live vote, written to the KGSL devfreq minimum.

use log::{info, warn};
use std::collections::HashMap;
use std::fs;
use std::sync::{Condvar, Mutex, OnceLock};
use std::thread;
use std::time::{Duration, Instant};

const MIN_FREQ: &str = "/sys/class/kgsl/kgsl-3d0/devfreq/min_freq";
const FREQUENCY_TABLES: [&str; 2] = [
    "/sys/class/kgsl/kgsl-3d0/gpu_available_frequencies",
    "/sys/class/kgsl/kgsl-3d0/devfreq/available_frequencies",
];
// Start GPU votes at a share of this device's maximum clock, then climb its supported levels.
// Frequencies themselves are discovered from the driver, never copied from another SoC.
const MIN_FLOOR_PERCENT: i64 = 45;
/// Each vote expires on its own, including when CPU-only reports keep a session alive.
pub const DEFAULT_LEASE: Duration = Duration::from_millis(250);

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Owner {
    Session(u64),
    ExpensiveRendering,
    Interaction,
}

/// Index 0 is no vote: the governor picks the clock. Read once, the table does not change.
pub fn steps_hz() -> &'static [i64] {
    static STEPS: OnceLock<Vec<i64>> = OnceLock::new();
    STEPS.get_or_init(|| {
        for path in FREQUENCY_TABLES {
            let Ok(contents) = fs::read_to_string(path) else { continue };
            let Some(levels) = levels_from_table(&contents) else { continue };
            info!("GPU floor uses {} driver levels up to {} Hz", levels.len() - 1, levels.last().unwrap());
            return levels;
        }
        warn!("No readable GPU frequency table; keeping CPU hints without GPU votes");
        vec![0]
    })
}

/// The driver frequencies from MIN_FLOOR_PERCENT of the maximum up, after a leading 0.
pub fn levels_from_table(contents: &str) -> Option<Vec<i64>> {
    let mut frequencies: Vec<i64> = contents
        .split_whitespace()
        .filter_map(|f| f.parse().ok())
        .filter(|&f| f > 0 && f <= 10_000_000_000)
        .collect();
    if frequencies.is_empty() {
        return None;
    }
    frequencies.sort_unstable();
    frequencies.dedup();
    let minimum = (frequencies.last().unwrap() * MIN_FLOOR_PERCENT + 99) / 100;
    let mut levels = vec![0];
    levels.extend(frequencies.into_iter().filter(|&f| f >= minimum));
    Some(levels)
}

#[derive(Default)]
struct State {
    votes: HashMap<Owner, (usize, Instant)>,
    applied: usize,
    /// The floor someone else had set before our first vote, restored after our last one.
    baseline: Option<String>,
    write_failed: bool,
}

impl State {
    fn top(&self) -> usize {
        self.votes.values().map(|&(step, _)| step).max().unwrap_or(0)
    }

    fn apply(&mut self) {
        let top = self.top();
        if top == self.applied {
            if top == 0 {
                self.baseline = None;
            }
            return;
        }
        if self.applied == 0 && self.baseline.is_none() {
            match fs::read_to_string(MIN_FREQ) {
                Ok(baseline) => self.baseline = Some(baseline),
                Err(e) => {
                    if !self.write_failed {
                        warn!("Cannot read GPU minimum frequency: {e}");
                    }
                    self.write_failed = true;
                    return;
                }
            }
        }
        let baseline = self.baseline.clone().unwrap_or_default();
        // Do not lower a higher floor that another owner had already requested.
        let value = if top == 0 {
            baseline
        } else {
            steps_hz()[top].max(baseline.trim().parse().unwrap_or(0)).to_string()
        };
        if let Err(e) = fs::write(MIN_FREQ, &value) {
            if !self.write_failed {
                warn!("Cannot write GPU minimum frequency: {e}");
            }
            self.write_failed = true;
            return;
        }
        self.write_failed = false;
        self.applied = top;
        if top == 0 {
            self.baseline = None;
        }
    }
}

struct Votes {
    state: Mutex<State>,
    changed: Condvar,
}

fn votes() -> &'static Votes {
    static VOTES: OnceLock<Votes> = OnceLock::new();
    VOTES.get_or_init(|| {
        thread::Builder::new()
            .name("laminar-gpu".to_owned())
            .spawn(lease_loop)
            .expect("Failed to start the GPU lease thread");
        Votes { state: Mutex::default(), changed: Condvar::new() }
    })
}

fn lease_loop() {
    let votes = votes();
    let mut state = votes.state.lock().unwrap();
    loop {
        let now = Instant::now();
        state.votes.retain(|_, &mut (_, deadline)| deadline > now);
        state.apply();
        if state.votes.is_empty() && state.applied == 0 {
            state = votes.changed.wait(state).unwrap();
        } else {
            // Also retry a failed release; an I/O error must not strand a high floor.
            let next = state
                .votes
                .values()
                .map(|&(_, deadline)| deadline)
                .fold(now + DEFAULT_LEASE, Instant::min);
            let timeout = next.saturating_duration_since(now);
            state = votes.changed.wait_timeout(state, timeout).unwrap().0;
        }
    }
}

/// Votes `step` for `owner` for `lease`; step 0 withdraws the vote.
pub fn vote(owner: Owner, step: usize, lease: Duration) {
    let votes = votes();
    let mut state = votes.state.lock().unwrap();
    if step > 0 && step < steps_hz().len() {
        state.votes.insert(owner, (step, Instant::now() + lease));
    } else {
        state.votes.remove(&owner);
    }
    state.apply();
    votes.changed.notify_one();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn levels_start_at_share_of_max() {
        let table = "1100000000 220000000 660000000 480000000 660000000 0 junk";
        assert_eq!(levels_from_table(table).unwrap(), vec![0, 660_000_000, 1_100_000_000]);
        assert!(levels_from_table("").is_none());
    }

    #[test]
    fn top_vote_wins() {
        let mut state = State::default();
        let later = Instant::now() + Duration::from_secs(1);
        state.votes.insert(Owner::Session(1), (2, later));
        state.votes.insert(Owner::Interaction, (1, later));
        assert_eq!(state.top(), 2);
        state.votes.remove(&Owner::Session(1));
        assert_eq!(state.top(), 1);
    }
}

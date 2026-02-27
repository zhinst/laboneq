// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;

use laboneq_dsl::types::SignalUid;

#[derive(Debug)]
pub(super) struct SignalDelay {
    pub signal_uid: SignalUid,
    pub delay_samples: i64,
    pub sampling_rate: f64,
    pub sample_multiple: u16,
}

#[derive(Debug)]
pub(super) struct OnDeviceDelay {
    pub signal_uid: SignalUid,
    pub on_port: f64,
    pub on_signal: f64,
}

fn gcd_reduce(numbers: &[u64]) -> u64 {
    use num_integer::gcd;
    numbers.iter().cloned().reduce(gcd).unwrap_or(1)
}

/// Compute on-device delays for signals based on their delays in samples,
/// sampling rates, and sample multiples.
///
/// The delays are split into port delays and signal delays, which are adjusted to
/// align with the device's sequencer grid.
pub(super) fn compute_on_device_delays(
    delays: Vec<SignalDelay>,
) -> impl Iterator<Item = OnDeviceDelay> {
    // Collect unique sequencer rates and find the maximum delay in seconds
    let mut unique_sequencer_rates = HashSet::new();
    let mut max_delay_seconds = 0f64;
    for signal_delay in &delays {
        let sequencer_rate = signal_delay.sampling_rate / (signal_delay.sample_multiple as f64);
        unique_sequencer_rates.insert(sequencer_rate as u64);
        let delay = signal_delay.delay_samples as f64 / signal_delay.sampling_rate;
        max_delay_seconds = max_delay_seconds.max(delay);
    }

    // Calculate max delay aligned to system grid
    let rates: Vec<u64> = unique_sequencer_rates.into_iter().collect();
    let common_sequencer_rate = gcd_reduce(&rates) as f64;
    let system_grid = 1.0 / common_sequencer_rate;
    let max_delay = (max_delay_seconds / system_grid).ceil() * system_grid;

    // Compute on-device delays for each signal
    delays.into_iter().map(move |info| {
        let max_delay_samples = max_delay * info.sampling_rate;
        let initial_delay_samples = max_delay_samples - info.delay_samples as f64;

        // Split into signal and port delays
        let delay_signal_samples = (initial_delay_samples / info.sample_multiple as f64).floor();
        let delay_signal = delay_signal_samples / info.sampling_rate * info.sample_multiple as f64;

        let port_delay_samples = initial_delay_samples % info.sample_multiple as f64;
        let port_delay = port_delay_samples / info.sampling_rate;

        // Apply threshold for very small values
        let delay_signal = if delay_signal > 1e-12 {
            delay_signal
        } else {
            0.0
        };
        let port_delay = if port_delay > 1e-12 { port_delay } else { 0.0 };
        OnDeviceDelay {
            signal_uid: info.signal_uid,
            on_port: port_delay,
            on_signal: delay_signal,
        }
    })
}

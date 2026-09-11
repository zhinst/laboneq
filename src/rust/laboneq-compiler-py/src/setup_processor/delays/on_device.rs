// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::num::NonZeroU16;

/// A signal's own delay, and the grid its sequencer runs on.
#[derive(Debug)]
pub(super) struct SignalDelay {
    /// The delay the signal already carries, in samples at `sampling_rate`.
    pub delay_samples: i64,
    pub sampling_rate: f64,
    /// The sequencer grid, in samples at `sampling_rate`.
    pub sample_multiple: NonZeroU16,
}

/// A wait in seconds, split into the part a sequencer can time and the part it
/// cannot.
#[derive(Debug)]
pub(super) struct OnDeviceDelay {
    /// Whole sequencer grid steps.
    pub on_grid: f64,
    /// The sub-grid remainder, which the device has to realise elsewhere.
    pub sub_grid: f64,
}

/// How long each signal has to wait so that every signal starts together.
///
/// The largest delay sets the start, rounded up to a grid every sequencer can
/// land on — the reciprocal of the greatest common divisor of their sequencer
/// rates. Each signal then waits the difference between that start and its own
/// delay.
///
/// Yields one delay per input, in the same order.
pub(super) fn compute_on_device_delays(
    delays: Vec<SignalDelay>,
) -> impl Iterator<Item = OnDeviceDelay> {
    let sequencer_rates = delays
        .iter()
        .map(|delay| (delay.sampling_rate / delay.sample_multiple.get() as f64) as u64);
    let system_grid = 1.0 / gcd_reduce(sequencer_rates) as f64;

    let max_delay_seconds = delays
        .iter()
        .map(|delay| delay.delay_samples as f64 / delay.sampling_rate)
        .fold(0.0, f64::max);
    let max_delay = (max_delay_seconds / system_grid).ceil() * system_grid;

    delays.into_iter().map(move |delay| {
        let wait_samples = max_delay * delay.sampling_rate - delay.delay_samples as f64;
        let grid = delay.sample_multiple.get() as f64;

        OnDeviceDelay {
            on_grid: threshold((wait_samples / grid).floor() / delay.sampling_rate * grid),
            sub_grid: threshold((wait_samples % grid) / delay.sampling_rate),
        }
    })
}

/// Values below a picosecond are rounding noise rather than a delay.
fn threshold(seconds: f64) -> f64 {
    if seconds > 1e-12 { seconds } else { 0.0 }
}

fn gcd_reduce(numbers: impl IntoIterator<Item = u64>) -> u64 {
    numbers.into_iter().reduce(num_integer::gcd).unwrap_or(1)
}

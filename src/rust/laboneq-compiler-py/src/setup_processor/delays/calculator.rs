// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::num::NonZeroU16;

use laboneq_dsl::types::SignalUid;
use laboneq_units::duration::{Duration, Second};

use super::on_device::{SignalDelay, compute_on_device_delays};

/// The delays that line the signals up, looked up by signal.
#[derive(Debug)]
pub(crate) struct DelayRegistry {
    signal_delays: HashMap<SignalUid, SignalDelayInfo>,
}

impl DelayRegistry {
    /// The part of the wait before the signal starts that fits the sequencer
    /// grid, plus the head start the signal's instrument needs.
    pub(crate) fn signal_start_delay(&self, signal: SignalUid) -> Duration<Second> {
        self.signal_delays
            .get(&signal)
            .map_or(0.0.into(), |info| info.start_delay)
    }

    /// The part of the wait that does not fit the signal's sequencer grid.
    pub(crate) fn signal_alignment_delay(&self, signal: SignalUid) -> Duration<Second> {
        self.signal_delays
            .get(&signal)
            .map_or(0.0.into(), |info| info.alignment_delay)
    }

    fn new() -> Self {
        Self {
            signal_delays: HashMap::new(),
        }
    }

    fn add_delay(
        &mut self,
        signal: SignalUid,
        start_delay: Duration<Second>,
        alignment_delay: Duration<Second>,
    ) {
        self.signal_delays.insert(
            signal,
            SignalDelayInfo {
                start_delay,
                alignment_delay,
            },
        );
    }
}

#[derive(Debug)]
struct SignalDelayInfo {
    start_delay: Duration<Second>,
    alignment_delay: Duration<Second>,
}

/// What must be known about one signal to line it up against the others.
pub(crate) struct SignalDelayProperties {
    pub(crate) uid: SignalUid,
    pub(crate) sampling_rate: f64,
    /// The sequencer grid, in samples at `sampling_rate`.
    pub(crate) sample_multiple: NonZeroU16,
    /// The delay the signal already carries, in samples at `sampling_rate`.
    pub(crate) delay_signal: i64,
    /// The head start the signal's instrument needs before it can play.
    pub(crate) lead_delay: Duration<Second>,
}

/// Compute the delay that lines each signal up with the others.
pub(crate) fn compute_signal_delays(signals: &[SignalDelayProperties]) -> DelayRegistry {
    let own_delays = signals
        .iter()
        .map(|signal| SignalDelay {
            delay_samples: signal.delay_signal,
            sampling_rate: signal.sampling_rate,
            sample_multiple: signal.sample_multiple,
        })
        .collect::<Vec<_>>();

    let mut delays = DelayRegistry::new();
    for (signal, wait) in signals.iter().zip(compute_on_device_delays(own_delays)) {
        delays.add_delay(
            signal.uid,
            (wait.on_grid + signal.lead_delay.value()).into(),
            wait.sub_grid.into(),
        );
    }
    delays
}

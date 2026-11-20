// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use crate::utils::SignalGridInfo;
use laboneq_common::types::AwgKey;
use laboneq_dsl::types::{AmplifierPump, Oscillator, SignalUid};
use laboneq_units::duration::{Duration, Frequency, Hertz, Second};

/// Everything the scheduler needs to know about a signal.
///
/// Hardware properties are exposed one by one, so the scheduler never sees the device.
pub trait SignalInfo {
    fn uid(&self) -> SignalUid;
    fn awg_key(&self) -> AwgKey;
    fn sampling_rate(&self) -> f64;
    /// Rate at which the sequencer issues instructions.
    fn sequencer_rate(&self) -> f64;
    fn oscillator_set_latency(&self) -> Duration<Second>;
    fn oscillator_reset_duration(&self) -> Duration<Second>;
    fn lo_frequency_granularity(&self) -> Option<Frequency<Hertz>>;
    fn oscillator(&self) -> Option<&Oscillator>;
    fn supports_initial_oscillator_frequency(&self) -> bool;
    fn amplifier_pump(&self) -> Option<&AmplifierPump>;
    fn supports_multiple_acquisition_lengths(&self) -> bool;
}

impl<T: SignalInfo> SignalGridInfo for T {
    fn uid(&self) -> SignalUid {
        self.uid()
    }

    fn sampling_rate(&self) -> f64 {
        self.sampling_rate()
    }

    fn sequencer_rate(&self) -> f64 {
        self.sequencer_rate()
    }
}

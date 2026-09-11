// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use laboneq_common::device_traits::DeviceTraits;
use laboneq_common::types::{AwgKey, PhysicalDeviceUid, SignalKind};
use laboneq_dsl::types::{
    AmplifierPump, DeviceUid, Oscillator, OscillatorKind, SignalUid, ValueOrParameter,
};
use laboneq_ir::signal::Signal;
use laboneq_scheduler::SignalInfo;

use laboneq_units::duration::{Duration, Frequency, Hertz, Second};

/// A view over a signal.
///
/// Provides convenient access to signal and device properties.
#[derive(Debug, Clone, PartialEq, Copy)]
pub struct SignalView<'a> {
    signal: &'a Signal,
}

impl SignalView<'_> {
    pub(crate) fn new(signal: &Signal) -> SignalView<'_> {
        SignalView { signal }
    }

    pub fn uid(&self) -> SignalUid {
        self.signal.uid
    }

    pub fn awg_key(&self) -> &AwgKey {
        &self.signal.awg_key
    }

    pub(crate) fn automute(&self) -> bool {
        self.signal.automute
    }

    pub fn device_uid(&self) -> DeviceUid {
        self.signal.device_uid
    }

    /// Physical device this signal's device maps to.
    /// This UID is used to group virtual devices that share the same
    /// physical hardware, enabling proper device detection.
    pub fn physical_device_uid(&self) -> PhysicalDeviceUid {
        self.signal.physical_device_uid
    }

    pub fn device_traits(&self) -> &'static DeviceTraits {
        self.signal.traits
    }

    pub fn sampling_rate(&self) -> f64 {
        self.signal.sampling_rate
    }

    pub(crate) fn oscillator(&self) -> Option<&Oscillator> {
        self.signal.oscillator.as_ref()
    }

    pub(crate) fn lo_frequency(&self) -> Option<&ValueOrParameter<f64>> {
        self.signal.lo_frequency.as_ref()
    }

    pub(crate) fn voltage_offset(&self) -> Option<&ValueOrParameter<f64>> {
        self.signal.voltage_offset.as_ref()
    }

    pub fn signal_kind(&self) -> &SignalKind {
        &self.signal.kind
    }

    pub(crate) fn amplifier_pump(&self) -> Option<&AmplifierPump> {
        self.signal.amplifier_pump.as_ref()
    }

    pub(crate) fn is_hardware_modulated(&self) -> bool {
        self.signal
            .oscillator
            .as_ref()
            .is_some_and(|osc| matches!(osc.kind, OscillatorKind::Hardware))
    }

    pub fn signal_delay(&self) -> Duration<Second> {
        self.signal.signal_delay
    }

    pub fn port_delay(&self) -> Option<&ValueOrParameter<Duration<Second>>> {
        self.signal.port_delay.as_ref()
    }

    pub fn start_delay(&self) -> Duration<Second> {
        self.signal.start_delay
    }
}

impl SignalInfo for SignalView<'_> {
    fn uid(&self) -> SignalUid {
        self.uid()
    }

    fn awg_key(&self) -> AwgKey {
        *self.awg_key()
    }

    fn sampling_rate(&self) -> f64 {
        self.sampling_rate()
    }

    fn sequencer_rate(&self) -> f64 {
        self.sampling_rate() / self.device_traits().sample_multiple as f64
    }

    fn oscillator_set_latency(&self) -> Duration<Second> {
        self.device_traits().oscillator_set_latency
    }

    fn oscillator_reset_duration(&self) -> Duration<Second> {
        self.device_traits().oscillator_reset_duration
    }

    fn lo_frequency_granularity(&self) -> Option<Frequency<Hertz>> {
        self.device_traits().lo_frequency_granularity
    }

    fn oscillator(&self) -> Option<&Oscillator> {
        self.oscillator()
    }

    fn lo_frequency(&self) -> Option<&ValueOrParameter<f64>> {
        self.lo_frequency()
    }

    fn supports_initial_oscillator_frequency(&self) -> bool {
        self.device_traits().supports_initial_oscillator_frequency
    }

    fn voltage_offset(&self) -> Option<&ValueOrParameter<f64>> {
        self.voltage_offset()
    }

    fn supports_initial_voltage_offset(&self) -> bool {
        self.device_traits().supports_initial_voltage_offset
    }

    fn amplifier_pump(&self) -> Option<&AmplifierPump> {
        self.amplifier_pump()
    }

    fn supports_multiple_acquisition_lengths(&self) -> bool {
        self.device_traits().supports_multiple_acquisition_lengths
    }
}

/// Build signal views for all signals of the setup.
pub(crate) fn signal_views(signals: &[Signal]) -> HashMap<SignalUid, SignalView<'_>> {
    signals
        .iter()
        .map(|signal| (signal.uid, SignalView::new(signal)))
        .collect()
}

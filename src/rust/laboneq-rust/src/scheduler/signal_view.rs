// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use laboneq_common::device_traits::DeviceTraits;
use laboneq_common::types::{AwgKey, DeviceKind};
use laboneq_scheduler::SignalInfo;
use laboneq_scheduler::experiment::types::{
    AmplifierPump, DeviceUid, Oscillator, OscillatorKind, SignalUid, ValueOrParameter,
};

use crate::scheduler::experiment::{Device, DeviceSetup, PortMode, Signal, SignalKind};

/// A view over a signal and its associated device.
///
/// Provides convenient access to signal and device properties.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct SignalView<'a> {
    device: &'a Device,
    signal: &'a Signal,
}

impl SignalView<'_> {
    pub(crate) fn new<'a>(device: &'a Device, signal: &'a Signal) -> SignalView<'a> {
        SignalView { device, signal }
    }

    pub(crate) fn device(&self) -> &Device {
        self.device
    }

    pub(crate) fn uid(&self) -> SignalUid {
        self.signal.uid
    }

    pub(crate) fn awg_key(&self) -> &AwgKey {
        &self.signal.awg_key
    }

    pub(crate) fn automute(&self) -> bool {
        self.signal.automute
    }

    pub(crate) fn device_kind(&self) -> &DeviceKind {
        &self.device.kind
    }

    pub(crate) fn device_uid(&self) -> DeviceUid {
        self.device.uid
    }

    pub(crate) fn sampling_rate(&self) -> f64 {
        self.signal.sampling_rate
    }

    pub(crate) fn channels(&self) -> &[u16] {
        &self.signal.channels
    }

    pub(crate) fn port_mode(&self) -> Option<&PortMode> {
        self.signal.port_mode.as_ref()
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

    pub(crate) fn signal_kind(&self) -> &SignalKind {
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

    fn device_traits(&self) -> &'static DeviceTraits {
        DeviceTraits::from_device_kind(self.device_kind())
    }

    fn oscillator(&self) -> Option<&Oscillator> {
        self.oscillator()
    }

    fn lo_frequency(&self) -> Option<&ValueOrParameter<f64>> {
        self.lo_frequency()
    }

    fn supports_initial_local_oscillator_frequency(&self) -> bool {
        DeviceTraits::from_device_kind(self.device_kind()).device_class != 0
    }

    fn voltage_offset(&self) -> Option<&ValueOrParameter<f64>> {
        self.voltage_offset()
    }

    fn supports_initial_voltage_offset(&self) -> bool {
        DeviceTraits::from_device_kind(self.device_kind()).device_class != 0
    }

    fn amplifier_pump(&self) -> Option<&AmplifierPump> {
        self.amplifier_pump()
    }

    fn supports_multiple_acquisition_lengths(&self) -> bool {
        matches!(self.device_kind(), DeviceKind::PrettyPrinterDevice)
    }
}

/// Build signal views for all signals in the device setup.
///
/// This function will panic if any signal or device is not found.
pub(crate) fn signal_views<'a>(
    device_setup: &'a DeviceSetup,
) -> HashMap<SignalUid, SignalView<'a>> {
    device_setup
        .signals
        .keys()
        .map(|uid| (*uid, build_signal_view(device_setup, uid)))
        .collect()
}

fn build_signal_view<'a>(device_setup: &'a DeviceSetup, signal_uid: &SignalUid) -> SignalView<'a> {
    let signal = device_setup
        .signals
        .get(signal_uid)
        .expect("Signal not found");
    let device = device_setup
        .devices
        .get(&signal.device_uid)
        .expect("Device not found");
    SignalView::new(device, signal)
}

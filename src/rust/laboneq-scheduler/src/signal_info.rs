// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

//! This module defines signal information that is required by
//! the Scheduler.
use crate::utils::SignalGridInfo;
use laboneq_common::{device_traits::DeviceTraits, types::AwgKey};
use laboneq_dsl::types::{AmplifierPump, Oscillator, SignalUid, ValueOrParameter};

pub trait SignalInfo {
    fn uid(&self) -> SignalUid;
    fn awg_key(&self) -> AwgKey;
    fn sampling_rate(&self) -> f64;
    fn device_traits(&self) -> &'static DeviceTraits;
    fn oscillator(&self) -> Option<&Oscillator>;
    fn lo_frequency(&self) -> Option<&ValueOrParameter<f64>>;
    fn supports_initial_local_oscillator_frequency(&self) -> bool;
    fn voltage_offset(&self) -> Option<&ValueOrParameter<f64>>;
    fn supports_initial_voltage_offset(&self) -> bool;
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

    fn sample_multiple(&self) -> u16 {
        self.device_traits().sample_multiple
    }
}

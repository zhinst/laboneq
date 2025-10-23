// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

//! This module defines signal information that is required by
//! the Scheduler.
use crate::{
    experiment::types::{Oscillator, RealValue, SignalUid},
    utils::SignalGridInfo,
};
use laboneq_common::{device_traits::DeviceTraits, types::AwgKey};

pub trait SignalInfo {
    fn uid(&self) -> SignalUid;
    fn awg_key(&self) -> AwgKey;
    fn sampling_rate(&self) -> f64;
    fn device_traits(&self) -> &'static DeviceTraits;
    fn oscillator(&self) -> Option<&Oscillator>;
    fn lo_frequency(&self) -> Option<&RealValue>;
    fn supports_initial_local_oscillator_frequency(&self) -> bool;
    fn voltage_offset(&self) -> Option<&RealValue>;
    fn supports_initial_voltage_offset(&self) -> bool;
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

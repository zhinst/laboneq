// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::ops::RangeInclusive;

use crate::types::DeviceKind;
use laboneq_units::duration::{Duration, Frequency, Hertz, Second, hertz, seconds};

/// Commonly used device traits
pub struct DeviceTraits {
    pub channels_per_awg: u16,
    pub sampling_rate: f64,
    pub sample_multiple: u16,
    pub device_class: u8,
    pub oscillator_set_latency: Duration<Second>,
    pub oscillator_reset_duration: Duration<Second>,
    pub lo_frequency_granularity: Option<Frequency<Hertz>>,
    pub lo_frequency_range: Option<RangeInclusive<Frequency<Hertz>>>,
    pub integration_dsp_latency: Option<Duration<Second>>,
    /// Granularity of port delay settings in number of samples
    pub port_delay_granularity: u8,
}

impl DeviceTraits {
    pub fn from_device_kind(kind: &DeviceKind) -> &'static Self {
        match kind {
            DeviceKind::Hdawg => &HDAWG_TRAITS,
            DeviceKind::Uhfqa => &UHFQA_TRAITS,
            DeviceKind::Shfsg => &SHFSG_TRAITS,
            DeviceKind::Shfqa => &SHFQA_TRAITS,
            DeviceKind::PrettyPrinterDevice => &PRETTYPRINTERDEVICE_TRAITS,
        }
    }

    pub fn min_lo_frequency(&self) -> Option<&Frequency<Hertz>> {
        self.lo_frequency_range.as_ref().map(|r| r.start())
    }

    pub fn max_lo_frequency(&self) -> Option<&Frequency<Hertz>> {
        self.lo_frequency_range.as_ref().map(|r| r.end())
    }
}

pub const HDAWG_TRAITS: DeviceTraits = DeviceTraits {
    sampling_rate: 2.4e9,
    channels_per_awg: 2,
    sample_multiple: 16,
    device_class: 0,
    oscillator_set_latency: seconds(304e-9),
    oscillator_reset_duration: seconds(80e-9),
    lo_frequency_granularity: None,
    lo_frequency_range: None,
    integration_dsp_latency: None,
    port_delay_granularity: 1,
};

pub const UHFQA_TRAITS: DeviceTraits = DeviceTraits {
    sampling_rate: 1.8e9,
    channels_per_awg: 2,
    sample_multiple: 8,
    device_class: 0,
    oscillator_set_latency: seconds(0.0),
    oscillator_reset_duration: seconds(40e-9),
    lo_frequency_granularity: None,
    lo_frequency_range: None,
    integration_dsp_latency: None,
    port_delay_granularity: 4,
};

pub const SHFSG_TRAITS: DeviceTraits = DeviceTraits {
    sampling_rate: 2e9,
    channels_per_awg: 1,
    sample_multiple: 16,
    device_class: 0,
    oscillator_set_latency: seconds(88e-9),
    oscillator_reset_duration: seconds(56e-9),
    lo_frequency_granularity: Some(hertz(100e6)),
    lo_frequency_range: Some(hertz(1e9)..=hertz(8.5e9)),
    integration_dsp_latency: None,
    port_delay_granularity: 1,
};

pub const SHFQA_TRAITS: DeviceTraits = DeviceTraits {
    sampling_rate: 2e9,
    channels_per_awg: 1,
    sample_multiple: 16,
    device_class: 0,
    oscillator_set_latency: seconds(88e-9),
    oscillator_reset_duration: seconds(56e-9),
    lo_frequency_granularity: Some(hertz(100e6)),
    lo_frequency_range: Some(hertz(1e9)..=hertz(8.5e9)),
    integration_dsp_latency: Some(seconds(212e-9)),
    port_delay_granularity: 4,
};

pub const PRETTYPRINTERDEVICE_TRAITS: DeviceTraits = DeviceTraits {
    sampling_rate: 2e9,
    channels_per_awg: 1,
    sample_multiple: 4,
    device_class: 1,
    oscillator_set_latency: seconds(36e-9),
    oscillator_reset_duration: seconds(32e-9),
    lo_frequency_granularity: None,
    lo_frequency_range: None,
    integration_dsp_latency: None,
    port_delay_granularity: 0, // Not applicable
};

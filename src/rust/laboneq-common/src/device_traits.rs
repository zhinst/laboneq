// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::ops::RangeInclusive;

use crate::types::DeviceKind;
use laboneq_units::duration::{Duration, Frequency, Hertz, Second, hertz, seconds};

/// Default lead times for various devices and setups
pub const DEFAULT_HDAWG_LEAD_PQSC: Duration<Second> = seconds(80e-9);
pub const DEFAULT_HDAWG_LEAD_PQSC_2GHZ: Duration<Second> = seconds(80e-9);
pub const DEFAULT_HDAWG_LEAD_DESKTOP_SETUP: Duration<Second> = seconds(20e-9); // PW 2022-09-21, dev2806, FPGA 68366, dev8047, FPGA 68666 & 68603
pub const DEFAULT_HDAWG_LEAD_DESKTOP_SETUP_2GHZ: Duration<Second> = seconds(24e-9);
pub const DEFAULT_UHFQA_LEAD_PQSC: Duration<Second> = seconds(80e-9);
pub const DEFAULT_SHFQA_LEAD_PQSC: Duration<Second> = seconds(80e-9);
pub const DEFAULT_SHFSG_LEAD_PQSC: Duration<Second> = seconds(80e-9);
pub const DEFAULT_TESTDEVICE_LEAD: Duration<Second> = seconds(1200e-9);

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
    pub supports_precompensation: bool,
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
    supports_precompensation: true,
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
    supports_precompensation: false,
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
    supports_precompensation: false,
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
    supports_precompensation: false,
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
    supports_precompensation: true,
};

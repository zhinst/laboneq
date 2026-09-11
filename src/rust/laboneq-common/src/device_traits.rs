// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::ops::RangeInclusive;

use laboneq_units::duration::{Duration, Frequency, Hertz, Second, hertz, seconds};

/// Default lead times for various devices and setups
pub const DEFAULT_HDAWG_LEAD_PQSC: Duration<Second> = seconds(80e-9);
pub const DEFAULT_HDAWG_LEAD_PQSC_2GHZ: Duration<Second> = seconds(80e-9);
pub const DEFAULT_HDAWG_LEAD_DESKTOP_SETUP: Duration<Second> = seconds(20e-9); // PW 2022-09-21, dev2806, FPGA 68366, dev8047, FPGA 68666 & 68603
pub const DEFAULT_HDAWG_LEAD_DESKTOP_SETUP_2GHZ: Duration<Second> = seconds(24e-9);
pub const DEFAULT_UHFQA_LEAD_PQSC: Duration<Second> = seconds(80e-9);
pub const DEFAULT_SHFQA_LEAD_PQSC: Duration<Second> = seconds(80e-9);
pub const DEFAULT_SHFSG_LEAD_PQSC: Duration<Second> = seconds(80e-9);
pub const DEFAULT_ZQCS_LEAD: Duration<Second> = seconds(1200e-9);

/// Default sampling rates for various devices and channels
pub const SHFSG_SAMPLING_RATE: Frequency<Hertz> = hertz(2e9);
pub const SHFQA_SAMPLING_RATE: Frequency<Hertz> = hertz(2e9);
pub const HDAWG_SAMPLING_RATE_WITH_SHF: Frequency<Hertz> = hertz(2.0e9);
pub const HDAWG_SAMPLING_RATE_WITHOUT_SHF: Frequency<Hertz> = hertz(2.4e9);
pub const UHFQA_SAMPLING_RATE: Frequency<Hertz> = hertz(1.8e9);
pub const ZQCS_OUTPUT_SAMPLING_RATE: Frequency<Hertz> = hertz(2e9);
pub const ZQCS_INPUT_SAMPLING_RATE: Frequency<Hertz> = hertz(4e9);

/// Commonly used device traits
#[derive(Debug, PartialEq)]
pub struct DeviceTraits {
    /// Name of the device model, as shown to the user.
    pub name: &'static str,
    pub channels_per_awg: u16,
    pub sample_multiple: u16,
    pub oscillator_set_latency: Duration<Second>,
    pub oscillator_reset_duration: Duration<Second>,
    pub lo_frequency_granularity: Option<Frequency<Hertz>>,
    pub lo_frequency_range: Option<RangeInclusive<Frequency<Hertz>>>,
    pub integration_dsp_latency: Option<Duration<Second>>,
    /// Granularity of port delay settings in number of samples
    pub port_delay_granularity: u8,
    pub supports_precompensation: bool,
    /// A calibrated oscillator frequency reaches the device as an initial value,
    /// so the schedule needs no operation to set it.
    pub supports_initial_oscillator_frequency: bool,
    /// A calibrated voltage offset reaches the device as an initial value, so
    /// the schedule needs no operation to set it.
    pub supports_initial_voltage_offset: bool,
    /// The output can mute itself while it plays nothing.
    pub supports_output_mute: bool,
    /// Acquisitions on one signal may differ in length.
    pub supports_multiple_acquisition_lengths: bool,
    /// A hardware-modulated RF signal accepts a pulse phase and a complex
    /// amplitude.
    pub supports_baseband_phase_modulation: bool,
    /// Feedback from an acquisition on this device reaches signals on the same
    /// physical device without a round trip to the central controller.
    pub supports_local_feedback: bool,
}

impl DeviceTraits {
    pub fn min_lo_frequency(&self) -> Option<&Frequency<Hertz>> {
        self.lo_frequency_range.as_ref().map(|r| r.start())
    }

    pub fn max_lo_frequency(&self) -> Option<&Frequency<Hertz>> {
        self.lo_frequency_range.as_ref().map(|r| r.end())
    }
}

/// Placeholder for tests that need a device, but no property of it.
#[cfg(feature = "test_utils")]
pub const TEST_DEVICE_TRAITS: DeviceTraits = DeviceTraits {
    name: "TESTDEVICE",
    channels_per_awg: 1,
    sample_multiple: 1,
    oscillator_set_latency: seconds(0.0),
    oscillator_reset_duration: seconds(0.0),
    lo_frequency_granularity: None,
    lo_frequency_range: None,
    integration_dsp_latency: None,
    port_delay_granularity: 1,
    supports_precompensation: false,
    supports_initial_oscillator_frequency: false,
    supports_initial_voltage_offset: false,
    supports_output_mute: false,
    supports_multiple_acquisition_lengths: false,
    supports_baseband_phase_modulation: false,
    supports_local_feedback: false,
};

// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

//! Traits of the QCCS device models.

use codegenerator_py::DeviceKind as CodegenDeviceKind;
use laboneq_common::device_traits::DeviceTraits;
use laboneq_units::duration::{hertz, seconds};

pub(crate) const HDAWG_TRAITS: DeviceTraits = DeviceTraits {
    name: "HDAWG",
    channels_per_awg: 2,
    sample_multiple: CodegenDeviceKind::HDAWG.traits().sample_multiple,
    oscillator_set_latency: seconds(304e-9),
    oscillator_reset_duration: seconds(80e-9),
    lo_frequency_granularity: None,
    lo_frequency_range: None,
    integration_dsp_latency: None,
    port_delay_granularity: 1,
    supports_precompensation: true,
    supports_initial_oscillator_frequency: false,
    supports_initial_voltage_offset: false,
    supports_output_mute: false,
    supports_multiple_acquisition_lengths: false,
    supports_baseband_phase_modulation: false,
    supports_local_feedback: false,
};

pub(crate) const UHFQA_TRAITS: DeviceTraits = DeviceTraits {
    name: "UHFQA",
    channels_per_awg: 2,
    sample_multiple: CodegenDeviceKind::UHFQA.traits().sample_multiple,
    oscillator_set_latency: seconds(0.0),
    oscillator_reset_duration: seconds(40e-9),
    lo_frequency_granularity: None,
    lo_frequency_range: None,
    integration_dsp_latency: None,
    port_delay_granularity: 4,
    supports_precompensation: false,
    supports_initial_oscillator_frequency: false,
    supports_initial_voltage_offset: false,
    supports_output_mute: false,
    supports_multiple_acquisition_lengths: false,
    supports_baseband_phase_modulation: false,
    supports_local_feedback: false,
};

pub(crate) const SHFSG_TRAITS: DeviceTraits = DeviceTraits {
    name: "SHFSG",
    channels_per_awg: 1,
    sample_multiple: CodegenDeviceKind::SHFSG.traits().sample_multiple,
    oscillator_set_latency: seconds(88e-9),
    oscillator_reset_duration: seconds(56e-9),
    lo_frequency_granularity: Some(hertz(100e6)),
    lo_frequency_range: Some(hertz(1e9)..=hertz(8.5e9)),
    integration_dsp_latency: None,
    port_delay_granularity: 1,
    supports_precompensation: false,
    supports_initial_oscillator_frequency: false,
    supports_initial_voltage_offset: false,
    supports_output_mute: true,
    supports_multiple_acquisition_lengths: false,
    supports_baseband_phase_modulation: false,
    supports_local_feedback: true,
};

pub(crate) const SHFQA_TRAITS: DeviceTraits = DeviceTraits {
    name: "SHFQA",
    channels_per_awg: 1,
    sample_multiple: CodegenDeviceKind::SHFQA.traits().sample_multiple,
    oscillator_set_latency: seconds(88e-9),
    oscillator_reset_duration: seconds(56e-9),
    lo_frequency_granularity: Some(hertz(100e6)),
    lo_frequency_range: Some(hertz(1e9)..=hertz(8.5e9)),
    integration_dsp_latency: Some(seconds(212e-9)),
    port_delay_granularity: 4,
    supports_precompensation: false,
    supports_initial_oscillator_frequency: false,
    supports_initial_voltage_offset: false,
    supports_output_mute: true,
    supports_multiple_acquisition_lengths: false,
    supports_baseband_phase_modulation: false,
    supports_local_feedback: true,
};

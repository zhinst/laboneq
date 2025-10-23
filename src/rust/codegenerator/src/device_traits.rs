// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use crate::ir::Samples;
use laboneq_common::device_traits as device_traits_common;

/// Device specific traits for code generation
///
/// NOTE: Mirror from `laboneq` Python package:
///     src/python/laboneq/compiler/common/device_type.py
///     Ensure that the values do match when changing.
/// NOTE: The frequency of the HDAWG may be either 2.4 GHz or 2 GHz depending on the
///     setup - the compiler determines which one during analyzing the device setup.
///     This is not reflected in the traits here yet - we have specified the maximum
///     frequency of 2.4 GHz used in a non-SHF* setup. todo: Add support for
///     the large setup case once this part of the code is used in rust.

#[derive(Debug)]
pub struct DeviceTraits {
    pub type_str: &'static str,
    pub sample_multiple: u16,
    pub sampling_rate: f64,
    pub min_play_wave: u32,
    pub max_play_zero_hold: Samples,
    pub amplitude_register_count: u16,
    pub supports_oscillator_switching: bool,
    pub supports_binary_waves: bool,
    pub supports_digital_iq_modulation: bool,
    pub supports_output_mute: bool,
    pub output_mute_engage_delay: f64,
    pub output_mute_disengage_delay: f64,
    pub supports_wait_wave: bool,
    pub supports_command_table: bool,
    pub supports_zsync: bool,
    pub has_prng: bool,
    pub require_play_zero_after_loop: bool,
    pub playwave_max_hint: Option<Samples>,
    pub is_qa_device: bool,
    pub number_of_trigger_bits: u8,
    pub ct_schema_version: Option<&'static str>,
}

pub const HDAWG_TRAITS: DeviceTraits = DeviceTraits {
    type_str: "HDAWG",
    sample_multiple: device_traits_common::HDAWG_TRAITS.sample_multiple,
    sampling_rate: device_traits_common::HDAWG_TRAITS.sampling_rate,
    min_play_wave: 32,
    max_play_zero_hold: (1 << 19) - 16,
    amplitude_register_count: 4,
    supports_oscillator_switching: false,
    supports_binary_waves: true,
    supports_digital_iq_modulation: true,
    supports_output_mute: false,
    output_mute_engage_delay: f64::NAN,    // Not supported
    output_mute_disengage_delay: f64::NAN, // Not supported
    supports_wait_wave: true,
    supports_command_table: true,
    supports_zsync: true,
    has_prng: true,
    require_play_zero_after_loop: false,
    playwave_max_hint: None,
    is_qa_device: false,
    number_of_trigger_bits: 4,
    ct_schema_version: Some("hd_1.1.0"),
};

pub const UHFQA_TRAITS: DeviceTraits = DeviceTraits {
    type_str: "UHFQA",
    sample_multiple: device_traits_common::UHFQA_TRAITS.sample_multiple,
    sampling_rate: device_traits_common::UHFQA_TRAITS.sampling_rate,
    min_play_wave: 16,
    max_play_zero_hold: 131056,
    amplitude_register_count: 1,
    supports_oscillator_switching: false,
    supports_binary_waves: true,
    supports_digital_iq_modulation: false,
    supports_output_mute: false,
    output_mute_engage_delay: f64::NAN,    // Not supported
    output_mute_disengage_delay: f64::NAN, // Not supported
    supports_wait_wave: true,
    supports_command_table: false,
    supports_zsync: false,
    has_prng: false,
    require_play_zero_after_loop: true,
    playwave_max_hint: None,
    is_qa_device: true,
    number_of_trigger_bits: 4,
    ct_schema_version: None,
};

pub const SHFSG_TRAITS: DeviceTraits = DeviceTraits {
    type_str: "SHFSG",
    sample_multiple: device_traits_common::SHFSG_TRAITS.sample_multiple,
    sampling_rate: device_traits_common::SHFSG_TRAITS.sampling_rate,
    min_play_wave: 32,
    max_play_zero_hold: (1 << 19) - 16,
    amplitude_register_count: 1,
    supports_oscillator_switching: true,
    supports_binary_waves: true,
    supports_digital_iq_modulation: true,
    supports_output_mute: true,
    // Verified by MH (2024-03-16) on dev12156 & dev12093, rev 69800.
    // PW (2024-07-02) bump by 26 ns, account for possible RTR delay
    output_mute_engage_delay: 24e-9 + 26e-9,
    // Verified by MH (2024-03-16) on dev12156 & dev12093 rev 69800.
    output_mute_disengage_delay: -100e-9,
    supports_wait_wave: true,
    supports_command_table: true,
    supports_zsync: true,
    has_prng: true,
    require_play_zero_after_loop: false,
    playwave_max_hint: None,
    is_qa_device: false,
    // todo: The documentation states 4 trigger bits, but the LabOne Q code only allows 1.
    number_of_trigger_bits: 1,
    ct_schema_version: Some("sg_1.2.0"),
};

pub const SHFQA_TRAITS: DeviceTraits = DeviceTraits {
    type_str: "SHFQA",
    sample_multiple: device_traits_common::SHFQA_TRAITS.sample_multiple,
    sampling_rate: device_traits_common::SHFQA_TRAITS.sampling_rate,
    min_play_wave: 32,
    max_play_zero_hold: (1 << 19) - 16,
    amplitude_register_count: 1,
    supports_oscillator_switching: false,
    supports_binary_waves: false,
    supports_digital_iq_modulation: false,
    supports_output_mute: true,
    // Verified by MH (2024-03-16) on dev12156 & dev12093, rev 69800.
    // Marker output setTrigger(1) lead time
    output_mute_engage_delay: 128e-9,
    // Verified by MH (2024-03-16) on dev12156 & dev12093 rev 69800.
    // Marker output setTrigger(0) delay time
    output_mute_disengage_delay: -16e-9,
    supports_wait_wave: false,
    supports_command_table: false,
    supports_zsync: true,
    has_prng: false,
    require_play_zero_after_loop: false,
    playwave_max_hint: Some(4096),
    is_qa_device: true,
    // todo: The documentation states 2 trigger bits, but the LabOne Q code only allows 1.
    number_of_trigger_bits: 1,
    ct_schema_version: None,
};

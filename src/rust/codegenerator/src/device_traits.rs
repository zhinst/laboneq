// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

/// Device specific traits for code generation
///
/// NOTE: Mirror from `laboneq` Python package:
///     src/python/laboneq/compiler/common/device_type.py
///     Ensure that the values do match when changing.
pub struct DeviceTraits {
    pub sample_multiple: u16,
    pub min_play_wave: u16,
    pub amplitude_register_count: u16,
    pub supports_oscillator_switching: bool,
    pub playwave_max_hint: Option<u64>,
    pub is_qa_device: bool,
}

pub const HDAWG_TRAITS: DeviceTraits = DeviceTraits {
    sample_multiple: 16,
    min_play_wave: 32,
    amplitude_register_count: 4,
    supports_oscillator_switching: false,
    playwave_max_hint: None,
    is_qa_device: false,
};

pub const UHFQA_TRAITS: DeviceTraits = DeviceTraits {
    sample_multiple: 8,
    min_play_wave: 16,
    amplitude_register_count: 1,
    supports_oscillator_switching: false,
    playwave_max_hint: None,
    is_qa_device: true,
};

pub const SHFSG_TRAITS: DeviceTraits = DeviceTraits {
    sample_multiple: 16,
    min_play_wave: 32,
    amplitude_register_count: 1,
    supports_oscillator_switching: true,
    playwave_max_hint: None,
    is_qa_device: false,
};

pub const SHFQA_TRAITS: DeviceTraits = DeviceTraits {
    sample_multiple: 16,
    min_play_wave: 32,
    amplitude_register_count: 1,
    supports_oscillator_switching: false,
    playwave_max_hint: Some(4096),
    is_qa_device: true,
};

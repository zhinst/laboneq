// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use crate::types::DeviceKind;

/// Commonly used device traits
pub struct DeviceTraits {
    pub channels_per_awg: u16,
    pub sampling_rate: f64,
    pub sample_multiple: u16,
    pub device_class: u8,
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
}

pub const HDAWG_TRAITS: DeviceTraits = DeviceTraits {
    sampling_rate: 2.4e9,
    channels_per_awg: 2,
    sample_multiple: 16,
    device_class: 0,
};

pub const UHFQA_TRAITS: DeviceTraits = DeviceTraits {
    sampling_rate: 1.8e9,
    channels_per_awg: 2,
    sample_multiple: 8,
    device_class: 0,
};

pub const SHFSG_TRAITS: DeviceTraits = DeviceTraits {
    sampling_rate: 2e9,
    channels_per_awg: 1,
    sample_multiple: 16,
    device_class: 0,
};

pub const SHFQA_TRAITS: DeviceTraits = DeviceTraits {
    sampling_rate: 2e9,
    channels_per_awg: 1,
    sample_multiple: 16,
    device_class: 0,
};

pub const PRETTYPRINTERDEVICE_TRAITS: DeviceTraits = DeviceTraits {
    sampling_rate: 2e9,
    channels_per_awg: 1,
    sample_multiple: 4,
    device_class: 1,
};

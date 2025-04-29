// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use crate::device_traits;

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum DeviceKind {
    HDAWG,
    SHFQA,
    SHFSG,
    UHFQA,
}

impl DeviceKind {
    pub(crate) const fn traits(&self) -> &device_traits::DeviceTraits {
        match self {
            DeviceKind::HDAWG => &device_traits::HDAWG_TRAITS,
            DeviceKind::SHFQA => &device_traits::SHFQA_TRAITS,
            DeviceKind::SHFSG => &device_traits::SHFSG_TRAITS,
            DeviceKind::UHFQA => &device_traits::UHFQA_TRAITS,
        }
    }

    pub(crate) fn as_str(&self) -> &'static str {
        match self {
            DeviceKind::HDAWG => "HDAWG",
            DeviceKind::SHFQA => "SHFQA",
            DeviceKind::SHFSG => "SHFSG",
            DeviceKind::UHFQA => "UHFQA",
        }
    }
}

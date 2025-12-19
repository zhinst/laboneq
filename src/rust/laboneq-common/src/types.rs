// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use crate::named_id::NamedId;

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct DeviceUid(pub NamedId);

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct PhysicalDeviceUid(pub u16);

impl From<u16> for PhysicalDeviceUid {
    fn from(value: u16) -> Self {
        PhysicalDeviceUid(value)
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct AwgKey(pub u64);

#[derive(Debug, Clone, Eq, PartialEq, Copy)]
pub enum DeviceKind {
    Hdawg,
    Shfqa,
    Shfsg,
    Uhfqa,
    PrettyPrinterDevice,
}

impl DeviceKind {
    pub fn is_qa_device(&self) -> bool {
        matches!(self, Self::Shfqa | Self::Uhfqa)
    }
}

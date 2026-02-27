// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use laboneq_common::{
    device_traits::DeviceTraits,
    types::{DeviceKind, PhysicalDeviceUid},
};
use laboneq_dsl::types::DeviceUid;

/// Device used in the experiment.
#[derive(Debug, Clone, PartialEq)]
pub struct Device {
    uid: DeviceUid,
    physical_device_uid: PhysicalDeviceUid,
    is_shfqc: bool,
    kind: DeviceKind,
}

impl Device {
    pub fn uid(&self) -> DeviceUid {
        self.uid
    }

    /// Physical device this device maps to
    /// This UID is used to group virtual devices that share the same
    /// physical hardware, enabling proper device detection.
    pub fn physical_device_uid(&self) -> PhysicalDeviceUid {
        self.physical_device_uid
    }

    /// Whether the device is part of a SHFQC
    /// This is needed as SHFQC device is split internally into virtual
    /// SHFQA + SHFSG devices.
    pub fn is_shfqc(&self) -> bool {
        self.is_shfqc
    }

    pub fn kind(&self) -> &DeviceKind {
        &self.kind
    }

    /// Get the traits of the device.
    pub fn traits(&self) -> &'static DeviceTraits {
        DeviceTraits::from_device_kind(self.kind())
    }
}

pub mod builder {
    use super::*;

    pub struct DeviceBuilder {
        uid: DeviceUid,
        physical_device_uid: PhysicalDeviceUid,
        is_shfqc: bool,
        kind: DeviceKind,
    }

    impl DeviceBuilder {
        pub fn new(
            uid: DeviceUid,
            physical_device_uid: PhysicalDeviceUid,
            kind: DeviceKind,
        ) -> Self {
            Self {
                uid,
                physical_device_uid,
                is_shfqc: false,
                kind,
            }
        }

        pub fn shfqc(mut self, is_shfqc: bool) -> Self {
            self.is_shfqc = is_shfqc;
            self
        }

        pub fn build(self) -> Device {
            Device {
                uid: self.uid,
                physical_device_uid: self.physical_device_uid,
                is_shfqc: self.is_shfqc,
                kind: self.kind,
            }
        }
    }
}

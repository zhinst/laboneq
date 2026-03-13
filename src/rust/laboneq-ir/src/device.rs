// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use laboneq_common::{
    device_options::DeviceOptions,
    device_traits::DeviceTraits,
    types::{DeviceKind, PhysicalDeviceUid, ReferenceClock},
};
use laboneq_dsl::types::DeviceUid;

/// Device used in the experiment.
///
/// A device represents a physical or virtual device used in the experiment, which
/// can have operations and signals associated with it.
#[derive(Debug, Clone, PartialEq)]
pub struct AwgDevice {
    uid: DeviceUid,
    physical_device_uid: PhysicalDeviceUid,
    is_shfqc: bool,
    kind: DeviceKind,
    options: Option<DeviceOptions>,
    reference_clock: Option<ReferenceClock>,
}

impl AwgDevice {
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

    pub fn kind(&self) -> DeviceKind {
        self.kind
    }

    /// Get the traits of the device.
    pub fn traits(&self) -> &'static DeviceTraits {
        DeviceTraits::from_device_kind(&self.kind())
    }

    /// Get the options of the device.
    pub fn options(&self) -> Option<&DeviceOptions> {
        self.options.as_ref()
    }

    /// Check if the device has a specific option.
    pub fn has_option(&self, option: &str) -> bool {
        self.options()
            .as_ref()
            .is_some_and(|opts| opts.contains(option))
    }

    pub fn reference_clock(&self) -> Option<&ReferenceClock> {
        self.reference_clock.as_ref()
    }

    pub fn builder(
        uid: DeviceUid,
        physical_device_uid: PhysicalDeviceUid,
        kind: DeviceKind,
    ) -> builder::AwgDeviceBuilder {
        builder::AwgDeviceBuilder::new(uid, physical_device_uid, kind)
    }
}

pub mod builder {
    use super::*;

    pub struct AwgDeviceBuilder {
        inner: AwgDevice,
    }

    impl AwgDeviceBuilder {
        pub fn new(
            uid: DeviceUid,
            physical_device_uid: PhysicalDeviceUid,
            kind: DeviceKind,
        ) -> Self {
            Self {
                inner: AwgDevice {
                    uid,
                    physical_device_uid,
                    is_shfqc: false,
                    kind,
                    options: None,
                    reference_clock: None,
                },
            }
        }

        pub fn shfqc(mut self, is_shfqc: bool) -> Self {
            self.inner.is_shfqc = is_shfqc;
            self
        }

        pub fn options(mut self, options: DeviceOptions) -> Self {
            self.inner.options = Some(options);
            self
        }

        pub fn reference_clock(mut self, reference_clock: ReferenceClock) -> Self {
            self.inner.reference_clock = Some(reference_clock);
            self
        }

        pub fn build(self) -> AwgDevice {
            self.inner
        }
    }
}

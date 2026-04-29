// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use laboneq_dsl::device_setup::AuxiliaryDevice;
use laboneq_dsl::types::{DeviceUid, SignalUid};

use crate::signal::Signal;
// Re-export for convenience
pub use crate::device::AwgDevice;

/// Device and signal setup used in the experiment.
#[derive(Debug, Clone, PartialEq)]
pub struct DeviceSetup {
    signals: HashMap<SignalUid, Signal>,
    awg_devices: Vec<AwgDevice>,
    auxiliary_devices: Vec<AuxiliaryDevice>,
    is_desktop_setup: bool,

    // Indexes for lookup
    awg_devices_indices: HashMap<DeviceUid, usize>,
}

impl DeviceSetup {
    pub fn new(
        signals: HashMap<SignalUid, Signal>,
        awg_devices: Vec<AwgDevice>,
        auxiliary_devices: Vec<AuxiliaryDevice>,
        is_desktop_setup: bool,
    ) -> Result<Self, String> {
        // Validate all signals reference existing devices
        for signal in signals.values() {
            if !awg_devices.iter().any(|d| d.uid() == signal.device_uid) {
                return Err(format!(
                    "Signal '{}' references unknown device",
                    signal.uid.0
                ));
            }
        }

        let awg_devices_indices = awg_devices
            .iter()
            .enumerate()
            .map(|(idx, device)| (device.uid(), idx))
            .collect();

        Ok(Self {
            signals,
            awg_devices,
            auxiliary_devices,
            is_desktop_setup,
            awg_devices_indices,
        })
    }

    pub fn signals(&self) -> impl Iterator<Item = &Signal> {
        self.signals.values()
    }

    pub fn signal_by_uid(&self, uid: &SignalUid) -> Option<&Signal> {
        self.signals.get(uid)
    }

    pub fn device_by_uid(&self, uid: &DeviceUid) -> Option<&AwgDevice> {
        self.awg_devices_indices
            .get(uid)
            .map(|&idx| &self.awg_devices[idx])
    }

    pub fn is_desktop_setup(&self) -> bool {
        self.is_desktop_setup
    }

    pub fn awg_devices(&self) -> impl Iterator<Item = &AwgDevice> {
        self.awg_devices.iter()
    }

    pub fn auxiliary_devices(&self) -> impl Iterator<Item = &AuxiliaryDevice> {
        self.auxiliary_devices.iter()
    }
}

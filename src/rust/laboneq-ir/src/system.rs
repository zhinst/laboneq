// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use laboneq_common::types::AwgKey;
use laboneq_dsl::types::{DeviceUid, SignalUid};

use crate::{awg::AwgCore, signal::Signal};
// Re-export for convenience
pub use crate::device::AwgDevice;

/// Device and signal setup used in the experiment.
#[derive(Debug, Clone, PartialEq)]
pub struct DeviceSetup {
    signals: HashMap<SignalUid, Signal>,
    awg_devices: HashMap<DeviceUid, AwgDevice>,
    awg_cores: HashMap<AwgKey, AwgCore>,
    is_desktop_setup: bool,
}

impl DeviceSetup {
    pub fn new(
        signals: HashMap<SignalUid, Signal>,
        awg_devices: HashMap<DeviceUid, AwgDevice>,
        awg_cores: Vec<AwgCore>,
        is_desktop_setup: bool,
    ) -> Result<Self, String> {
        // Validate all signals reference existing devices
        for signal in signals.values() {
            if !awg_devices.contains_key(&signal.device_uid) {
                return Err(format!(
                    "Signal '{}' references unknown device",
                    signal.uid.0
                ));
            }
        }
        Ok(Self {
            signals,
            awg_devices,
            awg_cores: awg_cores.into_iter().map(|awg| (awg.uid(), awg)).collect(),
            is_desktop_setup,
        })
    }

    pub fn signals(&self) -> impl Iterator<Item = &Signal> {
        self.signals.values()
    }

    pub fn signal_by_uid(&self, uid: &SignalUid) -> Option<&Signal> {
        self.signals.get(uid)
    }

    pub fn device_by_uid(&self, uid: &DeviceUid) -> Option<&AwgDevice> {
        self.awg_devices.get(uid)
    }

    pub fn awg_core(&self, awg_key: &AwgKey) -> Option<&AwgCore> {
        self.awg_cores.get(awg_key)
    }

    pub fn is_desktop_setup(&self) -> bool {
        self.is_desktop_setup
    }

    pub fn awg_devices(&self) -> impl Iterator<Item = &AwgDevice> {
        self.awg_devices.values()
    }
}

// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

//! Properties of the setup an experiment targets.

use std::collections::HashMap;

use laboneq_common::device_options::DeviceOptions;
use laboneq_common::types::{AuxiliaryDevice, AwgKey, ChannelKey};
use laboneq_dsl::types::{DeviceUid, PhysicalChannelUid, SignalUid};
use laboneq_error::laboneq_error;
use laboneq_units::duration::{Duration, Second};

use crate::Result;
use crate::ir::compilation_job::DeviceKind;

pub struct HardwareSetup {
    pub signals: Vec<SignalChannelProperties>,
    /// All the devices defined in in the setup.
    /// Not all devices may be used in the experiment.
    pub devices: Vec<SetupDevice>,
    pub auxiliary_devices: Vec<AuxiliaryDevice>,
}

impl HardwareSetup {
    /// A device, as the setup this experiment targets defines it.
    pub(crate) fn device(&self, uid: DeviceUid) -> Result<&SetupDevice> {
        self.devices
            .iter()
            .find(|device| device.uid == uid)
            .ok_or_else(|| laboneq_error!("Expected properties for device UID {}", uid.0))
    }
}

pub struct SetupDevice {
    pub uid: DeviceUid,
    pub kind: DeviceKind,
    /// Whether the device forms one half of an SHFQC, which appears as a
    /// virtual SHFQA and a virtual SHFSG.
    pub is_shfqc: bool,
    pub lead_delay: Duration<Second>,
    pub options: DeviceOptions,
}

pub struct SignalChannelProperties {
    pub signal_uid: SignalUid,
    pub awg_key: AwgKey,
    pub awg_index: u16,
    pub channels: Vec<u8>,
    pub routed_output_channel_map: HashMap<PhysicalChannelUid, u8>,
    pub ppc_channel: Option<ChannelKey>,
}

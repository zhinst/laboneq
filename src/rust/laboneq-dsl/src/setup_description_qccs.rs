// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use itertools::Itertools;

use laboneq_common::types::{AuxiliaryDeviceKind, SignalKind};

use crate::device_setup::{Instrument, InstrumentKind};
use crate::types::DeviceUid;

#[derive(Debug, Clone, PartialEq)]
pub struct SetupDescriptionQccs {
    pub instruments: Vec<Instrument>,
    pub auxiliary_devices: Vec<AuxiliaryDevice>,

    pub device_signals: Vec<PhysicalChannel>,
}

impl SetupDescriptionQccs {
    pub fn new(instruments: Vec<Instrument>, device_signals: Vec<PhysicalChannel>) -> Self {
        let (instruments, auxiliary_devices) =
            instruments.into_iter().partition_map(|i| match i.kind {
                InstrumentKind::Pqsc => {
                    itertools::Either::Right(AuxiliaryDevice::new(i.uid, AuxiliaryDeviceKind::Pqsc))
                }
                InstrumentKind::Qhub => {
                    itertools::Either::Right(AuxiliaryDevice::new(i.uid, AuxiliaryDeviceKind::Qhub))
                }
                InstrumentKind::Shfppc => itertools::Either::Right(AuxiliaryDevice::new(
                    i.uid,
                    AuxiliaryDeviceKind::Shfppc,
                )),
                _ => itertools::Either::Left(i),
            });

        Self {
            instruments,
            auxiliary_devices,
            device_signals,
        }
    }

    pub fn instruments(&self) -> &[Instrument] {
        &self.instruments
    }

    pub fn auxiliary_devices(&self) -> &[AuxiliaryDevice] {
        &self.auxiliary_devices
    }
}

/// Auxiliary devices used in the experiment, which do not have signals but are still relevant for the setup.
#[derive(Debug, Clone, PartialEq)]
pub struct AuxiliaryDevice {
    uid: DeviceUid,
    kind: AuxiliaryDeviceKind,
}

impl AuxiliaryDevice {
    pub fn new(uid: DeviceUid, kind: AuxiliaryDeviceKind) -> Self {
        Self { uid, kind }
    }

    pub fn uid(&self) -> DeviceUid {
        self.uid
    }

    pub fn kind(&self) -> AuxiliaryDeviceKind {
        self.kind
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PhysicalChannel {
    pub uid: String,
    pub device_uid: DeviceUid,
    pub ports: Vec<String>,
    pub channel_type: SignalKind,
}

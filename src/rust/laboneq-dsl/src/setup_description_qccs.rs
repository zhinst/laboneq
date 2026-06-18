// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use itertools::Itertools;

use laboneq_common::device_options::DeviceOptions;
use laboneq_common::types::AuxiliaryDeviceKind;
use laboneq_common::uid::PhysicalChannelUid;

use crate::device_setup::{Instrument, InstrumentKind};
use crate::types::DeviceUid;

#[derive(Debug, Clone, PartialEq)]
pub struct SetupDescriptionQccs {
    pub instruments: Vec<Instrument>,
    pub auxiliary_devices: Vec<AuxiliaryDevice>,

    pub device_signals: Vec<PhysicalChannel>,
    pub internal_connections: Vec<InternalConnection>,
}

impl SetupDescriptionQccs {
    pub fn new(
        instruments: Vec<Instrument>,
        device_signals: Vec<PhysicalChannel>,
        internal_connections: Vec<InternalConnection>,
    ) -> Self {
        let (instruments, auxiliary_devices) =
            instruments.into_iter().partition_map(|i| match i.kind {
                InstrumentKind::Pqsc => itertools::Either::Right(AuxiliaryDevice::new(
                    i.uid,
                    AuxiliaryDeviceKind::Pqsc,
                    i.options,
                )),
                InstrumentKind::Qhub => itertools::Either::Right(AuxiliaryDevice::new(
                    i.uid,
                    AuxiliaryDeviceKind::Qhub,
                    i.options,
                )),
                InstrumentKind::Shfppc => itertools::Either::Right(AuxiliaryDevice::new(
                    i.uid,
                    AuxiliaryDeviceKind::Shfppc,
                    i.options,
                )),
                _ => itertools::Either::Left(i),
            });

        Self {
            instruments,
            auxiliary_devices,
            device_signals,
            internal_connections,
        }
    }

    pub fn instruments(&self) -> &[Instrument] {
        &self.instruments
    }

    pub fn auxiliary_devices(&self) -> &[AuxiliaryDevice] {
        &self.auxiliary_devices
    }

    pub fn internal_connections(&self) -> &[InternalConnection] {
        &self.internal_connections
    }

    pub fn signals_by_device(
        &self,
        device_uid: DeviceUid,
    ) -> impl Iterator<Item = &PhysicalChannel> {
        self.device_signals
            .iter()
            .filter(move |s| s.device_uid == device_uid)
    }
}

/// Auxiliary devices used in the experiment, which do not have signals but are still relevant for the setup.
#[derive(Debug, Clone, PartialEq)]
pub struct AuxiliaryDevice {
    uid: DeviceUid,
    kind: AuxiliaryDeviceKind,
    options: DeviceOptions,
}

impl AuxiliaryDevice {
    pub fn new(uid: DeviceUid, kind: AuxiliaryDeviceKind, options: DeviceOptions) -> Self {
        Self { uid, kind, options }
    }

    pub fn uid(&self) -> DeviceUid {
        self.uid
    }

    pub fn kind(&self) -> AuxiliaryDeviceKind {
        self.kind
    }

    pub fn options(&self) -> &DeviceOptions {
        &self.options
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PhysicalChannel {
    pub uid: PhysicalChannelUid,
    pub device_uid: DeviceUid,
    pub ports: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InternalConnection {
    pub from_instrument: DeviceUid,
    pub from_port: String,
    pub to_instrument: DeviceUid,
    pub to_port: String,
}

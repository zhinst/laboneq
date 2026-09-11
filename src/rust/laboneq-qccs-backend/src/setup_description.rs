// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

//! The QCCS setup description as it arrives with the experiment: the instruments and
//! channels the compiler targets.

use itertools::Itertools;

use laboneq_common::device_options::DeviceOptions;
use laboneq_common::types::{AuxiliaryDevice, AuxiliaryDeviceKind};

use crate::device_kind::DeviceKind;
use laboneq_common::uid::{DeviceUid, PhysicalChannelUid};

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

#[derive(Debug, Clone, PartialEq)]
pub struct Instrument {
    pub uid: DeviceUid,
    pub kind: InstrumentKind,
    pub options: DeviceOptions,
    pub reference_clock: Option<ReferenceClock>,
}

#[derive(Debug, Clone, Eq, PartialEq, Copy)]
pub enum ReferenceClock {
    Internal,
    External,
}

/// Instrument that is part of the QCCS device setup.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Copy)]
pub enum InstrumentKind {
    Hdawg,
    Shfsg,
    Shfqa,
    Shfqc,
    Uhfqa,
    Shfppc,
    Pqsc,
    Qhub,
}

impl TryInto<DeviceKind> for InstrumentKind {
    type Error = String;

    fn try_into(self) -> Result<DeviceKind, Self::Error> {
        match self {
            InstrumentKind::Hdawg => Ok(DeviceKind::Hdawg),
            InstrumentKind::Shfsg => Ok(DeviceKind::Shfsg),
            InstrumentKind::Shfqa => Ok(DeviceKind::Shfqa),
            InstrumentKind::Uhfqa => Ok(DeviceKind::Uhfqa),
            _ => Err(format!(
                "Cannot convert instrument kind '{self:?}' to device kind"
            )),
        }
    }
}

impl std::str::FromStr for InstrumentKind {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_uppercase().as_str() {
            "HDAWG" => Ok(InstrumentKind::Hdawg),
            "SHFSG" => Ok(InstrumentKind::Shfsg),
            "SHFQA" => Ok(InstrumentKind::Shfqa),
            "SHFQC" => Ok(InstrumentKind::Shfqc),
            "UHFQA" => Ok(InstrumentKind::Uhfqa),
            "SHFPPC" => Ok(InstrumentKind::Shfppc),
            "PQSC" => Ok(InstrumentKind::Pqsc),
            "QHUB" => Ok(InstrumentKind::Qhub),
            _ => Err(format!("Unknown instrument kind: '{}'", s)),
        }
    }
}

impl std::fmt::Display for InstrumentKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let out = match self {
            InstrumentKind::Hdawg => "HDAWG",
            InstrumentKind::Shfsg => "SHFSG",
            InstrumentKind::Shfqa => "SHFQA",
            InstrumentKind::Uhfqa => "UHFQA",
            InstrumentKind::Shfqc => "SHFQC",
            InstrumentKind::Shfppc => "SHFPPC",
            InstrumentKind::Pqsc => "PQSC",
            InstrumentKind::Qhub => "QHUB",
        };
        write!(f, "{}", out)
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

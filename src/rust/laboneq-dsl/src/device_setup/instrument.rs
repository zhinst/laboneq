// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use laboneq_common::device_options::DeviceOptions;
use laboneq_common::types::{AuxiliaryDeviceKind, DeviceKind, PhysicalDeviceUid, ReferenceClock};

use crate::types::DeviceUid;

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
pub struct Instrument {
    pub uid: DeviceUid,
    pub physical_device_uid: PhysicalDeviceUid,
    pub kind: InstrumentKind,
    pub options: DeviceOptions,
    pub reference_clock: Option<ReferenceClock>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Copy)]
pub enum InstrumentKind {
    Hdawg,
    Shfsg,
    Shfqa,
    Shfqc,
    Uhfqa,
    Zqcs,
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
            InstrumentKind::Zqcs => Ok(DeviceKind::Zqcs),
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
            "ZQCS" => Ok(InstrumentKind::Zqcs),
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
            InstrumentKind::Zqcs => "ZQCS",
            InstrumentKind::Shfppc => "SHFPPC",
            InstrumentKind::Pqsc => "PQSC",
            InstrumentKind::Qhub => "QHUB",
        };
        write!(f, "{}", out)
    }
}

// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::fmt::Display;

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct PhysicalDeviceUid(pub u16);

impl From<u16> for PhysicalDeviceUid {
    fn from(value: u16) -> Self {
        PhysicalDeviceUid(value)
    }
}

/// Unique identifier for an AWG. This is a device-agnostic identifier that can be used to reference an AWG across different devices.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct AwgKey(pub u64);

/// Device kind that can have signals or operations performed on it.
///
/// This is different from [`AuxiliaryDeviceKind`] which represents devices without signals, such as SHFPPC or synchronization devices.
#[derive(Debug, Clone, Eq, PartialEq, Copy, Hash)]
pub enum DeviceKind {
    Hdawg,
    Shfqa,
    Shfsg,
    Uhfqa,
    Zqcs,
}

impl DeviceKind {
    pub fn is_qa_device(&self) -> bool {
        matches!(self, Self::Shfqa | Self::Uhfqa)
    }
}

impl std::str::FromStr for DeviceKind {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_uppercase().as_str() {
            "HDAWG" => Ok(DeviceKind::Hdawg),
            "SHFQA" => Ok(DeviceKind::Shfqa),
            "SHFSG" => Ok(DeviceKind::Shfsg),
            "UHFQA" => Ok(DeviceKind::Uhfqa),
            "ZQCS" => Ok(DeviceKind::Zqcs),
            _ => Err(format!("Unknown device kind: '{}'", s)),
        }
    }
}

impl Display for DeviceKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let out = match self {
            DeviceKind::Hdawg => "HDAWG",
            DeviceKind::Shfqa => "SHFQA",
            DeviceKind::Shfsg => "SHFSG",
            DeviceKind::Uhfqa => "UHFQA",
            DeviceKind::Zqcs => "ZQCS",
        };
        write!(f, "{}", out)
    }
}

/// Device kind that represents devices without signals, such as SHFPPC or synchronization devices.
#[derive(Debug, Clone, Eq, PartialEq, Copy, Hash)]
pub enum AuxiliaryDeviceKind {
    Shfppc,
    Pqsc,
    Qhub,
}

impl std::str::FromStr for AuxiliaryDeviceKind {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_uppercase().as_str() {
            "SHFPPC" => Ok(AuxiliaryDeviceKind::Shfppc),
            "PQSC" => Ok(AuxiliaryDeviceKind::Pqsc),
            "QHUB" => Ok(AuxiliaryDeviceKind::Qhub),
            _ => Err(format!("Unknown instrument kind: '{}'", s)),
        }
    }
}

impl Display for AuxiliaryDeviceKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let out = match self {
            AuxiliaryDeviceKind::Shfppc => "SHFPPC",
            AuxiliaryDeviceKind::Pqsc => "PQSC",
            AuxiliaryDeviceKind::Qhub => "QHUB",
        };
        write!(f, "{}", out)
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Copy)]
pub enum ReferenceClock {
    Internal,
    External,
}

impl std::str::FromStr for ReferenceClock {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_uppercase().as_str() {
            "INTERNAL" => Ok(ReferenceClock::Internal),
            "EXTERNAL" => Ok(ReferenceClock::External),
            _ => Err(format!("Unknown reference clock: '{}'", s)),
        }
    }
}

impl Display for ReferenceClock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let out = match self {
            ReferenceClock::Internal => "INTERNAL",
            ReferenceClock::External => "EXTERNAL",
        };
        write!(f, "{}", out)
    }
}

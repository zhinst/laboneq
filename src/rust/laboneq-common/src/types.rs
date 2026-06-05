// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::fmt::Display;

mod literal;
pub use literal::*;

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

/// Physical channel address in the ZQCS rack, encoded as `shelf:slot:frontend:port`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct GeoLocation {
    pub shelf: u16,
    pub slot: u16,
    pub frontend: u16,
    pub port: u16,
}

impl std::str::FromStr for GeoLocation {
    type Err = String;

    fn from_str(port_str: &str) -> Result<Self, Self::Err> {
        let invalid = || format!("Invalid geolocation key: '{}'", port_str);
        let parts: Vec<u16> = port_str
            .split(':')
            .map(|p| p.parse::<u16>().map_err(|_| invalid()))
            .collect::<Result<_, _>>()?;
        if parts.len() != 4 {
            return Err(invalid());
        }
        Ok(Self {
            shelf: parts[0],
            slot: parts[1],
            frontend: parts[2],
            port: parts[3],
        })
    }
}

impl Display for GeoLocation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}:{}:{}:{}",
            self.shelf, self.slot, self.frontend, self.port
        )
    }
}

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

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SignalKind {
    Rf,
    Integration,
    Iq,
}

impl std::str::FromStr for SignalKind {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "rf" => Ok(SignalKind::Rf),
            "iq" => Ok(SignalKind::Iq),
            "integration" => Ok(SignalKind::Integration),
            _ => Err(format!("Unknown signal type: {}", s)),
        }
    }
}

#[cfg(test)]
mod geolocation_tests {
    use super::*;
    use std::str::FromStr;

    #[test]
    fn from_str_valid() {
        let geo = GeoLocation::from_str("1:2:3:4").unwrap();
        assert_eq!(
            geo,
            GeoLocation {
                shelf: 1,
                slot: 2,
                frontend: 3,
                port: 4
            }
        );
    }

    #[test]
    fn from_str_invalid() {
        assert!(GeoLocation::from_str("1:2:3").is_err());
        assert!(GeoLocation::from_str("1:2:3:4:5").is_err());
        assert!(GeoLocation::from_str("a:b:c:d").is_err());
    }

    #[test]
    fn display_roundtrip() {
        let geo = GeoLocation {
            shelf: 1,
            slot: 2,
            frontend: 3,
            port: 4,
        };
        assert_eq!(geo.to_string(), "1:2:3:4");
    }
}

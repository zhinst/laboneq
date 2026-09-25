// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::fmt::Display;
use std::num::NonZeroU16;

mod literal;
pub use literal::*;

use crate::device_options::DeviceOptions;
use crate::uid::DeviceUid;

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
/// Components are one-based; a zero component is unspecified (wildcard not supported).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct GeoLocation {
    pub shelf: NonZeroU16,
    pub slot: NonZeroU16,
    pub frontend: NonZeroU16,
    pub port: NonZeroU16,
}

impl TryFrom<[u16; 4]> for GeoLocation {
    type Error = String;
    fn try_from([shelf, slot, frontend, port]: [u16; 4]) -> Result<Self, Self::Error> {
        let non_zero = |value, name: &str| {
            NonZeroU16::new(value).ok_or_else(|| format!("Invalid geolocation key: '{shelf}:{slot}:{frontend}:{port}'; '{name}' must be non-zero"))
        };
        Ok(Self {
            shelf: non_zero(shelf, "shelf")?,
            slot: non_zero(slot, "slot")?,
            frontend: non_zero(frontend, "frontend")?,
            port: non_zero(port, "port")?,
        })
    }
}

impl std::str::FromStr for GeoLocation {
    type Err = String;

    fn from_str(port_str: &str) -> Result<Self, Self::Err> {
        let invalid = || format!("Invalid geolocation key: '{port_str}'");
        let parts: Vec<u16> = port_str
            .split(':')
            .map(|p| p.parse::<u16>().map_err(|_| invalid()))
            .collect::<Result<_, _>>()?;
        let components: [u16; 4] = parts.try_into().map_err(|_| invalid())?;
        Self::try_from(components)
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

/// A device without signals, as used in the experiment -- see [`AuxiliaryDeviceKind`].
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

#[derive(Debug, Clone, PartialEq, Eq, Hash, Copy, PartialOrd, Ord)]
pub struct ChannelKey {
    device_uid: DeviceUid,
    channel: u16,
}

impl ChannelKey {
    pub fn new<T>(device_uid: DeviceUid, channel: T) -> Self
    where
        T: Into<u16>,
    {
        Self {
            device_uid,
            channel: channel.into(),
        }
    }

    pub fn device_uid(&self) -> DeviceUid {
        self.device_uid
    }

    pub fn channel(&self) -> u16 {
        self.channel
    }
}

impl Display for ChannelKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}", self.device_uid.0, self.channel)
    }
}

#[cfg(test)]
mod geolocation_tests {
    use super::*;
    use std::str::FromStr;

    #[test]
    fn from_str_valid() {
        let geo = GeoLocation::from_str("1:2:3:4").unwrap();
        assert_eq!(geo, GeoLocation::try_from([1, 2, 3, 4]).unwrap());
    }

    #[test]
    fn from_str_invalid() {
        assert!(GeoLocation::from_str("1:2:3").is_err());
        assert!(GeoLocation::from_str("1:2:3:4:5").is_err());
        assert!(GeoLocation::from_str("a:b:c:d").is_err());
    }

    #[test]
    fn display_roundtrip() {
        let result = GeoLocation::try_from([1, 2, 3, 4]).unwrap().to_string();
        assert_eq!(result, "1:2:3:4");
    }
}

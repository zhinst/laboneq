// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use laboneq_common::types::DeviceKind;
use laboneq_error::laboneq_error;

use crate::Result;

pub struct Port {
    pub channel: u8,
}

/// Parse a port string into a `Port` struct based on the device kind.
pub fn parse_port(port: &str, device: DeviceKind) -> Result<Port> {
    match device {
        DeviceKind::Shfsg => {
            parse_shfsg_port(port).ok_or_else(|| laboneq_error!("Invalid SHFSG port '{}'", port))
        }
        DeviceKind::Hdawg => {
            parse_channel(port).ok_or_else(|| laboneq_error!("Invalid HDAWG port '{}'", port))
        }
        DeviceKind::Shfqa => {
            parse_channel(port).ok_or_else(|| laboneq_error!("Invalid SHFQA port '{}'", port))
        }
        DeviceKind::Uhfqa => {
            parse_channel(port).ok_or_else(|| laboneq_error!("Invalid UHFQA port '{}'", port))
        }
        _ => Err(laboneq_error!("Unsupported device kind")),
    }
}

/// Parse SHFSG channel from a string.
fn parse_shfsg_port(port: &str) -> Option<Port> {
    const MAX_OUTPUT_CHANNEL: u8 = 8;

    let channel: Option<u8> = port
        .strip_prefix("SGCHANNELS/")?
        .strip_suffix("/OUTPUT")?
        .parse()
        .ok();
    if channel <= Some(MAX_OUTPUT_CHANNEL) {
        return Some(Port { channel: channel? });
    }
    None
}

/// Extract channel numbers from port names, e.g., "SIGOUTS/0" -> 0, "SIGOUTS/1" -> 1, etc.
///
/// TODO: Port the explicit parsing and validation from Python to Rust to avoid relying on the port naming convention here
fn parse_channel(port: &str) -> Option<Port> {
    let channel: Option<u8> = port.split('/').find_map(|part| part.parse().ok());
    channel.map(|channel| Port { channel })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_shfsg_port() {
        let port = parse_port("SGCHANNELS/0/OUTPUT", DeviceKind::Shfsg).unwrap();
        assert_eq!(port.channel, 0);

        let port = parse_port("SGCHANNELS/8/OUTPUT", DeviceKind::Shfsg).unwrap();
        assert_eq!(port.channel, 8);

        let port = parse_port("SGCHANNELS/9/OUTPUT", DeviceKind::Shfsg);
        assert!(port.is_err());
    }
}

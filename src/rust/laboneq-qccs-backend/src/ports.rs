// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use laboneq_dsl::device_setup::InstrumentKind;
use laboneq_error::laboneq_error;

use crate::Result;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Port {
    pub path: String,
    pub channel: u8,
    pub direction: IoDirection,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Copy)]
pub enum IoDirection {
    Input,
    Output,
}

/// Parse a port string into a [`Port`] based on the device kind.
pub fn parse_port(port: &str, device: InstrumentKind) -> Result<Port> {
    match device {
        InstrumentKind::Shfsg => {
            parse_shfsg_port(port).ok_or_else(|| laboneq_error!("Invalid SHFSG port '{}'", port))
        }
        InstrumentKind::Hdawg => {
            parse_hdawg_port(port).ok_or_else(|| laboneq_error!("Invalid HDAWG port '{}'", port))
        }
        InstrumentKind::Shfqa => {
            parse_shfqa_port(port).ok_or_else(|| laboneq_error!("Invalid SHFQA port '{}'", port))
        }
        InstrumentKind::Uhfqa => {
            parse_uhfqa_port(port).ok_or_else(|| laboneq_error!("Invalid UHFQA port '{}'", port))
        }
        InstrumentKind::Shfqc => parse_shfqa_port(port)
            .or_else(|| parse_shfsg_port(port))
            .ok_or_else(|| laboneq_error!("Invalid SHFQC port '{}'", port)),
        InstrumentKind::Shfppc => {
            parse_shfppc_port(port).ok_or_else(|| laboneq_error!("Invalid SHFPPC port '{}'", port))
        }
        device => Err(laboneq_error!(
            "Unsupported device: '{device:?}' for port parsing"
        )),
    }
}

/// Expect exactly one port in the given slice, returning an error if there are zero or multiple ports.
pub(crate) fn expect_one_port(ports: &[Port]) -> Result<&Port> {
    if ports.len() != 1 {
        return Err(laboneq_error!(
            "Expected exactly one port, found '{}'",
            ports
                .iter()
                .map(|p| p.path.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    Ok(&ports[0])
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
        return Some(Port {
            path: port.to_string(),
            channel: channel?,
            direction: IoDirection::Output,
        });
    }
    None
}

fn parse_shfqa_port(port: &str) -> Option<Port> {
    const MAX_CHANNEL: u8 = 4;

    let qa_prefix = port.strip_prefix("QACHANNELS/")?;
    if let Some(suffix) = qa_prefix.strip_suffix("/INPUT")
        && let Some(channel) = suffix.parse().ok()
        && channel < MAX_CHANNEL
    {
        return Some(Port {
            path: port.to_string(),
            channel,
            direction: IoDirection::Input,
        });
    }
    if let Some(suffix) = qa_prefix.strip_suffix("/OUTPUT")
        && let Some(channel) = suffix.parse().ok()
        && channel < MAX_CHANNEL
    {
        return Some(Port {
            path: port.to_string(),
            channel,
            direction: IoDirection::Output,
        });
    }
    None
}

fn parse_hdawg_port(port: &str) -> Option<Port> {
    const MAX_CHANNEL: u8 = 8;

    if let Some(channel) = port.strip_prefix("SIGOUTS/").and_then(|s| s.parse().ok())
        && channel < MAX_CHANNEL
    {
        return Some(Port {
            path: port.to_string(),
            channel,
            direction: IoDirection::Output,
        });
    }
    None
}

fn parse_uhfqa_port(port: &str) -> Option<Port> {
    const MAX_CHANNEL: u8 = 2;

    if let Some(channel) = port.strip_prefix("SIGINS/").and_then(|s| s.parse().ok())
        && channel < MAX_CHANNEL
    {
        return Some(Port {
            path: port.to_string(),
            channel,
            direction: IoDirection::Input,
        });
    }
    if let Some(channel) = port.strip_prefix("SIGOUTS/").and_then(|s| s.parse().ok())
        && channel < MAX_CHANNEL
    {
        return Some(Port {
            path: port.to_string(),
            channel,
            direction: IoDirection::Output,
        });
    }
    None
}

fn parse_shfppc_port(port: &str) -> Option<Port> {
    const MAX_CHANNEL: u8 = 4;

    if let Some(channel) = port
        .strip_prefix("PPCHANNELS/")
        .and_then(|s| s.parse().ok())
        && channel < MAX_CHANNEL
    {
        return Some(Port {
            path: port.to_string(),
            channel,
            direction: IoDirection::Output,
        });
    }
    None
}

pub(crate) fn is_shfsg_port(port: &Port) -> bool {
    port.path.starts_with("SGCHANNELS/")
}

pub(crate) fn is_shfqa_port(port: &Port) -> bool {
    port.path.starts_with("QACHANNELS/")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_shfsg_port() {
        let port = parse_port("SGCHANNELS/0/OUTPUT", InstrumentKind::Shfsg).unwrap();
        assert_eq!(port.channel, 0);

        let port = parse_port("SGCHANNELS/8/OUTPUT", InstrumentKind::Shfsg).unwrap();
        assert_eq!(port.channel, 8);

        let port = parse_port("SGCHANNELS/9/OUTPUT", InstrumentKind::Shfsg);
        assert!(port.is_err());
    }
}

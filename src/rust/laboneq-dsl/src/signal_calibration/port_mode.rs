// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::str::FromStr;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PortMode {
    LF,
    RF,
}

impl FromStr for PortMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "lf" => Ok(PortMode::LF),
            "rf" => Ok(PortMode::RF),
            _ => Err(format!("Unknown port mode: {}", s)),
        }
    }
}

impl std::fmt::Display for PortMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            PortMode::LF => "LF",
            PortMode::RF => "RF",
        };
        write!(f, "{}", s)
    }
}

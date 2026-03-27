// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use crate::types::{OscillatorUid, ValueOrParameter};

#[derive(Debug, Clone, PartialEq, Copy)]
pub enum OscillatorKind {
    Auto,
    Hardware,
    Software,
}

impl std::fmt::Display for OscillatorKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OscillatorKind::Auto => write!(f, "Auto"),
            OscillatorKind::Hardware => write!(f, "Hardware"),
            OscillatorKind::Software => write!(f, "Software"),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Oscillator {
    pub uid: OscillatorUid, // NOTE: Needed for legacy reasons, should be removed in future
    pub frequency: ValueOrParameter<f64>,
    pub kind: OscillatorKind,
}

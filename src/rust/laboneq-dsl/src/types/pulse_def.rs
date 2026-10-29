// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use crate::types::{NumericLiteral, PulseUid};
use numeric_array::NumericArray;

use laboneq_units::duration::{Duration, Sample, Second, samples};

#[derive(Debug, Clone)]
pub struct PulseDef {
    pub uid: PulseUid,
    pub kind: PulseKind,
    pub can_compress: bool,
    /// Amplitude as a numeric literal
    /// NOTE: `NumericLiteral` is used here to preserve the original representation
    /// (e.g., integer, float, complex) for accurate serialization to Python. This
    /// is due to the fact that `PulseDef` is serialized to Python as long as waveform
    /// sampling is implemented in Python.
    pub amplitude: NumericLiteral,
}

impl PulseDef {
    pub fn length(&self) -> PulseLength {
        match &self.kind {
            PulseKind::Functional(func) => PulseLength::Seconds(func.length),
            PulseKind::Sampled(obj) => PulseLength::Samples(samples(obj.samples.len())),
            PulseKind::LengthOnly { length } | PulseKind::MarkerPulse { length } => {
                PulseLength::Seconds(*length)
            }
        }
    }
}

#[derive(Debug, Clone)]
pub enum PulseKind {
    Functional(FunctionalPulse),
    Sampled(SampledPulse),
    LengthOnly { length: Duration<Second> },
    MarkerPulse { length: Duration<Second> },
}

#[derive(Debug, Clone, PartialEq)]
pub enum PulseFunction {
    /// LabOne Q's standard pulses.
    Builtin(BuiltinPulse),
    /// A custom-defined functional pulse.
    Custom { function: String },
}

impl PulseFunction {
    pub const CONSTANT_PULSE_NAME: &str = "const";

    pub fn name(&self) -> &str {
        match self {
            PulseFunction::Builtin(pulse) => pulse.name(),
            PulseFunction::Custom { function } => function,
        }
    }

    pub fn is_constant(&self) -> bool {
        self.name() == Self::CONSTANT_PULSE_NAME
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BuiltinPulse {
    Const,
}

impl BuiltinPulse {
    /// Returns the registered name under LabOne Q's pulse library.
    pub fn name(self) -> &'static str {
        match self {
            BuiltinPulse::Const => PulseFunction::CONSTANT_PULSE_NAME,
        }
    }

    /// Returns the standard pulse registered under `name`, if any.
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            PulseFunction::CONSTANT_PULSE_NAME => Some(BuiltinPulse::Const),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct FunctionalPulse {
    pub length: Duration<Second>,
    pub function: PulseFunction,
}

#[derive(Debug, Clone)]
pub struct SampledPulse {
    pub samples: NumericArray,
}

pub enum PulseLength {
    Seconds(Duration<Second>),
    Samples(Duration<Sample, usize>),
}

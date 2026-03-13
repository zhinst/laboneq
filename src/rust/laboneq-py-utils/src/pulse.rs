// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use laboneq_dsl::types::{NumericLiteral, PulseUid};
use numeric_array::NumericArray;

use laboneq_units::duration::{Duration, Second};

#[derive(Debug, Clone, PartialEq)]
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

#[derive(Debug, Clone, PartialEq)]
pub enum PulseKind {
    Functional(FunctionalPulse),
    Sampled(SampledPulse),
    LengthOnly { length: Duration<Second> },
    MarkerPulse { length: Duration<Second> },
}

#[derive(Debug, Clone, PartialEq)]
pub enum PulseFunction {
    Constant,
    Custom { function: String },
}

impl PulseFunction {
    pub const CONSTANT_PULSE_NAME: &str = "const";
}

#[derive(Debug, Clone, PartialEq)]
pub struct FunctionalPulse {
    pub length: Duration<Second>,
    pub function: PulseFunction,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SampledPulse {
    pub samples: NumericArray,
}

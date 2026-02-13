// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use laboneq_dsl::types::{NumericLiteral, PulseUid};
use pyo3::prelude::*;

use laboneq_units::duration::{Duration, Second};

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

#[derive(Debug, Clone)]
pub enum PulseKind {
    Functional(FunctionalPulse),
    Sampled(SampledPulse),
    LengthOnly { length: Duration<Second> },
    MarkerPulse { length: Duration<Second> },
}

#[derive(Debug, Clone)]
pub enum PulseFunction {
    Constant,
    Custom { function: String },
}

impl PulseFunction {
    pub const CONSTANT_PULSE_NAME: &str = "const";
}

#[derive(Debug, Clone)]
pub struct FunctionalPulse {
    pub length: Duration<Second>,
    pub function: PulseFunction,
}

#[derive(Debug, Clone)]
pub struct SampledPulse {
    pub samples: Arc<Py<PyAny>>,
    // Convenience field for length in samples
    pub length: usize,
}

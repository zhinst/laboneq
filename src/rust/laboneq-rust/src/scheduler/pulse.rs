// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use laboneq_dsl::types::{NumericLiteral, PulseUid};
use pyo3::prelude::*;

use laboneq_units::duration::{Duration, Second};

pub(crate) struct PulseDef {
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

pub(crate) enum PulseKind {
    Functional(PulseFunctional),
    Sampled(PulseSampled),
    LengthOnly { length: Duration<Second> },
    MarkerPulse { length: Duration<Second> },
}

pub(crate) enum PulseFunction {
    #[expect(dead_code, reason = "Not yet implemented")]
    Constant,
    Custom {
        function: String,
    },
}

impl PulseFunction {
    pub(crate) const CONSTANT_PULSE_NAME: &str = "const";
}

pub(crate) struct PulseFunctional {
    pub length: Duration<Second>,
    pub function: PulseFunction,
}

pub(crate) struct PulseSampled {
    pub samples: Py<PyAny>,
    // Convenience field for length in samples
    pub length: usize,
}

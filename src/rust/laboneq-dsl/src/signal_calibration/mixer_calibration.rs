// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use crate::types::ValueOrParameter;

#[derive(Debug, Clone, PartialEq)]
pub struct MixerCalibration {
    pub voltage_offset_i: Option<ValueOrParameter<f64>>,
    pub voltage_offset_q: Option<ValueOrParameter<f64>>,
    pub correction_matrix: Option<CorrectionMatrix>,
}

/// Correction matrix for mixer calibration, represented as a 2x2 matrix:
/// [ a00, a01 ]
/// [ a10, a11 ]
#[derive(Debug, Clone, PartialEq)]
pub struct CorrectionMatrix {
    pub a00: ValueOrParameter<f64>, // I gain
    pub a01: ValueOrParameter<f64>, // Q->I coupling
    pub a10: ValueOrParameter<f64>, // I->Q coupling
    pub a11: ValueOrParameter<f64>, // Q gain
}

impl CorrectionMatrix {
    /// Create a matrix from row-major array: [a00, a01, a10, a11]
    pub fn from_row_major(values: [ValueOrParameter<f64>; 4]) -> Self {
        Self {
            a00: values[0],
            a01: values[1],
            a10: values[2],
            a11: values[3],
        }
    }
}

impl MixerCalibration {
    /// Get I channel gains (diagonal, off_diagonal)
    pub fn gains_i(&self) -> Option<(ValueOrParameter<f64>, ValueOrParameter<f64>)> {
        self.correction_matrix.as_ref().map(|cm| (cm.a00, cm.a10))
    }

    /// Get Q channel gains (diagonal, off_diagonal)
    pub fn gains_q(&self) -> Option<(ValueOrParameter<f64>, ValueOrParameter<f64>)> {
        self.correction_matrix.as_ref().map(|cm| (cm.a11, cm.a01))
    }
}

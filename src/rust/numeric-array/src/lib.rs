// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use num_complex::Complex;

#[cfg(feature = "pyo3")]
mod py_bindings;

#[derive(Debug, Clone, PartialEq)]
pub enum NumericArray {
    Integer64(Vec<i64>),
    Float64(Vec<f64>),
    Complex64(Vec<Complex<f64>>),
}

impl NumericArray {
    pub fn abs_at_index(&self, index: usize) -> Option<f64> {
        match self {
            NumericArray::Integer64(vec) => vec.get(index).map(|x| x.abs() as f64),
            NumericArray::Float64(vec) => vec.get(index).map(|x| x.abs()),
            NumericArray::Complex64(vec) => vec.get(index).map(|x| x.norm()),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            NumericArray::Integer64(vec) => vec.len(),
            NumericArray::Float64(vec) => vec.len(),
            NumericArray::Complex64(vec) => vec.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        match self {
            NumericArray::Integer64(vec) => vec.is_empty(),
            NumericArray::Float64(vec) => vec.is_empty(),
            NumericArray::Complex64(vec) => vec.is_empty(),
        }
    }
}

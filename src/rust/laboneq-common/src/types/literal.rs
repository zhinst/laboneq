// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use num_complex::Complex64;

#[derive(Debug, Clone, PartialEq)]
pub enum Literal {
    Real(f64),
    Complex(Complex64),
    Integer(i64),
    Text(String),
}

impl From<f64> for Literal {
    fn from(value: f64) -> Self {
        Literal::Real(value)
    }
}

impl From<Complex64> for Literal {
    fn from(value: Complex64) -> Self {
        Literal::Complex(value)
    }
}

impl From<i64> for Literal {
    fn from(value: i64) -> Self {
        Literal::Integer(value)
    }
}

impl From<&str> for Literal {
    fn from(value: &str) -> Self {
        Literal::Text(value.to_string())
    }
}

impl From<String> for Literal {
    fn from(value: String) -> Self {
        Literal::Text(value)
    }
}

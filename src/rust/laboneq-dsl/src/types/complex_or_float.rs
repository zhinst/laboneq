// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use num_complex::Complex64;

use crate::types::NumericLiteral;

#[derive(Debug, Clone, Copy)]
pub enum ComplexOrFloat {
    Float(f64),
    Complex(Complex64),
}

impl Eq for ComplexOrFloat {}

impl std::fmt::Display for ComplexOrFloat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ComplexOrFloat::Float(v) => write!(f, "{}", v),
            ComplexOrFloat::Complex(v) => write!(f, "{} + {}j", v.re, v.im),
        }
    }
}

impl PartialEq for ComplexOrFloat {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (ComplexOrFloat::Float(a), ComplexOrFloat::Float(b)) => a == b,
            (ComplexOrFloat::Complex(a), ComplexOrFloat::Complex(b)) => a == b,
            _ => false,
        }
    }
}

impl TryFrom<NumericLiteral> for ComplexOrFloat {
    type Error = &'static str;
    fn try_from(value: NumericLiteral) -> Result<ComplexOrFloat, Self::Error> {
        match value {
            NumericLiteral::Float(v) => Ok(ComplexOrFloat::Float(v)),
            NumericLiteral::Complex(v) => Ok(ComplexOrFloat::Complex(v)),
            NumericLiteral::Int(_) => Err("Cannot convert integer to complex or float value"),
        }
    }
}

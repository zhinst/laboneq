// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use num_complex::Complex64;
use num_traits::cast::ToPrimitive;

#[derive(Debug, Clone, Copy)]
pub enum NumericLiteral {
    Int(i64),
    Float(f64),
    Complex(Complex64),
}

impl Eq for NumericLiteral {}

impl std::fmt::Display for NumericLiteral {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NumericLiteral::Int(v) => write!(f, "{}", v),
            NumericLiteral::Float(v) => write!(f, "{}", v),
            NumericLiteral::Complex(v) => write!(f, "{} + {}j", v.re, v.im),
        }
    }
}

impl TryFrom<NumericLiteral> for f64 {
    type Error = &'static str;
    fn try_from(value: NumericLiteral) -> Result<Self, Self::Error> {
        match value {
            NumericLiteral::Int(v) => v
                .to_f64()
                .ok_or("Integer value is too large to convert to f64"),
            NumericLiteral::Float(v) => Ok(v),
            NumericLiteral::Complex(v) => {
                if v.im == 0.0 {
                    Ok(v.re)
                } else {
                    Err("Cannot convert complex to f64")
                }
            }
        }
    }
}

impl From<f64> for NumericLiteral {
    fn from(value: f64) -> Self {
        NumericLiteral::Float(value)
    }
}

impl TryFrom<NumericLiteral> for usize {
    type Error = &'static str;
    fn try_from(value: NumericLiteral) -> Result<usize, Self::Error> {
        match value {
            NumericLiteral::Int(v) => usize::try_from(v)
                .map_err(|_| "Cannot convert negative integer to unsigned integer"),
            NumericLiteral::Float(v) => {
                if v.fract() == 0.0 {
                    usize::try_from(v as i64)
                        .map_err(|_| "Cannot convert negative float to unsigned integer")
                } else {
                    Err("Cannot convert float to unsigned integer")
                }
            }
            NumericLiteral::Complex(_) => Err("Cannot convert complex to unsigned integer"),
        }
    }
}

impl TryFrom<NumericLiteral> for Complex64 {
    type Error = &'static str;
    fn try_from(value: NumericLiteral) -> Result<Complex64, Self::Error> {
        match value {
            NumericLiteral::Int(v) => Ok(Complex64::new(v as f64, 0.0)),
            NumericLiteral::Float(v) => Ok(Complex64::new(v, 0.0)),
            NumericLiteral::Complex(v) => Ok(v),
        }
    }
}

impl PartialEq for NumericLiteral {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (NumericLiteral::Int(a), NumericLiteral::Int(b)) => a == b,
            (NumericLiteral::Float(a), NumericLiteral::Float(b)) => a == b,
            (NumericLiteral::Complex(a), NumericLiteral::Complex(b)) => a == b,
            (NumericLiteral::Int(a), NumericLiteral::Float(b)) => *a as f64 == *b,
            (NumericLiteral::Float(a), NumericLiteral::Int(b)) => *a == *b as f64,
            _ => false,
        }
    }
}

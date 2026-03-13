// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use num_complex::Complex64;

use crate::types::{NumericLiteral, ParameterUid};

#[derive(Debug, Clone, PartialEq, Copy)]
pub enum ValueOrParameter<T> {
    Value(T),
    Parameter(ParameterUid),
    // Value resolved from parameter
    ResolvedParameter { value: T, uid: ParameterUid },
}

impl<T> ValueOrParameter<T> {
    /// If this is a fixed value (not a parameter), return it. Otherwise, return None.
    pub fn fixed_value(&self) -> Option<T>
    where
        T: Copy,
    {
        match self {
            ValueOrParameter::Value(v) => Some(*v),
            ValueOrParameter::ResolvedParameter { .. } => None,
            _ => None,
        }
    }
}

impl TryFrom<ValueOrParameter<f64>> for f64 {
    type Error = &'static str;

    fn try_from(value: ValueOrParameter<f64>) -> Result<Self, Self::Error> {
        match value {
            ValueOrParameter::Value(v) => Ok(v),
            ValueOrParameter::Parameter(_) => Err("Cannot convert Parameter to f64"),
            ValueOrParameter::ResolvedParameter { value, uid: _ } => Ok(value),
        }
    }
}

impl TryFrom<NumericLiteral> for ValueOrParameter<f64> {
    type Error = &'static str;

    fn try_from(value: NumericLiteral) -> Result<ValueOrParameter<f64>, Self::Error> {
        Ok(ValueOrParameter::Value(value.try_into()?))
    }
}

impl TryFrom<NumericLiteral> for ValueOrParameter<Complex64> {
    type Error = &'static str;

    fn try_from(value: NumericLiteral) -> Result<ValueOrParameter<Complex64>, Self::Error> {
        Ok(ValueOrParameter::Value(value.try_into()?))
    }
}

impl From<f64> for ValueOrParameter<f64> {
    fn from(value: f64) -> Self {
        ValueOrParameter::Value(value)
    }
}

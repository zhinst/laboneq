// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use numeric_array::NumericArray;
use std::sync::Arc;

use crate::types::{NumericLiteral, ParameterUid};

#[derive(Debug, Clone, PartialEq)]
pub struct SweepParameter {
    pub uid: ParameterUid,
    pub values: Arc<NumericArray>,
}

impl SweepParameter {
    pub fn new<T: Into<NumericArray>>(uid: ParameterUid, values: T) -> Self {
        Self {
            uid,
            values: Arc::new(values.into()),
        }
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn value_numeric_at_index(&self, index: usize) -> Option<NumericLiteral> {
        match self.values.as_ref() {
            NumericArray::Integer64(arr) => arr.get(index).map(|v| NumericLiteral::Int(*v)),
            NumericArray::Float64(arr) => arr.get(index).map(|v| NumericLiteral::Float(*v)),
            NumericArray::Complex64(arr) => arr.get(index).map(|v| NumericLiteral::Complex(*v)),
        }
    }

    pub fn slice(&self, range: std::ops::Range<usize>) -> Self {
        Self {
            uid: self.uid,
            values: self.values.slice(range).into(),
        }
    }
}

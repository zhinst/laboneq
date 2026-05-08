// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use numeric_array::NumericArray;
use std::sync::Arc;

use crate::types::{NumericLiteral, ParameterUid};
use laboneq_common::named_id::NamedId;

#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct SweepParameter {
    pub uid: ParameterUid,
    pub values: Arc<NumericArray>,
    pub axis_name: Option<NamedId>,
}

impl SweepParameter {
    pub fn new<T: Into<NumericArray>>(uid: ParameterUid, values: T) -> Result<Self, &'static str> {
        let values = values.into();
        if values.is_empty() {
            return Err("Sweep parameter length must be at least 1");
        }
        Ok(Self {
            uid,
            values: Arc::new(values),
            axis_name: None,
        })
    }

    pub fn new_with_axis_name<T: Into<NumericArray>>(
        uid: ParameterUid,
        values: T,
        axis_name: NamedId,
    ) -> Result<Self, &'static str> {
        let mut sweep_parameter = Self::new(uid, values)?;
        sweep_parameter.axis_name = Some(axis_name);
        Ok(sweep_parameter)
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
            axis_name: self.axis_name,
        }
    }

    pub fn values(&self) -> Box<dyn Iterator<Item = NumericLiteral> + '_> {
        match self.values.as_ref() {
            NumericArray::Integer64(arr) => Box::new(arr.iter().map(|v| NumericLiteral::Int(*v))),
            NumericArray::Float64(arr) => Box::new(arr.iter().map(|v| NumericLiteral::Float(*v))),
            NumericArray::Complex64(arr) => {
                Box::new(arr.iter().map(|v| NumericLiteral::Complex(*v)))
            }
        }
    }

    pub fn inner_values(&self) -> &Arc<NumericArray> {
        &self.values
    }
}

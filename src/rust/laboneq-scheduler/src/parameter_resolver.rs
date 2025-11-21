// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use crate::ParameterStore;
use crate::error::{Error, Result};
use crate::experiment::sweep_parameter::SweepParameter;
use crate::experiment::types::{NumericLiteral, ParameterUid};

pub struct ParameterResolver<'a> {
    parameters: &'a HashMap<ParameterUid, SweepParameter>,
    iteration: HashMap<ParameterUid, usize>,
    nt_parameters: &'a ParameterStore,
}

impl<'a> ParameterResolver<'a> {
    pub fn new(
        parameters: &'a HashMap<ParameterUid, SweepParameter>,
        nt_parameters: &'a ParameterStore,
    ) -> Self {
        Self {
            parameters,
            iteration: HashMap::new(),
            nt_parameters,
        }
    }

    pub fn set_iteration(&mut self, param: ParameterUid, index: usize) {
        self.iteration.insert(param, index);
    }

    pub fn current_iteration(&self, param: &ParameterUid) -> Result<usize> {
        if let Some(sweep_param) = self.iteration.get(param) {
            return Ok(*sweep_param);
        }
        if self.nt_parameters.get(param).is_some() {
            return Ok(0);
        }
        Err(Error::new(format!("Undefined parameter '{}'.", param.0)))
    }

    pub fn child_scope(&self) -> Self {
        Self {
            parameters: self.parameters,
            iteration: self.iteration.clone(),
            nt_parameters: self.nt_parameters,
        }
    }

    pub fn get_value(&self, param: &ParameterUid) -> Result<NumericLiteral> {
        if let Some(index) = self.iteration.get(param)
            && let Some(parameter) = self.parameters.get(param)
        {
            return Ok(parameter.value_numeric_at_index(*index).unwrap());
        }
        if let Some(value) = self.nt_parameters.get(param).cloned() {
            return Ok(value);
        }
        Err(Error::new(format!("Undefined parameter '{}'.", param.0)))
    }
}

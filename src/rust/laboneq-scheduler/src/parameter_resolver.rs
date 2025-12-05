// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};

use crate::ParameterStore;
use crate::error::{Error, Result};
use crate::experiment::sweep_parameter::SweepParameter;
use crate::experiment::types::{NumericLiteral, ParameterUid};

pub(crate) struct ParameterResolver<'a> {
    parameters: &'a HashMap<ParameterUid, SweepParameter>,
    iteration: HashMap<ParameterUid, usize>,
    nt_parameters: &'a ParameterStore,
    available_parameters: HashSet<ParameterUid>,
}

impl<'a> ParameterResolver<'a> {
    pub(crate) fn new(
        parameters: &'a HashMap<ParameterUid, SweepParameter>,
        nt_parameters: &'a ParameterStore,
    ) -> Self {
        Self {
            parameters,
            iteration: HashMap::new(),
            nt_parameters,
            available_parameters: nt_parameters.available_parameters(),
        }
    }

    /// Sets the current iteration index for a sweep parameter.
    pub(crate) fn set_iteration(&mut self, param: ParameterUid, index: usize) -> Result<()> {
        self.check_parameter_availability(&param)?;
        self.iteration.insert(param, index);
        Ok(())
    }

    /// Returns the current iteration index for a sweep parameter.
    pub(crate) fn current_iteration(&self, param: &ParameterUid) -> Result<usize> {
        self.check_parameter_availability(param)?;
        if let Some(sweep_param) = self.iteration.get(param) {
            return Ok(*sweep_param);
        }
        if self.nt_parameters.get(param).is_some() {
            return Ok(0);
        }
        unreachable!("Undefined parameter '{}'.", param.0)
    }

    /// Creates a child scope of the parameter resolver with additional available parameters.
    pub(crate) fn child_scope(&self, available_parameters: &[ParameterUid]) -> Result<Self> {
        Ok(Self {
            parameters: self.parameters,
            iteration: self.iteration.clone(),
            nt_parameters: self.nt_parameters,
            available_parameters: self.merge_available_parameters(available_parameters),
        })
    }

    /// Returns the sweep parameter associated with the given UID.
    pub(crate) fn try_resolve_parameter(&self, uid: &ParameterUid) -> Result<&SweepParameter> {
        self.check_parameter_availability(uid)?;
        Ok(self
            .parameters
            .get(uid)
            .unwrap_or_else(|| panic!("Undefined parameter '{}'.", uid.0)))
    }

    /// Get the numeric value of a parameter at the current iteration of the sweep.
    pub(crate) fn get_value(&self, param: &ParameterUid) -> Result<NumericLiteral> {
        self.check_parameter_availability(param)?;
        if let Some(index) = self.iteration.get(param)
            && let Some(parameter) = self.parameters.get(param)
        {
            return Ok(parameter.value_numeric_at_index(*index).unwrap());
        }
        if let Some(value) = self.nt_parameters.get(param).cloned() {
            return Ok(value);
        }
        unreachable!("Undefined parameter '{}'.", param.0)
    }

    fn merge_available_parameters(&self, other: &[ParameterUid]) -> HashSet<ParameterUid> {
        let mut this = self.available_parameters.clone();
        this.extend(other);
        this
    }

    fn check_parameter_availability(&self, uid: &ParameterUid) -> Result<()> {
        if !self.available_parameters.contains(uid) {
            return Err(Error::new(format!("Undefined parameter '{}'.", uid.0)));
        }
        Ok(())
    }
}

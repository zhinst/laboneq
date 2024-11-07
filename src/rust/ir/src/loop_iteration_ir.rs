// Copyright 2024 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::{
    collections::HashSet,
    sync::{Arc, Mutex},
};

use pyo3::{types::PyList, Py, PyAny, Python};

use crate::{
    common::{py_deep_copy, RuntimeError},
    interval_ir::IntervalIr,
    DeepCopy,
};

#[derive(Debug)]
pub struct LoopIterationPreambleIr {
    pub interval: Arc<Mutex<IntervalIr>>,
}

impl DeepCopy for LoopIterationPreambleIr {
    fn deep_copy(&self) -> Result<Self, RuntimeError> {
        let guard = self.interval.lock().map_err(|_| RuntimeError::Lock())?;
        Ok(Self {
            interval: Arc::new(Mutex::new(guard.deep_copy()?)),
        })
    }
}

#[derive(Debug)]
pub struct LoopIterationIr {
    pub interval: Arc<Mutex<IntervalIr>>,

    pub section: String,
    pub trigger_output: HashSet<(String, i64)>,
    pub prng_setup: Option<Py<PyAny>>,

    pub iteration: i64,
    pub sweep_parameters: Py<PyList>,
    pub num_repeats: i64,
    pub shadow: bool,
    pub prng_sample: Option<String>,
}

impl DeepCopy for LoopIterationIr {
    fn deep_copy(&self) -> Result<Self, RuntimeError> {
        let prng_clone = match self.prng_setup {
            Some(ref prng) => match py_deep_copy(prng) {
                Ok(prng) => Some(prng),
                Err(_) => return Err(RuntimeError::PyhonDeepCop()),
            },
            None => None,
        };

        let sweep_parameters_clone = py_deep_copy(self.sweep_parameters.as_any())
            .map_err(|_| RuntimeError::PyhonDeepCop())?;
        let sweep_parameters_clone = Python::with_gil(|py| {
            sweep_parameters_clone
                .extract::<Py<PyList>>(py)
                .map_err(|_| RuntimeError::PyhonExtraction())
        })?;

        let guard = self.interval.lock().map_err(|_| RuntimeError::Lock())?;

        Ok(Self {
            interval: Arc::new(Mutex::new(guard.deep_copy()?)),
            section: self.section.clone(),
            trigger_output: self.trigger_output.clone(),
            prng_setup: prng_clone,
            iteration: self.iteration,
            sweep_parameters: sweep_parameters_clone,
            num_repeats: self.num_repeats,
            shadow: self.shadow,
            prng_sample: self.prng_sample.clone(),
        })
    }
}

// Copyright 2024 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::sync::{Arc, Mutex};

use pyo3::{types::PyList, Py, Python};

use crate::{
    common::{py_deep_copy, RuntimeError},
    interval_ir::IntervalIr,
    DeepCopy,
};

#[derive(Debug)]
pub struct SetOscillatorFrequencyIr {
    pub interval: Arc<Mutex<IntervalIr>>,

    pub section: String,
    pub oscillators: Py<PyList>,
    pub params: Vec<String>,
    pub values: Vec<f64>,
    pub iteration: i64,
}

impl DeepCopy for SetOscillatorFrequencyIr {
    fn deep_copy(&self) -> Result<Self, RuntimeError> {
        let guard = self.interval.lock().map_err(|_| RuntimeError::Lock())?;

        let oscillators_clone =
            py_deep_copy(self.oscillators.as_any()).map_err(|_| RuntimeError::PyhonDeepCop())?;
        let oscillators_clone = Python::with_gil(|py| {
            oscillators_clone
                .extract::<Py<PyList>>(py)
                .map_err(|_| RuntimeError::PyhonExtraction())
        })?;

        Ok(Self {
            interval: Arc::new(Mutex::new(guard.deep_copy()?)),
            section: self.section.clone(),
            oscillators: oscillators_clone,
            params: self.params.clone(),
            values: self.values.clone(),
            iteration: self.iteration,
        })
    }
}

#[derive(Debug)]
pub struct InitialOscillatorFrequencyIr {
    pub interval: Arc<Mutex<IntervalIr>>,

    pub section: String,
    pub oscillators: Py<PyList>,
    pub values: Vec<f64>,
}

impl DeepCopy for InitialOscillatorFrequencyIr {
    fn deep_copy(&self) -> Result<Self, RuntimeError> {
        let guard = self.interval.lock().map_err(|_| RuntimeError::Lock())?;

        let oscillators_clone =
            py_deep_copy(self.oscillators.as_any()).map_err(|_| RuntimeError::PyhonDeepCop())?;
        let oscillators_clone = Python::with_gil(|py| {
            oscillators_clone
                .extract::<Py<PyList>>(py)
                .map_err(|_| RuntimeError::PyhonExtraction())
        })?;

        Ok(Self {
            interval: Arc::new(Mutex::new(guard.deep_copy()?)),
            section: self.section.clone(),
            oscillators: oscillators_clone,
            values: self.values.clone(),
        })
    }
}

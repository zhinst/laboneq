// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

//! Result module for the code generator.
//!
//! This module defines the `AwgCodeGenerationResultPy` class, which is used to
//! represent the result of the code generation process for an AWG.

use crate::awg_event::AwgEvent;
use pyo3::prelude::*;

/// Result structure for single AWG code generation.
#[pyclass(name = "AwgCodeGenerationResult", frozen, unsendable)]
pub struct AwgCodeGenerationResultPy {
    awg_events: Vec<Py<AwgEvent>>,
}

impl AwgCodeGenerationResultPy {
    fn new(awg_events: Vec<Py<AwgEvent>>) -> Self {
        AwgCodeGenerationResultPy { awg_events }
    }

    pub fn create(awg_events: Vec<AwgEvent>) -> PyResult<Self> {
        Python::with_gil(|py| {
            let py_awg_events: Vec<Py<AwgEvent>> = awg_events
                .into_iter()
                .map(|event| Py::new(py, event).expect("Failed to create AwgEvent"))
                .collect();
            let output = AwgCodeGenerationResultPy::new(py_awg_events);
            Ok(output)
        })
    }

    pub fn default() -> Self {
        AwgCodeGenerationResultPy { awg_events: vec![] }
    }
}

#[pymethods]
impl AwgCodeGenerationResultPy {
    #[getter]
    fn awg_events(&self) -> &Vec<Py<AwgEvent>> {
        &self.awg_events
    }
}

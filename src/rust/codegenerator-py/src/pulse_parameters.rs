// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use codegenerator::ir::experiment::PulseParametersId;
use pyo3::types::PyDict;
use pyo3::{intern, prelude::*};

#[derive(Debug, Clone)]
pub struct PulseParameters {
    id: PulseParametersId,
    // Original parameters from pulse (Python dictionary)
    pulse_parameters: Arc<Py<PyAny>>,
    // Original parameters from play (Python dictionary)
    play_parameters: Arc<Py<PyAny>>,
    // Combined parameters (Python dictionary)
    parameters: Arc<Py<PyAny>>,
}

impl PulseParameters {
    pub fn id(&self) -> PulseParametersId {
        self.id
    }

    pub fn parameters(&self) -> &Py<PyAny> {
        &self.parameters
    }

    pub fn pulse_parameters(&self) -> &Py<PyAny> {
        &self.pulse_parameters
    }

    pub fn play_parameters(&self) -> &Py<PyAny> {
        &self.play_parameters
    }
}

pub(crate) fn create_pulse_parameters<'a>(
    py: Python,
    pulse_parameters: Option<&Bound<'a, PyDict>>,
    play_parameters: Option<&Bound<'a, PyDict>>,
) -> PyResult<PulseParameters> {
    let combine_pulse_parameters = py
        .import(intern!(py, "laboneq.core.utilities.pulse_sampler"))?
        .getattr(intern!(py, "combine_pulse_parameters"))?;
    let create_pulse_parameters_id = py
        .import(intern!(py, "laboneq.compiler.common.pulse_parameters"))?
        .getattr(intern!(py, "create_pulse_parameters_id"))?;
    let merged_parameters =
        combine_pulse_parameters.call1((pulse_parameters, py.None(), play_parameters))?;
    let id = create_pulse_parameters_id
        .call1((pulse_parameters, play_parameters))?
        .extract::<u64>()?;
    Ok(PulseParameters {
        id: PulseParametersId(id),
        pulse_parameters: pulse_parameters
            .map(|p| Arc::new(p.clone().unbind().into()))
            .unwrap_or_else(|| Arc::new(PyDict::new(py).into())),
        play_parameters: play_parameters
            .map(|p| Arc::new(p.clone().unbind().into()))
            .unwrap_or_else(|| Arc::new(PyDict::new(py).into())),
        parameters: Arc::new(merged_parameters.unbind()),
    })
}

#[pyclass]
pub struct PulseParametersPy {
    pub parameters: PulseParameters,
}

#[pymethods]
impl PulseParametersPy {
    #[getter]
    pub fn parameters(&self) -> &Py<PyAny> {
        self.parameters.parameters()
    }

    #[getter]
    pub fn pulse_parameters(&mut self) -> &Py<PyAny> {
        self.parameters.pulse_parameters()
    }

    #[getter]
    pub fn play_parameters(&mut self) -> &Py<PyAny> {
        self.parameters.play_parameters()
    }
}

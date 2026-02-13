// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use codegenerator::ir::experiment::PulseParametersId;
use codegenerator_utils::pulse_parameters::PulseParameters;
use laboneq_common::named_id::NamedIdStore;
use laboneq_dsl::types::ExternalParameterUid;
use laboneq_py_utils::{
    py_export::pulse_parameters_to_py_dict, py_object_interner::PyObjectInterner,
};
use pyo3::prelude::*;

#[derive(Debug, Clone)]
#[pyclass(name = "PulseParameters")]
pub struct PulseParametersPy {
    pub id: PulseParametersId,
    // Original parameters from pulse (Python dictionary)
    pub pulse_parameters: Arc<Py<PyAny>>,
    // Original parameters from play (Python dictionary)
    pub play_parameters: Arc<Py<PyAny>>,
    // Combined parameters (Python dictionary)
    pub parameters: Arc<Py<PyAny>>,
}

#[pymethods]
impl PulseParametersPy {
    #[getter]
    pub fn parameters(&self) -> &Py<PyAny> {
        &self.parameters
    }

    #[getter]
    pub fn pulse_parameters(&self) -> &Py<PyAny> {
        &self.pulse_parameters
    }

    #[getter]
    pub fn play_parameters(&self) -> &Py<PyAny> {
        &self.play_parameters
    }
}

pub(crate) fn pulse_parameters_to_py(
    py: Python,
    parameters: &PulseParameters,
    id_store: &NamedIdStore,
    py_object_store: &PyObjectInterner<ExternalParameterUid>,
) -> PulseParametersPy {
    let pulse_parameters =
        pulse_parameters_to_py_dict(py, &parameters.pulse_parameters, id_store, py_object_store)
            .unwrap();
    let play_parameters =
        pulse_parameters_to_py_dict(py, &parameters.play_parameters, id_store, py_object_store)
            .unwrap();
    let all_parameters =
        pulse_parameters_to_py_dict(py, &parameters.parameters, id_store, py_object_store).unwrap();
    PulseParametersPy {
        id: PulseParametersId(parameters.id),
        pulse_parameters: Arc::new(pulse_parameters.into()),
        play_parameters: Arc::new(play_parameters.into()),
        parameters: Arc::new(all_parameters.into()),
    }
}

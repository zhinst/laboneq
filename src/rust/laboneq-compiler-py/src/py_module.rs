// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

//! The Python module exposed by this crate.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use laboneq_dsl::types::{NumericLiteral, ParameterUid};
use laboneq_error::LabOneQError;

use crate::error::CompilationError;
use crate::py_experiment::ExperimentPy;
use crate::py_pulse_sheet_schedule::PulseSheetSchedulePy;
use crate::rt_compiler::{RealTimeCompilerInput, compile_realtime};

pub fn create_py_module<'py>(py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyModule>> {
    use laboneq_opentelemetry_python::span_buffer::SpanBufferPy;

    let m = PyModule::new(py, name)?;
    m.add_function(wrap_pyfunction!(compile_realtime_py, &m)?)?;
    m.add_function(wrap_pyfunction!(init_logging_py, &m)?)?;
    m.add_class::<SpanBufferPy>()?;
    Ok(m)
}

#[pyclass(name = "RealTimeCompilerOutput", frozen)]
struct RealTimeCompilerOutputPy {
    /// Parameters used in the experiment
    #[pyo3(get)]
    used_parameters: HashSet<String>,
    #[pyo3(get)]
    pulse_sheet_schedule: Option<Py<PulseSheetSchedulePy>>,
    codegen_output: Py<PyAny>,
}

#[pymethods]
impl RealTimeCompilerOutputPy {
    fn codegen_output<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Ok(self.codegen_output.bind(py).into())
    }
}

#[pyfunction(name = "compile_realtime")]
fn compile_realtime_py(
    py: Python,
    experiment: &ExperimentPy,
    parameters: HashMap<String, f64>,
    chunking_info: Option<(usize, usize)>, // Current chunk index, total chunk count
) -> PyResult<RealTimeCompilerOutputPy> {
    let rt_input = RealTimeCompilerInput {
        experiment: &experiment.inner,
        parameters: parameters
            .into_iter()
            .map(|(k, v)| {
                let uid = experiment.inner.id_store.get(&k).ok_or_else(|| {
                    laboneq_error::laboneq_error!("Parameter {k} not found in experiment ID store")
                })?;
                Ok((ParameterUid(uid), NumericLiteral::Float(v)))
            })
            .collect::<Result<HashMap<ParameterUid, NumericLiteral>, LabOneQError>>()?,
        chunking_info,
        backend: experiment.backend.as_ref(),
        backend_data: experiment.backend_data.as_ref(),
        signals: &experiment.signals,
        compiler_settings: &experiment.compiler_settings,
    };

    let result = compile_realtime(rt_input)
        .map_err(|e| CompilationError::new(e, Arc::clone(&experiment.inner.id_store)))?;

    let result_py = RealTimeCompilerOutputPy {
        used_parameters: result
            .used_parameters
            .iter()
            .map(|p| experiment.inner.id_store.resolve(p.0).unwrap().to_string())
            .collect(),
        pulse_sheet_schedule: result
            .pulse_sheet_schedule
            .map(|inner| Py::new(py, PulseSheetSchedulePy { inner }))
            .transpose()?,
        codegen_output: result.codegen_output.to_python(py)?.into(),
    };
    Ok(result_py)
}

#[pyfunction(name = "init_logging")]
fn init_logging_py(log_level: i64) -> PyResult<()> {
    laboneq_py_utils::logging::init_logging_py(log_level)
}

// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use pyo3::prelude::*;

use laboneq_compiled_experiment_py::{compiled_experiment_from_capnp, peek_backend_name};
use laboneq_qccs_codegen_artifacts_py::qccs_artifacts_from_capnp;

/// Builds the `compiled_experiment` Python module.
pub(crate) fn create_py_module<'py>(py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyModule>> {
    let m = PyModule::new(py, name)?;
    m.add_function(wrap_pyfunction!(deserialize_py, &m)?)?;
    m.add_function(wrap_pyfunction!(peek_backend_name_py, &m)?)?;
    Ok(m)
}

/// Deserialize `CompiledExperiment` Cap'n Proto bytes produced by the QCCS backend into a
/// `ScheduledExperiment`.
#[pyfunction(name = "deserialize")]
fn deserialize_py<'py>(py: Python<'py>, capnp_data: &[u8]) -> PyResult<Bound<'py, PyAny>> {
    Ok(compiled_experiment_from_capnp(
        py,
        capnp_data,
        qccs_artifacts_from_capnp,
    )?)
}

/// Reads just the producing backend's identifier out of a serialized message, without
/// decoding anything else.
#[pyfunction]
#[pyo3(name = "peek_backend_name")]
fn peek_backend_name_py(capnp_data: &[u8]) -> PyResult<String> {
    Ok(peek_backend_name(capnp_data)?)
}

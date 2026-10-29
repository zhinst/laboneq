// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use pyo3::{
    prelude::*,
    types::{PyBytes, PyDict},
};

use laboneq_compiler_py::compile_experiment;
use laboneq_experiment_py::serialize_experiment;
use laboneq_qccs_backend::{BACKEND_NAME, QccsBackend, setup_description_py};

/// Serialize a Python experiment object to Cap'n Proto bytes, with `setup_description` in the
/// QCCS setup-description schema.
#[pyfunction(name = "serialize_experiment", signature = (experiment, setup_description, packed=false))]
pub(crate) fn serialize_experiment_py<'py>(
    py: Python<'py>,
    experiment: &Bound<'_, PyAny>,
    setup_description: &Bound<'_, PyAny>,
    packed: bool,
) -> PyResult<Bound<'py, PyBytes>> {
    let message = setup_description_py::message_from_py(setup_description)?;
    let setup_description = message.get_root_as_reader().map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("Invalid setup description: {e}"))
    })?;
    serialize_experiment(py, experiment, BACKEND_NAME, setup_description, packed)
}

/// Build an experiment from Cap'n Proto bytes plus device/signal configuration, and
/// compile it into `CompiledExperiment` Cap'n Proto bytes.
///
/// `packed` describes `capnp_data` only; the returned bytes are always unpacked.
#[pyfunction(name = "compile_experiment", signature = (capnp_data, packed=false, compiler_settings=None))]
pub(crate) fn compile_experiment_py(
    py: Python<'_>,
    capnp_data: &[u8],
    packed: bool,
    compiler_settings: Option<Bound<'_, PyDict>>,
) -> PyResult<Vec<u8>> {
    compile_experiment(
        py,
        capnp_data,
        packed,
        compiler_settings,
        QccsBackend::default(),
    )
}

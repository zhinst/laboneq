// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use pyo3::{prelude::*, types::PyDict};

use laboneq_compiler_py::compile_experiment;
use laboneq_qccs_backend::QccsBackend;

/// Build an experiment from Cap'n Proto bytes plus device/signal configuration.
#[pyfunction(name = "compile_experiment", signature = (capnp_data, packed=false, compiler_settings=None))]
pub(crate) fn compile_experiment_py<'py>(
    py: Python<'py>,
    capnp_data: &[u8],
    packed: bool,
    compiler_settings: Option<Bound<'_, PyDict>>,
) -> PyResult<Bound<'py, PyAny>> {
    let exp_py = compile_experiment(
        py,
        capnp_data,
        packed,
        compiler_settings,
        QccsBackend::default(),
    )?;
    Ok(exp_py)
}

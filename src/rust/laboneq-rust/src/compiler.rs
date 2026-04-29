// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use pyo3::{prelude::*, types::PyDict};

use laboneq_compiler_py::{build_experiment_with_backend_capnp_py, py_experiment::ExperimentPy};
use laboneq_qccs_backend::QccsBackend;

/// Build an experiment from Cap'n Proto bytes plus device/signal configuration.
#[pyfunction(name = "build_experiment_capnp", signature = (capnp_data, desktop_setup, packed=false, compiler_settings=None))]
pub(crate) fn build_experiment_capnp_py(
    py: Python<'_>,
    capnp_data: &[u8],
    desktop_setup: bool,
    packed: bool,
    compiler_settings: Option<Bound<'_, PyDict>>,
) -> PyResult<ExperimentPy> {
    let exp_py = build_experiment_with_backend_capnp_py(
        py,
        capnp_data,
        desktop_setup,
        packed,
        compiler_settings,
        QccsBackend::default(),
    )?;
    Ok(exp_py)
}

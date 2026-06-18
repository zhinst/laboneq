// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use laboneq_py_utils::device_setup_fingerprint::device_setup_fingerprint_py;

pub(crate) fn create_py_module<'a>(py: Python<'a>, name: &str) -> PyResult<Bound<'a, PyModule>> {
    let m = PyModule::new(py, name)?;
    m.add_function(wrap_pyfunction!(device_setup_fingerprint_py, &m)?)?;
    Ok(m)
}

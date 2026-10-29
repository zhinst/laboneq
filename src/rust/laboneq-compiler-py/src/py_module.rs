// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

//! The Python module exposed by this crate.

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

pub fn create_py_module<'py>(py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyModule>> {
    use laboneq_opentelemetry_python::span_buffer::SpanBufferPy;

    let m = PyModule::new(py, name)?;
    m.add_function(wrap_pyfunction!(init_logging_py, &m)?)?;
    m.add_class::<SpanBufferPy>()?;
    Ok(m)
}

#[pyfunction(name = "init_logging")]
fn init_logging_py(log_level: i64) -> PyResult<()> {
    laboneq_py_utils::logging::init_logging_py(log_level)
}

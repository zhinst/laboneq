// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use laboneq_scheduler::experiment::types::NumericLiteral;
use pyo3::{prelude::*, types::PyComplex};

/// Convert a [`NumericLiteral`] to a Python object.
pub(crate) fn numeric_literal_to_py(py: Python, value: &NumericLiteral) -> PyResult<Py<PyAny>> {
    match value {
        NumericLiteral::Int(v) => Ok(v.into_pyobject(py)?.unbind().into()),
        NumericLiteral::Float(v) => Ok(v.into_pyobject(py)?.unbind().into()),
        NumericLiteral::Complex(v) => Ok(PyComplex::from_doubles(py, v.re, v.im)
            .into_pyobject(py)?
            .unbind()
            .into()),
    }
}

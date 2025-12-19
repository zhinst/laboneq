// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use pyo3::{prelude::*, types::PyComplex};

use laboneq_scheduler::experiment::types::ComplexOrFloat;
use laboneq_scheduler::experiment::types::NumericLiteral;

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

/// Convert a [`ComplexOrFloat`] to a Python object.
pub(crate) fn complex_or_float_to_py(py: Python, value: &ComplexOrFloat) -> PyResult<Py<PyAny>> {
    match value {
        ComplexOrFloat::Float(v) => Ok(v.into_pyobject(py)?.unbind().into()),
        ComplexOrFloat::Complex(v) => Ok(PyComplex::from_doubles(py, v.re, v.im)
            .into_pyobject(py)?
            .unbind()
            .into()),
    }
}

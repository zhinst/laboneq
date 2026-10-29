// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use num_complex::Complex64;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyBytes, PyComplex, PyFloat};

pub enum PyConstant<'a> {
    None,
    Float(f64),
    Int(i64),
    Complex(Complex64),
    Str(&'a str),
    RawBytes(&'a [u8]),
    Bool(bool),
    /// Any other value, handed over as JSON.
    Opaque,
}

/// Classifies `obj` as a readable constant for the Compiler.
#[inline]
pub fn classify_constant<'a>(obj: &'a Bound<'_, PyAny>) -> PyResult<PyConstant<'a>> {
    if obj.is_none() {
        return Ok(PyConstant::None);
    }
    // Floats are the most common values, so check them first.
    // This does not cast ints to float, so it can stay before the int extraction.
    if let Ok(value) = obj.cast::<PyFloat>() {
        return Ok(PyConstant::Float(value.value()));
    }
    if let Ok(value) = obj.extract::<i64>() {
        return Ok(PyConstant::Int(value));
    }
    if obj.is_instance_of::<PyComplex>() {
        return Ok(PyConstant::Complex(obj.extract()?));
    }
    if let Ok(value) = obj.extract::<&str>() {
        return Ok(PyConstant::Str(value));
    }
    if let Ok(value) = obj.cast::<PyBytes>() {
        return Ok(PyConstant::RawBytes(value.as_bytes()));
    }
    classify_number(obj)
}

/// Classifies numbers of other types, such as numpy's `float32` and `complex64`, by the
/// abstract base class they register with. Integers beyond `i64` stay opaque, so they
/// are not rounded to a float.
#[cold]
fn classify_number<'a>(obj: &'a Bound<'_, PyAny>) -> PyResult<PyConstant<'a>> {
    let py = obj.py();
    let numbers = py.import(intern!(py, "numbers"))?;
    if obj.is_instance(&numbers.getattr(intern!(py, "Integral"))?)? {
        return Ok(PyConstant::Opaque);
    }
    if obj.is_instance(&numbers.getattr(intern!(py, "Real"))?)? {
        return Ok(PyConstant::Float(obj.extract()?));
    }
    if obj.is_instance(&numbers.getattr(intern!(py, "Complex"))?)? {
        return Ok(PyConstant::Complex(obj.extract()?));
    }
    Ok(PyConstant::Opaque)
}

/// Classifies `obj` as [`classify_constant`] does, except a `bool`,
/// which becomes a [`PyConstant::Bool`] instead of `1` or `0`.
#[inline]
pub fn classify_value<'a>(obj: &'a Bound<'_, PyAny>) -> PyResult<PyConstant<'a>> {
    if let Ok(value) = obj.cast::<PyBool>() {
        return Ok(PyConstant::Bool(value.is_true()));
    }
    classify_constant(obj)
}

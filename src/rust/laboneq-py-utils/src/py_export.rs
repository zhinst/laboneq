// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

//! This crate provides common utilities for exporting LabOneQ data structures to Python.

use std::collections::HashMap;

use laboneq_common::named_id::NamedIdStore;
use laboneq_dsl::{
    operation::PulseParameterValue,
    types::{
        ComplexOrFloat, ExternalParameterUid, NumericLiteral, PulseParameterUid, ValueOrParameter,
    },
};
use pyo3::{
    prelude::*,
    types::{PyComplex, PyDict},
};

use crate::{
    pulse::{PulseDef, PulseFunction, PulseKind},
    py_object_interner::PyObjectInterner,
};

/// Convert a [`NumericLiteral`] to a Python object.
pub fn numeric_literal_to_py(py: Python, value: &NumericLiteral) -> PyResult<Py<PyAny>> {
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
pub fn complex_or_float_to_py(py: Python, value: &ComplexOrFloat) -> PyResult<Py<PyAny>> {
    match value {
        ComplexOrFloat::Float(v) => Ok(v.into_pyobject(py)?.unbind().into()),
        ComplexOrFloat::Complex(v) => Ok(PyComplex::from_doubles(py, v.re, v.im)
            .into_pyobject(py)?
            .unbind()
            .into()),
    }
}

/// Convert a [`PulseDef`] to a Python object.
pub fn pulse_def_to_py(
    py: Python,
    id_store: &NamedIdStore,
    pulse_def: &PulseDef,
) -> PyResult<Py<PyAny>> {
    let m = py.import("laboneq")?;
    let pulse_def_py_cls = m
        .getattr("data")
        .and_then(|m| m.getattr("compilation_job"))
        .and_then(|m| m.getattr("PulseDef"))?;

    let kwargs = PyDict::new(py);
    kwargs.set_item("uid", id_store.resolve(pulse_def.uid).unwrap())?;
    kwargs.set_item(
        "amplitude",
        numeric_literal_to_py(py, &pulse_def.amplitude)?,
    )?;
    kwargs.set_item("can_compress", pulse_def.can_compress)?;

    match &pulse_def.kind {
        PulseKind::Functional(func) => {
            let function_name = match func.function {
                PulseFunction::Constant => "const",
                PulseFunction::Custom { ref function } => function.as_str(),
            };
            kwargs.set_item("function", function_name)?;
            kwargs.set_item("length", func.length.value())?;
            kwargs.set_item("samples", py.None())?;
        }
        PulseKind::Sampled(sampled) => {
            kwargs.set_item("samples", sampled.samples.clone_ref(py))?;
            kwargs.set_item("function", py.None())?;
            kwargs.set_item("length", py.None())?;
        }
        PulseKind::LengthOnly { length } => {
            kwargs.set_item("samples", py.None())?;
            kwargs.set_item("function", py.None())?;
            kwargs.set_item("length", length.value())?;
        }
        PulseKind::MarkerPulse { length } => {
            kwargs.set_item("samples", py.None())?;
            kwargs.set_item("function", "const")?;
            kwargs.set_item("length", length.value())?;
        }
    }
    let pulse_def_py = pulse_def_py_cls.call((), Some(&kwargs))?;
    Ok(pulse_def_py.into())
}

pub fn pulse_parameters_to_py_dict(
    py: Python,
    parameters: &HashMap<PulseParameterUid, PulseParameterValue>,
    id_store: &NamedIdStore,
    py_objects: &PyObjectInterner<ExternalParameterUid>,
) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);
    for (key, value) in parameters.iter() {
        let key_str = id_store.resolve(*key).unwrap();
        match value {
            PulseParameterValue::ExternalParameter(uid) => {
                dict.set_item(key_str, py_objects.resolve(uid))?;
            }
            PulseParameterValue::ValueOrParameter(value_or_param) => {
                dict.set_item(key_str, value_or_parameter_to_py(py, value_or_param)?)?;
            }
        }
    }
    Ok(dict.into())
}

/// Convert a [`ValueOrParameter`] to a Python object.
fn value_or_parameter_to_py(
    py: Python,
    value: &ValueOrParameter<NumericLiteral>,
) -> PyResult<Py<PyAny>> {
    match value {
        ValueOrParameter::Value(value) => numeric_literal_to_py(py, value),
        ValueOrParameter::Parameter(value) => {
            let id = value.0.to_string();
            Ok(id.into_pyobject(py)?.unbind().into())
        }
        ValueOrParameter::ResolvedParameter { value, .. } => numeric_literal_to_py(py, value),
    }
}

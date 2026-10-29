// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

//! This crate provides common utilities for exporting LabOneQ data structures to Python.

use std::collections::HashMap;

use pyo3::{IntoPyObjectExt, prelude::*, types::PyDict};

use laboneq_common::named_id::NamedIdStore;
use laboneq_common::types::Literal;
use laboneq_dsl::operation::ExternalOrValue;
use laboneq_dsl::types::{
    AcquisitionType, AveragingMode, ComplexOrFloat, ExternalParameterUid, NumericLiteral, PulseDef,
    PulseKind, PulseParameterUid, ValueOrParameter,
};

use crate::py_object_store::PyObjectStore;

/// Convert a [`NumericLiteral`] to a Python object.
pub fn numeric_literal_to_py<'py>(
    py: Python<'py>,
    value: &NumericLiteral,
) -> PyResult<Bound<'py, PyAny>> {
    match value {
        NumericLiteral::Int(v) => v.into_bound_py_any(py),
        NumericLiteral::Float(v) => v.into_bound_py_any(py),
        NumericLiteral::Complex(v) => v.into_bound_py_any(py),
    }
}

/// Convert a [`ComplexOrFloat`] to a Python object.
pub fn complex_or_float_to_py<'py>(
    py: Python<'py>,
    value: &ComplexOrFloat,
) -> PyResult<Bound<'py, PyAny>> {
    match value {
        ComplexOrFloat::Float(v) => v.into_bound_py_any(py),
        ComplexOrFloat::Complex(v) => v.into_bound_py_any(py),
    }
}

/// Convert a [`PulseDef`] to a Python object.
pub fn pulse_def_to_py<'py>(
    py: Python<'py>,
    id_store: &NamedIdStore,
    pulse_def: &PulseDef,
) -> PyResult<Bound<'py, PyAny>> {
    let pulse_def_py_cls = py
        .import("laboneq.core.utilities.pulse_sampler")?
        .getattr("PulseDef")?;

    let kwargs = PyDict::new(py);
    kwargs.set_item("uid", id_store.resolve(pulse_def.uid).unwrap())?;
    kwargs.set_item(
        "amplitude",
        numeric_literal_to_py(py, &pulse_def.amplitude)?,
    )?;
    kwargs.set_item("can_compress", pulse_def.can_compress)?;

    match &pulse_def.kind {
        PulseKind::Functional(func) => {
            kwargs.set_item("function", func.function.name())?;
            kwargs.set_item("length", func.length.value())?;
            kwargs.set_item("samples", py.None())?;
        }
        PulseKind::Sampled(sampled) => {
            kwargs.set_item("samples", sampled.samples.to_py(py)?)?;
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
    pulse_def_py_cls.call((), Some(&kwargs))
}

pub fn pulse_parameters_to_py_dict(
    py: Python,
    parameters: &HashMap<PulseParameterUid, ExternalOrValue>,
    id_store: &NamedIdStore,
    external_parameters: &PyObjectStore<ExternalParameterUid>,
) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);
    for (key, value) in parameters.iter() {
        let key_str = id_store.resolve(*key).unwrap();
        let value = external_or_value_to_py(py, value, id_store, external_parameters)?;
        dict.set_item(key_str, value)?;
    }
    Ok(dict.into())
}

fn external_or_value_to_py<'py>(
    py: Python<'py>,
    value: &ExternalOrValue,
    id_store: &NamedIdStore,
    external_parameters: &PyObjectStore<ExternalParameterUid>,
) -> PyResult<Bound<'py, PyAny>> {
    match value {
        ExternalOrValue::ExternalParameter(uid) => Ok(external_parameters
            .resolve_to_python(uid)
            .expect("Internal error: Failed to resolve external parameter uid to python object")
            .bind(py))
        .cloned(),
        ExternalOrValue::ValueOrParameter(value_or_param) => {
            value_or_parameter_to_py(py, value_or_param, id_store)
        }
        ExternalOrValue::Bool(value) => value.into_bound_py_any(py),
        ExternalOrValue::None => Ok(py.None().into_bound(py)),
    }
}

/// Convert a [`ValueOrParameter`] to a Python object.
fn value_or_parameter_to_py<'py>(
    py: Python<'py>,
    value: &ValueOrParameter<NumericLiteral>,
    id_store: &NamedIdStore,
) -> PyResult<Bound<'py, PyAny>> {
    match value {
        ValueOrParameter::Value(value) => numeric_literal_to_py(py, value),
        ValueOrParameter::Parameter(value) => {
            let id = id_store.resolve(*value).unwrap();
            Ok(id.into_bound_py_any(py)?)
        }
        ValueOrParameter::ResolvedParameter { value, .. } => numeric_literal_to_py(py, value),
    }
}

pub fn value_to_py<'py>(py: Python<'py>, value: &Literal) -> PyResult<Bound<'py, PyAny>> {
    match value {
        Literal::Integer(v) => Ok(v.into_bound_py_any(py)?),
        Literal::Real(v) => Ok(v.into_bound_py_any(py)?),
        Literal::Complex(v) => Ok(v.into_bound_py_any(py)?),
        Literal::Text(v) => Ok(v.into_bound_py_any(py)?),
    }
}

pub fn averaging_mode_to_py(value: &AveragingMode) -> &'static str {
    match value {
        AveragingMode::Cyclic => "CYCLIC",
        AveragingMode::Sequential => "SEQUENTIAL",
        AveragingMode::SingleShot => "SINGLE_SHOT",
    }
}

pub fn acquisition_type_to_py(value: &AcquisitionType) -> &'static str {
    match value {
        AcquisitionType::Raw => "RAW",
        AcquisitionType::Integration => "INTEGRATION",
        AcquisitionType::Discrimination => "DISCRIMINATION",
        AcquisitionType::Spectroscopy => "SPECTROSCOPY",
        AcquisitionType::SpectroscopyIq => "SPECTROSCOPY_IQ",
        AcquisitionType::SpectroscopyPsd => "SPECTROSCOPY_PSD",
    }
}

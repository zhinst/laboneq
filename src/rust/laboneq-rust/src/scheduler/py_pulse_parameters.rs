// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use laboneq_common::named_id::NamedIdStore;
use pyo3::{
    prelude::*,
    types::{PyComplex, PyDict},
};

use crate::{error::Result, scheduler::py_object_interner::PyObjectInterner};
use laboneq_scheduler::experiment::types::{
    ExternalParameterUid, NumericLiteral, PulseParameterUid, PulseParameterValue, ValueOrParameter,
};

pub(super) fn pulse_parameters_to_py_dict(
    py: Python,
    parameters: &HashMap<PulseParameterUid, PulseParameterValue>,
    id_store: &NamedIdStore,
    py_objects: &PyObjectInterner<ExternalParameterUid>,
) -> Result<Py<PyDict>> {
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

fn numeric_literal_to_py(py: Python, value: &NumericLiteral) -> PyResult<Py<PyAny>> {
    match value {
        NumericLiteral::Int(v) => Ok(v.into_pyobject(py)?.unbind().into()),
        NumericLiteral::Float(v) => Ok(v.into_pyobject(py)?.unbind().into()),
        NumericLiteral::Complex(v) => Ok(PyComplex::from_doubles(py, v.re, v.im)
            .into_pyobject(py)?
            .unbind()
            .into()),
    }
}

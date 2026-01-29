// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use laboneq_common::named_id::NamedIdStore;
use laboneq_dsl::operation::PulseParameterValue;
use laboneq_dsl::types::{
    ExternalParameterUid, NumericLiteral, PulseParameterUid, ValueOrParameter,
};
use pyo3::{prelude::*, types::PyDict};

use crate::error::Result;
use crate::scheduler::py_export::numeric_literal_to_py;
use crate::scheduler::py_object_interner::PyObjectInterner;

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

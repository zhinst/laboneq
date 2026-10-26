// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

//! Reads compiler settings from Python.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use laboneq_common::compiler_settings::CompilerSettings;
use laboneq_common::settings::{SettingSpec, SettingValue};

use crate::error::LabOneQException;

/// Read the compiler settings a caller passed, keeping each value's type.
///
/// `bool` is checked before `int` because a Python `bool` is an `int`.
pub(crate) fn setting_values_from_py_dict(
    ob: &Bound<PyDict>,
) -> PyResult<Vec<(String, SettingValue)>> {
    ob.iter()
        .map(|(key, value)| {
            let key: String = key.extract()?;
            let value = if let Ok(value) = value.extract::<bool>() {
                SettingValue::Bool(value)
            } else if let Ok(value) = value.extract::<i64>() {
                SettingValue::Int(value)
            } else if let Ok(value) = value.extract::<f64>() {
                SettingValue::Float(value)
            } else if let Ok(value) = value.extract::<String>() {
                SettingValue::Str(value)
            } else {
                return Err(PyValueError::new_err(format!(
                    "Compiler setting {key} must be a bool, int, float or str, got {value}"
                )));
            };
            Ok((key, value))
        })
        .collect()
}

/// Validate the settings a caller passed against the frontend's and the backend's.
pub(crate) fn compiler_settings_from_py_dict(
    backend_settings: &[SettingSpec],
    ob: Option<&Bound<PyDict>>,
) -> PyResult<CompilerSettings> {
    let pairs = ob.map(setting_values_from_py_dict).transpose()?;
    CompilerSettings::parse(backend_settings, pairs.unwrap_or_default())
        .map_err(|err| LabOneQException::new_err(err.to_string()))
}

// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

//! Python/PyO3 helpers used by `capnp_serializer`.

use pyo3::exceptions::PyValueError;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::types::{IntoPyDict, PyModule};

use laboneq_common::compiler_settings::CompilerSettings;
use laboneq_common::settings::{SettingSpec, SettingValue};

use crate::error::LabOneQException;
/// Check whether `obj` is an exact instance of the Python type `ty`.
///
/// Unlike `isinstance`, this does **not** match subclasses — it compares
/// the fully-qualified class identity (`__module__` + `__qualname__`) so that
/// two unrelated classes that happen to share a `__name__` are not confused.
///
/// Fast path: tries pointer identity first (avoids string allocations in the
/// common case). Falls back to string comparison when the type objects differ
/// (e.g. across module reloads).
pub(crate) fn is_exact_type(obj: &Bound<'_, PyAny>, ty: &Bound<'_, PyAny>) -> PyResult<bool> {
    let py = obj.py();
    let obj_type = obj.get_type();
    if obj_type.is(ty) {
        return Ok(true);
    }
    let obj_module: String = obj_type.getattr(intern!(py, "__module__"))?.extract()?;
    let obj_qualname: String = obj_type.getattr(intern!(py, "__qualname__"))?.extract()?;
    let ty_module: String = ty.getattr(intern!(py, "__module__"))?.extract()?;
    let ty_qualname: String = ty.getattr(intern!(py, "__qualname__"))?.extract()?;
    Ok(obj_module == ty_module && obj_qualname == ty_qualname)
}

/// Convert an (N, 2) numpy float64 I/Q array to a 1-D complex128 array.
pub(crate) fn iq_to_complex<'py>(
    np: &Bound<'py, PyModule>,
    arr: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let shape: Vec<usize> = arr.getattr(intern!(arr.py(), "shape"))?.extract()?;
    if shape.len() != 2 || shape[1] != 2 {
        return Err(PyValueError::new_err(format!(
            "Sampled pulse must have shape (N,) or (N, 2), got {:?}",
            shape
        )));
    }
    let py = arr.py();
    let arr64 = np.call_method(
        intern!(py, "ascontiguousarray"),
        (arr,),
        Some(&[("dtype", "float64")].into_py_dict(py)?),
    )?;
    arr64
        .call_method1(intern!(py, "view"), (intern!(py, "complex128"),))?
        .call_method1(intern!(py, "reshape"), (-1i64,))
}

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

// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

//! Python/PyO3 helpers used by `capnp_serializer`.

use pyo3::exceptions::PyValueError;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::types::{IntoPyDict, PyModule};

use laboneq_common::compiler_settings::CompilerSettings;

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

/// Iterate over experiment signals, handling both DSL (dict) and DATA (list) shapes.
///
/// DSL experiments store signals as a dict (`{uid: signal}`), so we call `.values()`.
/// DATA experiments store them as a plain iterable.  This helper normalises both
/// cases into a single `Bound<PyAny>` that can be iterated.
pub(crate) fn signal_iterable<'py>(signals: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let py = signals.py();
    if signals.hasattr(intern!(py, "values"))? {
        signals.call_method0(intern!(py, "values"))
    } else {
        Ok(signals.clone())
    }
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

/// Creates code generator settings from a Python dictionary
pub(crate) fn compiler_settings_from_py_dict(ob: &Bound<PyDict>) -> PyResult<CompilerSettings> {
    // Convert PyDict to key-value pairs
    let pairs: Result<Vec<(String, String)>, PyErr> = ob
        .iter()
        .map(|(key, value)| {
            let key_str: String = key.extract()?;
            let value_str: String = value.str()?.extract()?;
            Ok((key_str, value_str))
        })
        .collect();

    CompilerSettings::from_key_value_pairs(pairs?)
        .map_err(|err| PyValueError::new_err(err.to_string()))
}

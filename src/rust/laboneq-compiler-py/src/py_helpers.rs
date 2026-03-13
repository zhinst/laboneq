// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

//! Shared Python/PyO3 helpers used by both `py_conversion` and `capnp_serializer`.

use pyo3::intern;
use pyo3::prelude::*;

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

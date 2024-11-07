// Copyright 2024 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use pyo3::prelude::*;
use thiserror::Error;

pub fn py_deep_copy(obj: &Py<PyAny>) -> PyResult<PyObject> {
    Python::with_gil(|py| {
        let copy_module = PyModule::import_bound(py, "copy")?;
        let copy_fn = copy_module.getattr("deep_copy")?;
        copy_fn.call1((obj,)).map(|obj| obj.to_object(py))
    })
}

#[derive(Error, Debug)]
pub enum RuntimeError {
    #[error("Failed to lock shared python reference.")]
    Lock(),
    #[error("Failed to deep copy object on python heap.")]
    PyhonDeepCop(),
    #[error("Failed to extract type from python object.")]
    PyhonExtraction(),
}

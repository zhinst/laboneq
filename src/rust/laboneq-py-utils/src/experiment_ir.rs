// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use pyo3::prelude::*;

use laboneq_dsl::types::ExternalParameterUid;
use laboneq_ir::ExperimentIr;

use crate::py_object_interner::PyObjectInterner;

/// A Python representation of the Experiment IR.
///
/// This class wraps the [`ExperimentIr`] struct and acts as a bridge between
/// the Rust components called from Python.
#[pyclass(name = "ExperimentIr", frozen)]
pub struct ExperimentIrPy {
    pub inner: ExperimentIr,
    pub py_object_store: Arc<PyObjectInterner<ExternalParameterUid>>,
}

#[pymethods]
impl ExperimentIrPy {
    fn device_type_by_uid(&self, device_uid: &str) -> String {
        let uid = self.inner.id_store.get(device_uid).unwrap().into();
        let device = self.inner.device_setup.device_by_uid(&uid).unwrap();
        device.kind().to_string()
    }
}

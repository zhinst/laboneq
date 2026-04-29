// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use laboneq_common::compiler_settings::CompilerSettings;
use pyo3::prelude::*;

use laboneq_dsl::types::ExternalParameterUid;
use laboneq_ir::ExperimentIr;

use laboneq_py_utils::py_object_interner::PyObjectInterner;

use crate::compiler_backend::PreprocessedBackendData;

/// A Python representation of the Experiment IR.
///
/// This class wraps the [`ExperimentIr`] struct and acts as a bridge between
/// the Rust components called from Python.
#[pyclass(name = "ExperimentIr", frozen)]
pub struct ExperimentIrPy {
    pub inner: ExperimentIr,
    pub py_object_store: Arc<PyObjectInterner<ExternalParameterUid>>,
    pub compiler_settings: CompilerSettings,
    pub backend_data: Arc<dyn PreprocessedBackendData + Send + Sync>,
}

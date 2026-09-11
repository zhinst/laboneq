// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use pyo3::prelude::*;
use std::sync::Arc;

use laboneq_common::compiler_settings::CompilerSettings;
use laboneq_ir::signal::Signal;

use crate::compiler_backend::{DynCompilerBackend, PreprocessedBackendData};
use crate::experiment::Experiment;

#[pyclass(name = "Experiment", frozen)]
pub(crate) struct ExperimentPy {
    pub(crate) inner: Experiment,
    // NOTE: The usage of Arc here is to allow sharing the id_store across Python bindings
    // Remove when Python bindings are no longer needed
    pub(crate) signals: Arc<[Signal]>,
    pub(crate) compiler_settings: CompilerSettings,
    pub(crate) backend: Arc<dyn DynCompilerBackend>,
    pub(crate) backend_data: Arc<dyn PreprocessedBackendData + Send + Sync>,
}

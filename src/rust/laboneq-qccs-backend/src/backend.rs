// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use laboneq_compiler_py::compiler_backend::CompilerBackendResult;
use laboneq_compiler_py::compiler_backend::{CompilerBackend, ExperimentView};

use crate::preprocessor::QccsBackendPreprocessedData;
use crate::preprocessor::preprocess_experiment;

#[derive(Default)]
pub struct QccsBackend {}

impl CompilerBackend for QccsBackend {
    type Output = QccsBackendPreprocessedData;

    fn preprocess_experiment(
        &self,
        experiment: ExperimentView,
    ) -> CompilerBackendResult<Self::Output> {
        preprocess_experiment(experiment)
    }
}

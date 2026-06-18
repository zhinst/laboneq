// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use laboneq_error::LabOneQError;
use pyo3::prelude::*;

use codegenerator_py::{HardwareSetup, SignalChannelProperties, generate_code_py};
use laboneq_common::compiler_settings::CompilerSettings;
use laboneq_compiler_py::compiler_backend::{
    CodeGenArtifact, CompilerBackend, Error as CompilerError, ExperimentView, FeedbackCalculator,
    QccsFeedbackCalculator, SignalView,
};
use laboneq_compiler_py::compiler_backend::{CompilerBackendResult, PreprocessOutput};
use laboneq_dsl::types::ExternalParameterUid;
use laboneq_ir::ExperimentIr;
use laboneq_py_utils::py_object_interner::PyObjectInterner;

use crate::preprocessor::QccsBackendPreprocessedData;
use crate::preprocessor::preprocess_experiment;

#[derive(Default)]
pub struct QccsBackend {}

impl CompilerBackend for QccsBackend {
    type Output = QccsBackendPreprocessedData;
    type CodeGenArtifact = CodeGenArtifactQccs;

    fn preprocess_experiment(
        &self,
        experiment: ExperimentView,
    ) -> CompilerBackendResult<PreprocessOutput<Self::Output>> {
        preprocess_experiment(experiment)
    }

    fn generate_code(
        &self,
        experiment: ExperimentIr,
        compiler_settings: &CompilerSettings,
        py_object_store: &PyObjectInterner<ExternalParameterUid>,
        backend_data: &Self::Output,
    ) -> CompilerBackendResult<Self::CodeGenArtifact> {
        let additional_signals = backend_data.signals().map(|s| SignalChannelProperties {
            signal_uid: s.uid,
            awg_key: s.awg_key,
            awg_index: s.awg_index,
            channels: s.channels.iter().map(|c| *c as u8).collect(),
            routed_output_channel_map: backend_data.routed_output_channel_map().clone(),
            ppc_channel: s.ppc_channel,
        });
        let setup_desc = HardwareSetup {
            signals: additional_signals.collect(),
            auxiliary_devices: backend_data.auxiliary_devices().to_vec(),
        };

        let out = Python::attach(|py| -> Result<Py<PyAny>, LabOneQError> {
            Ok(generate_code_py(
                py,
                experiment,
                &setup_desc,
                compiler_settings,
                py_object_store,
            )?
            .unbind())
        })?;
        Ok(CodeGenArtifactQccs { code: out })
    }

    fn device_class(&self) -> usize {
        0
    }

    fn feedback_calculator(
        &self,
        signals: &[SignalView],
        _compiler_settings: &CompilerSettings,
    ) -> Result<
        Option<Box<dyn FeedbackCalculator<Error = CompilerError> + Send + Sync + 'static>>,
        CompilerError,
    > {
        let model = QccsFeedbackCalculator::new(signals.iter().cloned())?;
        Ok(Some(Box::new(model)))
    }
}

pub struct CodeGenArtifactQccs {
    code: Py<PyAny>,
}

impl CodeGenArtifact for CodeGenArtifactQccs {
    fn to_python(&self, py: Python) -> PyResult<Py<PyAny>> {
        Ok(self.code.clone_ref(py))
    }
}

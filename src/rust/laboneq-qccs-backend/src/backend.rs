// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::num::NonZeroU32;

use pyo3::types::PyDict;
use pyo3::{intern, prelude::*};

use codegenerator_py::{
    HardwareSetup, SetupDevice, SignalChannelProperties, artifacts_to_py, generate_code_py,
};
use laboneq_common::compiler_settings::CompilerSettings;
use laboneq_common::named_id::NamedIdStore;
use laboneq_common::settings::SettingSpec;
use laboneq_compiler_py::capnp_artifacts::ArtifactSerializerContext;
use laboneq_compiler_py::compiler_backend::{
    CombinedOutput, CompilerArtifact, CompilerBackend, Error as CompilerError, ExperimentView,
    FeedbackCalculator, PrngSequence, ProcessedExperiment, SignalDelays,
};
use laboneq_compiler_py::compiler_backend::{CompilerBackendResult, PreprocessOutput};
use laboneq_dsl::types::{ExternalParameterUid, HandleUid, SignalUid};
use laboneq_error::{LabOneQError, laboneq_error};
use laboneq_ir::ExperimentIr;
use laboneq_py_utils::py_object_store::PyObjectStore;

use crate::BACKEND_NAME;
use crate::linker::QccsLinker;
use crate::preprocessor::QccsBackendPreprocessedData;
use crate::preprocessor::preprocess_experiment;
use crate::qccs_feedback_calculator::{FeedbackSignal, QccsFeedbackCalculator};

#[derive(Default)]
pub struct QccsBackend {}

impl CompilerBackend for QccsBackend {
    type Output = QccsBackendPreprocessedData;
    type CodeGenArtifact = CodeGenArtifactQccs;
    type CompilerArtifact = CompilationArtifactsQccs;
    type CombinedOutput = QccsCombinedOutput;
    type Linker = QccsLinker;
    type EmulatedHqcsProgram = ();
    fn settings(&self) -> &'static [SettingSpec] {
        codegenerator_py::CODE_GENERATOR_SETTINGS
    }

    fn name(&self) -> &'static str {
        BACKEND_NAME
    }

    fn preprocess_experiment(
        &self,
        experiment: ExperimentView,
    ) -> CompilerBackendResult<PreprocessOutput<Self::Output>> {
        preprocess_experiment(experiment)
    }

    fn validate(
        &self,
        experiment: &ProcessedExperiment<'_>,
        backend_data: &Self::Output,
    ) -> CompilerBackendResult<()> {
        crate::validation::validate(experiment, backend_data)
    }

    fn prng_sequence(&self, seed: u32, range: u32) -> CompilerBackendResult<Box<dyn PrngSequence>> {
        let generator = crate::prng::PrngGeneratorQccs::new(seed, 0, range - 1)
            .map_err(|e| laboneq_error!("{e}"))?;
        Ok(Box::new(generator))
    }

    fn generate_code(
        &self,
        experiment: &ExperimentIr,
        compiler_settings: &CompilerSettings,
        external_parameters: &PyObjectStore<ExternalParameterUid>,
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
        let devices = backend_data
            .device_kinds()
            .iter()
            .map(|(uid, kind)| SetupDevice {
                uid: *uid,
                kind: kind.for_codegen(),
                is_shfqc: backend_data.is_shfqc(*uid),
                lead_delay: backend_data
                    .lead_delays()
                    .get(uid)
                    .copied()
                    .unwrap_or_default(),
                options: backend_data
                    .device_options(*uid)
                    .expect("Expected options for every device in the setup")
                    .clone(),
            })
            .collect();
        let setup_desc = HardwareSetup {
            signals: additional_signals.collect(),
            devices,
            auxiliary_devices: backend_data.auxiliary_devices().to_vec(),
        };

        let id_store = experiment.id_store;

        let out = Python::attach(|py| -> Result<Py<PyAny>, LabOneQError> {
            let artifacts = generate_code_py(
                py,
                experiment,
                &setup_desc,
                compiler_settings,
                external_parameters,
            )?;
            let artifacts_py = artifacts_to_py(py, artifacts, id_store, external_parameters)?;
            Ok(artifacts_py.into())
        })?;
        Ok(CodeGenArtifactQccs { inner: out })
    }

    fn feedback_calculator(
        &self,
        signal_delays: &[SignalDelays],
        _compiler_settings: &CompilerSettings,
        backend_data: &Self::Output,
    ) -> Result<Option<Box<dyn FeedbackCalculator<Error = CompilerError>>>, CompilerError> {
        let feedback_signals = signal_delays.iter().map(|delays| {
            let s = backend_data
                .get_signal(delays.uid)
                .expect("Internal error: Signal unknown to the backend");
            FeedbackSignal {
                uid: s.uid,
                awg_key: s.awg_key,
                signal_kind: s.kind.clone(),
                device_kind: s.device_kind,
                device_traits: s.device_kind.traits(),
                is_shfqc: backend_data.is_shfqc(s.device_uid),
                sampling_rate: s.sampling_rate.value(),
                signal_delay: delays.signal_delay,
                port_delay: s.calibration.port_delay,
                start_delay: delays.start_delay,
            }
        });
        let model = QccsFeedbackCalculator::new(feedback_signals)?;
        Ok(Some(Box::new(model)))
    }

    fn linker(&self) -> QccsLinker {
        QccsLinker::default()
    }

    fn finish_attempt(
        &self,
        combined: &mut Self::CombinedOutput,
        compiler_settings: &CompilerSettings,
        chunk_count: Option<NonZeroU32>,
    ) -> CompilerBackendResult<()> {
        Python::attach(|py| {
            let kwargs = PyDict::new(py);
            kwargs.set_item(intern!(py, "log_report"), compiler_settings.log_report)?;
            kwargs.set_item(
                intern!(py, "ignore_resource_limitation_errors"),
                compiler_settings.ignore_resource_exhaustion,
            )?;
            kwargs.set_item(intern!(py, "uses_chunking"), chunk_count.is_some())?;
            output_module(py)?
                .call_method(
                    intern!(py, "finish_attempt"),
                    (combined.0.bind(py),),
                    Some(&kwargs),
                )
                .map_err(|e| LabOneQError::from_pyerr(py, e))?;
            Ok(())
        })
    }

    fn produce_output(
        &self,
        combined: Self::CombinedOutput,
        _hqcs_program: (),
    ) -> CompilerBackendResult<
        laboneq_compiler_py::compiler_backend::CompilationOutput<Self::CompilerArtifact>,
    > {
        Python::attach(|py| {
            let artifacts = output_module(py)?
                .call_method1(intern!(py, "produce_output"), (combined.0.bind(py),))?;
            Ok(laboneq_compiler_py::compiler_backend::CompilationOutput {
                artifacts: CompilationArtifactsQccs {
                    artifacts: artifacts.into(),
                },
            })
        })
    }
}

fn output_module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    py.import(intern!(py, "laboneq.qccs.compiler.output"))
}

/// QCCS's linked real-time compilation output, the Python `CombinedRTOutputSeqC`.
pub struct QccsCombinedOutput(pub(crate) Py<PyAny>);

impl CombinedOutput for QccsCombinedOutput {
    fn raw_acquisition_lengths(
        &self,
        pairs: &[(SignalUid, HandleUid)],
        id_store: &NamedIdStore,
    ) -> CompilerBackendResult<Vec<(SignalUid, HandleUid, usize)>> {
        Python::attach(|py| {
            let obj = self.0.bind(py);
            pairs
                .iter()
                .map(|&(signal, handle)| -> CompilerBackendResult<_> {
                    let signal_str = id_store
                        .resolve(signal)
                        .expect("Internal error: signal not found in ID store");
                    let length: usize = obj
                        .call_method1(intern!(py, "get_raw_acquire_length"), (signal_str,))?
                        .extract()?;
                    Ok((signal, handle, length))
                })
                .collect()
        })
    }

    fn total_execution_time(&self) -> CompilerBackendResult<f64> {
        Python::attach(|py| {
            Ok(self
                .0
                .bind(py)
                .getattr(intern!(py, "total_execution_time"))?
                .extract()?)
        })
    }

    fn max_execution_time_per_step(&self) -> CompilerBackendResult<f64> {
        Python::attach(|py| {
            Ok(self
                .0
                .bind(py)
                .getattr(intern!(py, "max_execution_time_per_step"))?
                .extract()?)
        })
    }
}

/// QCCS's code generation output, the Python `SeqCGenOutput`.
pub struct CodeGenArtifactQccs {
    pub(crate) inner: Py<PyAny>,
}

pub struct CompilationArtifactsQccs {
    artifacts: Py<PyAny>,
}

impl CompilerArtifact for CompilationArtifactsQccs {
    fn to_capnp_writer<'a>(
        &'a self,
        _ctx: &'a ArtifactSerializerContext<'a>,
    ) -> impl laboneq_capnp::artifacts::ArtifactsWriter + 'a {
        crate::artifacts_capnp::QccsArtifactsCapnpWriter {
            artifacts: &self.artifacts,
        }
    }
}

// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

//! Result module for the code generator.
//!
//! This module defines the [`AwgCodeGenerationResultPy`] class, which is used to
//! represent the result of the code generation process for an AWG.
use std::collections::{HashMap, HashSet};

use crate::common_types::AwgKeyPy;
use crate::waveform_sampler::IntegrationWeight;
use crate::{awg_event::AwgEvent, waveform_sampler::SampledWaveformSignaturePy};
use codegenerator::handle_feedback_registers::{
    Acquisition, FeedbackConfig, FeedbackRegisterAllocation,
};
use codegenerator::{AwgCompilationInfo, SampledWaveform, WaveDeclaration, ir};
use pyo3::prelude::*;
use pyo3::types::PyDict;

#[pyclass(name = "WaveDeclaration")]
#[derive(Debug)]
pub struct WaveDeclarationPy {
    pub obj: WaveDeclaration,
}

#[pymethods]
impl WaveDeclarationPy {
    #[getter]
    pub fn length(&self) -> i64 {
        self.obj.length
    }

    #[getter]
    pub fn signature_string(&self) -> &str {
        &self.obj.signature_string
    }

    #[getter]
    pub fn has_marker1(&self) -> bool {
        self.obj.has_marker1
    }

    #[getter]
    pub fn has_marker2(&self) -> bool {
        self.obj.has_marker2
    }
}

#[pyclass(name = "SampledWaveform")]
#[derive(Debug)]
pub struct SampledWaveformPy {
    pub obj: SampledWaveform<SampledWaveformSignaturePy>,
}

#[pymethods]
impl SampledWaveformPy {
    #[getter]
    pub fn signals(&self) -> &HashSet<String> {
        &self.obj.signals
    }

    #[getter]
    pub fn signature_string(&self) -> &str {
        &self.obj.signature_string
    }

    #[getter]
    pub fn signature(&self, py: Python) -> PyObject {
        self.obj.signature.signature.clone_ref(py)
    }
}

#[pyclass(name = "IntegrationWeight")]
#[derive(Debug)]
pub struct IntegrationWeightPy {
    #[pyo3(get)]
    pub signals: HashSet<String>,
    #[pyo3(get)]
    pub samples_i: PyObject,
    #[pyo3(get)]
    pub samples_q: PyObject,
    #[pyo3(get)]
    pub downsampling_factor: Option<usize>,
    #[pyo3(get)]
    pub basename: String,
}

#[pyclass(name = "SignalIntegrationInfo")]
#[derive(Debug)]
pub struct SignalIntegrationInfo {
    #[pyo3(get)]
    pub is_play: bool,
    #[pyo3(get)]
    pub length: ir::Samples,
}

/// Result structure for single AWG code generation.
#[pyclass(name = "AwgCodeGenerationResult", frozen, unsendable)]
pub struct AwgCodeGenerationResultPy {
    awg_events: Vec<Py<AwgEvent>>,
    sampled_waveforms: Vec<Py<SampledWaveformPy>>,
    wave_declarations: Vec<Py<WaveDeclarationPy>>,
    integration_weights: Vec<Py<IntegrationWeightPy>>,
    awg_info: AwgCompilationInfo,
    #[pyo3(get)]
    global_delay: ir::Samples,
    #[pyo3(get)]
    signal_delays: HashMap<String, f64>,
    #[pyo3(get)]
    integration_lengths: HashMap<String, Py<SignalIntegrationInfo>>,
    feedback_register: Option<FeedbackRegisterAllocation>,
    source_feedback_register: Option<FeedbackRegisterAllocation>,
}

impl AwgCodeGenerationResultPy {
    #[allow(clippy::too_many_arguments)]
    pub fn create(
        awg_events: Vec<AwgEvent>,
        sampled_waveforms: Vec<SampledWaveform<SampledWaveformSignaturePy>>,
        wave_declarations: Vec<WaveDeclaration>,
        integration_weights: Vec<IntegrationWeight>,
        awg_info: AwgCompilationInfo,
        global_delay: ir::Samples,
        signal_delays: &HashMap<&str, f64>,
        integration_lengths: HashMap<String, SignalIntegrationInfo>,
        feedback_register: Option<FeedbackRegisterAllocation>,
        source_feedback_register: Option<FeedbackRegisterAllocation>,
    ) -> PyResult<Self> {
        Python::with_gil(|py| {
            let awg_events: Vec<Py<AwgEvent>> = awg_events
                .into_iter()
                .map(|event| Py::new(py, event).expect("Failed to create AwgEvent"))
                .collect();
            let sampled_waveforms: Vec<Py<SampledWaveformPy>> = Python::with_gil(|py| {
                sampled_waveforms
                    .into_iter()
                    .map(|sampled| Py::new(py, SampledWaveformPy { obj: sampled }).unwrap())
                    .collect()
            });
            let wave_declarations: Vec<Py<WaveDeclarationPy>> = Python::with_gil(|py| {
                wave_declarations
                    .into_iter()
                    .map(|decl| Py::new(py, WaveDeclarationPy { obj: decl }).unwrap())
                    .collect()
            });
            let integration_weights: Vec<Py<IntegrationWeightPy>> = Python::with_gil(|py| {
                integration_weights
                    .into_iter()
                    .map(|weight| {
                        Py::new(
                            py,
                            IntegrationWeightPy {
                                signals: weight.signals.iter().map(|s| s.to_string()).collect(),
                                samples_i: weight.samples_i,
                                samples_q: weight.samples_q,
                                downsampling_factor: weight.downsampling_factor,
                                basename: weight.basename,
                            },
                        )
                        .unwrap()
                    })
                    .collect()
            });
            let integration_lengths = integration_lengths
                .into_iter()
                .map(|(k, v)| {
                    (
                        k,
                        Py::new(
                            py,
                            SignalIntegrationInfo {
                                is_play: v.is_play,
                                length: v.length,
                            },
                        )
                        .unwrap(),
                    )
                })
                .collect();
            let output = AwgCodeGenerationResultPy {
                awg_events,
                sampled_waveforms,
                wave_declarations,
                integration_weights,
                awg_info,
                global_delay,
                signal_delays: signal_delays
                    .iter()
                    .map(|(k, v)| (k.to_string(), *v))
                    .collect(),
                integration_lengths,
                feedback_register,
                source_feedback_register,
            };
            Ok(output)
        })
    }

    pub fn default() -> Self {
        AwgCodeGenerationResultPy {
            awg_events: vec![],
            sampled_waveforms: vec![],
            wave_declarations: vec![],
            integration_weights: vec![],
            awg_info: AwgCompilationInfo::default(),
            global_delay: 0,
            signal_delays: HashMap::new(),
            integration_lengths: HashMap::new(),
            feedback_register: None,
            source_feedback_register: None,
        }
    }
}

#[pymethods]
impl AwgCodeGenerationResultPy {
    #[getter]
    fn awg_events(&self) -> &Vec<Py<AwgEvent>> {
        &self.awg_events
    }

    #[getter]
    fn sampled_waveforms(&self) -> &Vec<Py<SampledWaveformPy>> {
        &self.sampled_waveforms
    }

    #[getter]
    fn wave_declarations(&self) -> &Vec<Py<WaveDeclarationPy>> {
        &self.wave_declarations
    }

    #[getter]
    fn integration_weights(&self) -> &Vec<Py<IntegrationWeightPy>> {
        &self.integration_weights
    }

    #[getter]
    fn has_readout_feedback(&self) -> bool {
        self.awg_info.has_readout_feedback()
    }

    #[getter]
    fn feedback_register(&self, py: Python) -> PyResult<Option<PyObject>> {
        match &self.feedback_register {
            Some(FeedbackRegisterAllocation::Global { register }) => {
                Ok(Some((*register as usize).into_pyobject(py)?.into()))
            }
            Some(FeedbackRegisterAllocation::Local) => Ok(Some("local".into_pyobject(py)?.into())),
            None => Ok(None),
        }
    }

    #[getter]
    fn source_feedback_register(&self, py: Python) -> PyResult<Option<PyObject>> {
        match &self.source_feedback_register {
            Some(FeedbackRegisterAllocation::Global { register }) => {
                Ok(Some((*register as usize).into_pyobject(py)?.into()))
            }
            Some(FeedbackRegisterAllocation::Local) => Ok(Some("local".into_pyobject(py)?.into())),
            None => Ok(None),
        }
    }

    #[getter]
    fn ppc_device(&self) -> Option<&str> {
        self.awg_info
            .ppc_device()
            .map(|device| device.device.as_str())
    }

    #[getter]
    fn ppc_channel(&self) -> Option<i64> {
        self.awg_info
            .ppc_device()
            .map(|device| device.channel as i64)
    }
}

#[pyclass(name = "SeqCGenOutput")]
pub struct SeqCGenOutputPy {
    awg_results: Vec<Py<AwgCodeGenerationResultPy>>,
    #[pyo3(get)]
    total_execution_time: f64,
    #[pyo3(get)]
    qa_signal_by_handle: HashMap<String, (String, Py<AwgKeyPy>)>,
    simultaneous_acquires: Vec<Vec<Acquisition>>,
}

impl SeqCGenOutputPy {
    pub fn new(
        py: Python,
        awg_results: Vec<AwgCodeGenerationResultPy>,
        total_execution_time: f64,
        mut feedback_register_config: FeedbackConfig,
    ) -> Self {
        let awg_results: Vec<Py<AwgCodeGenerationResultPy>> = awg_results
            .into_iter()
            .map(|result| Py::new(py, result).expect("Failed to create AwgCodeGenerationResultPy"))
            .collect();
        let qa_signal_by_handle: HashMap<String, (String, Py<AwgKeyPy>)> = feedback_register_config
            .handles()
            .map(|handle| {
                let signal_info = feedback_register_config.feedback_source(handle).unwrap();
                (
                    handle.to_string(),
                    (
                        signal_info.signal.uid.to_string(),
                        Py::new(py, AwgKeyPy::new(signal_info.awg_key.clone()))
                            .expect("Failed to create AwgKey"),
                    ),
                )
            })
            .collect();
        SeqCGenOutputPy {
            awg_results,
            total_execution_time,
            qa_signal_by_handle,
            simultaneous_acquires: feedback_register_config.into_acquisitions(),
        }
    }
}

#[pymethods]
impl SeqCGenOutputPy {
    #[getter]
    fn awg_results(&self) -> &Vec<Py<AwgCodeGenerationResultPy>> {
        &self.awg_results
    }

    #[getter]
    fn qa_signal_by_handle(&self) -> &HashMap<String, (String, Py<AwgKeyPy>)> {
        &self.qa_signal_by_handle
    }

    #[getter]
    fn simultaneous_acquires(&self, py: Python) -> PyResult<Vec<PyObject>> {
        let mut sim_acquires = Vec::new();
        for acquisitions in &self.simultaneous_acquires {
            let dict = PyDict::new(py);
            for acquisition in acquisitions {
                dict.set_item(acquisition.signal.clone(), acquisition.handle.to_string())?;
            }
            sim_acquires.push(dict.into_pyobject(py)?.into());
        }
        Ok(sim_acquires)
    }
}

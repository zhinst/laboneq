// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

//! Result module for the code generator.
//!
//! This module defines the [`AwgCodeGenerationResultPy`] class, which is used to
//! represent the result of the code generation process for an AWG.
use crate::waveform_sampler::IntegrationWeight;
use crate::waveform_sampler::SampledWaveformSignaturePy;
use codegenerator::handle_feedback_registers::FeedbackRegisterAllocation;
use codegenerator::handle_feedback_registers::{Acquisition, FeedbackConfig};
use codegenerator::ir::PpcDevice;
use codegenerator::ir::compilation_job::AwgKind;
use codegenerator::{SampledWaveform, WaveDeclaration, ir};
use indexmap::IndexMap;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pythonize::pythonize;
use sampled_event_handler::FeedbackRegisterConfig;
use sampled_event_handler::ParameterPhaseIncrement;
use sampled_event_handler::SHFPPCSweeperConfig;
use seqc_tracker::wave_index_tracker::SignalType;
use seqc_tracker::wave_index_tracker::WaveIndex;
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

#[pyo3::pyclass(name = "FeedbackRegisterConfig")]
#[derive(Debug, Clone)]
pub(crate) struct FeedbackRegisterConfigPy {
    #[pyo3(get)]
    local: bool,

    // Receiver (SG instruments)
    #[pyo3(get, set)]
    source_feedback_register: Option<i64>,
    #[pyo3(get)]
    register_index_select: Option<u8>,
    #[pyo3(get)]
    codeword_bitshift: Option<u8>,
    #[pyo3(get)]
    codeword_bitmask: Option<u16>,
    #[pyo3(get)]
    command_table_offset: Option<u32>,

    // Transmitter (QA instruments)
    #[pyo3(get, set)]
    target_feedback_register: Option<i64>,
}

#[pymethods]
impl FeedbackRegisterConfigPy {
    fn __eq__(&self, other: &FeedbackRegisterConfigPy) -> bool {
        self.local == other.local
            && self.source_feedback_register == other.source_feedback_register
            && self.register_index_select == other.register_index_select
            && self.codeword_bitshift == other.codeword_bitshift
            && self.codeword_bitmask == other.codeword_bitmask
            && self.command_table_offset == other.command_table_offset
            && self.target_feedback_register == other.target_feedback_register
    }
}

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
    pub fn signature(&self, py: Python) -> Py<PyAny> {
        self.obj.signature.signature.clone_ref(py)
    }
}

#[pyclass(name = "IntegrationWeight")]
#[derive(Debug)]
pub struct IntegrationWeightPy {
    #[pyo3(get)]
    pub signals: HashSet<String>,
    #[pyo3(get)]
    pub samples_i: Py<PyAny>,
    #[pyo3(get)]
    pub samples_q: Py<PyAny>,
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
    #[pyo3(get)]
    seqc: String,
    #[pyo3(get)]
    wave_indices: Vec<(String, (u32, String))>,
    #[pyo3(get)]
    command_table: Option<Py<PyAny>>,
    #[pyo3(get)]
    shf_sweeper_config: Option<Py<PyAny>>,
    sampled_waveforms: Vec<Py<SampledWaveformPy>>,
    integration_weights: Vec<Py<IntegrationWeightPy>>,
    #[pyo3(get)]
    signal_delays: HashMap<String, f64>,
    #[pyo3(get)]
    integration_lengths: HashMap<String, Py<SignalIntegrationInfo>>,
    #[pyo3(get)]
    parameter_phase_increment_map: Option<HashMap<String, Vec<i64>>>,
    #[pyo3(get)]
    feedback_register_config: FeedbackRegisterConfigPy,
}

impl AwgCodeGenerationResultPy {
    #[allow(clippy::too_many_arguments)]
    pub fn create(
        seqc: String,
        wave_indices: IndexMap<String, (WaveIndex, SignalType)>,
        command_table: Option<Value>,
        shf_sweeper_config: Option<SHFPPCSweeperConfig>,
        sampled_waveforms: Vec<SampledWaveform<SampledWaveformSignaturePy>>,
        integration_weights: Vec<IntegrationWeight>,
        signal_delays: &HashMap<&str, f64>,
        integration_lengths: HashMap<String, SignalIntegrationInfo>,
        ppc_device: Option<&Arc<PpcDevice>>,
        parameter_phase_increment_map: Option<HashMap<String, Vec<ParameterPhaseIncrement>>>,
        feedback_register_config: FeedbackRegisterConfig,
        target_feedback_register: Option<i64>,
        source_feedback_register: Option<FeedbackRegisterAllocation>,
    ) -> PyResult<Self> {
        Python::attach(|py| {
            let sampled_waveforms: Vec<Py<SampledWaveformPy>> = sampled_waveforms
                .into_iter()
                .map(|sampled| Py::new(py, SampledWaveformPy { obj: sampled }).unwrap())
                .collect();
            let integration_weights: Vec<Py<IntegrationWeightPy>> = integration_weights
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
                .collect();
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
            let shf_sweeper_config = match shf_sweeper_config {
                Some(mut config) => Some(
                    pythonize(
                        py,
                        &config.finalize(Arc::clone(
                            ppc_device.expect("Internal error: PPC device missing"),
                        )),
                    )?
                    .unbind(),
                ),
                None => None,
            };
            let parameter_phase_increment_map = parameter_phase_increment_map.map(|p| {
                p.into_iter()
                    .map(|(k, v)| {
                        (
                            k,
                            v.iter()
                                .map(|p| match p {
                                    ParameterPhaseIncrement::Index(i) => *i as i64,
                                    _ => -1, // Convert other types to -1
                                })
                                .collect::<Vec<i64>>(),
                        )
                    })
                    .collect()
            });
            let command_table = match command_table {
                Some(ct) => Some({
                    let ct = pythonize(py, &ct)?;
                    ct.unbind()
                }),
                None => None,
            };
            let signal_delays = signal_delays
                .iter()
                .map(|(k, v)| (k.to_string(), *v))
                .collect();
            // We take a detour via a Vec to ensure the order of wave indices is preserved
            let wave_indices = wave_indices
                .into_iter()
                .map(|(k, (v, signal_type))| {
                    (
                        k,
                        (
                            v,
                            match signal_type {
                                SignalType::COMPLEX => "complex",
                                SignalType::SIGNAL(s) => match s {
                                    AwgKind::DOUBLE => "double",
                                    AwgKind::SINGLE => "single",
                                    AwgKind::IQ => "iq",
                                },
                            }
                            .to_string(),
                        ),
                    )
                })
                .collect::<Vec<(String, (u32, String))>>();
            let feedback_register_config = FeedbackRegisterConfigPy {
                local: feedback_register_config.local,
                source_feedback_register: source_feedback_register.map(|sfr| match sfr {
                    FeedbackRegisterAllocation::Local => -1,
                    FeedbackRegisterAllocation::Global { register } => register as i64,
                }),
                register_index_select: feedback_register_config.register_index_select,
                codeword_bitshift: feedback_register_config.codeword_bitshift,
                codeword_bitmask: feedback_register_config.codeword_bitmask,
                command_table_offset: feedback_register_config.command_table_offset,
                target_feedback_register,
            };
            let output = AwgCodeGenerationResultPy {
                seqc,
                wave_indices,
                command_table,
                shf_sweeper_config,
                sampled_waveforms,
                integration_weights,
                signal_delays,
                integration_lengths,
                parameter_phase_increment_map,
                feedback_register_config,
            };
            Ok(output)
        })
    }

    pub fn default() -> Self {
        AwgCodeGenerationResultPy {
            seqc: String::new(),
            wave_indices: vec![],
            command_table: None,
            shf_sweeper_config: None,
            sampled_waveforms: vec![],
            integration_weights: vec![],
            signal_delays: HashMap::new(),
            integration_lengths: HashMap::new(),
            parameter_phase_increment_map: None,
            feedback_register_config: FeedbackRegisterConfigPy {
                local: false,
                source_feedback_register: None,
                register_index_select: None,
                codeword_bitshift: None,
                codeword_bitmask: None,
                command_table_offset: None,
                target_feedback_register: None,
            },
        }
    }
}

#[pymethods]
impl AwgCodeGenerationResultPy {
    #[getter]
    fn sampled_waveforms(&self) -> &Vec<Py<SampledWaveformPy>> {
        &self.sampled_waveforms
    }

    #[getter]
    fn integration_weights(&self) -> &Vec<Py<IntegrationWeightPy>> {
        &self.integration_weights
    }
}

#[pyclass(name = "SeqCGenOutput")]
pub struct SeqCGenOutputPy {
    awg_results: Vec<Py<AwgCodeGenerationResultPy>>,
    #[pyo3(get)]
    total_execution_time: f64,
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
        SeqCGenOutputPy {
            awg_results,
            total_execution_time,
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
    fn simultaneous_acquires(&self, py: Python) -> PyResult<Vec<Py<PyAny>>> {
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

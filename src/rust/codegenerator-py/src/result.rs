// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

//! Result module for the code generator.
//!
//! This module defines the [`AwgCodeGenerationResultPy`] class, which is used to
//! represent the result of the code generation process for an AWG.
use laboneq_common::named_id::NamedIdStore;
use pyo3::prelude::*;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use codegenerator::ir::compilation_job::ChannelIndex;
use codegenerator::ir::compilation_job::DeviceUid;
use codegenerator::ir::{Samples, compilation_job::AwgKind};
use codegenerator::result::ParameterPhaseIncrement;
use codegenerator::result::SeqCGenOutput;
use codegenerator::result::ShfPpcSweepJson;
use codegenerator::result::SignalType;
use codegenerator::result::{AwgCodeGenerationResult, MarkerMode};

use crate::waveform_sampler::WaveformSamplerPy;

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

#[pyclass(name = "SampledWaveform")]
#[derive(Debug)]
pub struct SampledWaveformPy {
    #[pyo3(get)]
    signals: HashSet<String>,
    signature_string: Arc<String>,
    signature: Arc<Py<PyAny>>,
}

// SAFETY: Safe to Send/Sync across threads.
// - Py<T> fields use PyO3's delayed reference count mechanism for thread-safe drops
// - Other fields (HashSet<String>, Arc<String>) are inherently Send/Sync
// - No thread-local storage or thread-specific state
unsafe impl Send for SampledWaveformPy {}
unsafe impl Sync for SampledWaveformPy {}

#[pymethods]
impl SampledWaveformPy {
    #[getter]
    pub fn signature_string(&self) -> &str {
        &self.signature_string
    }

    #[getter]
    pub fn signature(&self) -> &Py<PyAny> {
        &self.signature
    }
}

#[pyclass(name = "IntegrationWeight")]
#[derive(Debug)]
pub(crate) struct IntegrationWeightPy {
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

// SAFETY: Safe to Send/Sync across threads.
// - Py<PyAny> fields (samples_i, samples_q) use PyO3's thread-safe drop mechanism
// - Other fields (HashSet, String, Option<usize>) are inherently Send/Sync
// - No thread-local storage or thread-specific state
unsafe impl Send for IntegrationWeightPy {}
unsafe impl Sync for IntegrationWeightPy {}

#[pyclass(name = "SignalIntegrationInfo")]
#[derive(Debug)]
pub(crate) struct SignalIntegrationInfoPy {
    #[pyo3(get)]
    pub is_play: bool,
    #[pyo3(get)]
    pub length: Samples,
}

#[pyclass(name = "PpcSweeperConfig")]
#[derive(Debug)]
pub struct PpcSweeperConfigPy {
    inner: ShfPpcSweepJson,
}

#[pymethods]
impl PpcSweeperConfigPy {
    #[getter]
    pub fn ppc_device(&self) -> &str {
        &self.inner.ppc_device.device
    }

    #[getter]
    pub fn ppc_channel(&self) -> i64 {
        self.inner.ppc_device.channel as i64
    }

    #[getter]
    pub fn json(&self) -> &str {
        &self.inner.json
    }
}

/// Result structure for single AWG code generation.
#[pyclass(name = "AwgCodeGenerationResult", frozen)]
pub struct AwgCodeGenerationResultPy {
    #[pyo3(get)]
    seqc: String,
    #[pyo3(get)]
    wave_indices: Vec<(String, (u32, String))>,
    #[pyo3(get)]
    command_table: Option<String>,
    #[pyo3(get)]
    shf_sweeper_config: Option<Py<PpcSweeperConfigPy>>,
    sampled_waveforms: Vec<Py<SampledWaveformPy>>,
    integration_weights: Vec<Py<IntegrationWeightPy>>,
    #[pyo3(get)]
    signal_delays: HashMap<String, f64>,
    #[pyo3(get)]
    integration_lengths: HashMap<String, Py<SignalIntegrationInfoPy>>,
    #[pyo3(get)]
    parameter_phase_increment_map: Option<HashMap<String, Vec<i64>>>,
    #[pyo3(get)]
    feedback_register_config: FeedbackRegisterConfigPy,
    #[pyo3(get)]
    channel_properties: Vec<ChannelPropertiesPy>,
}

// SAFETY: Safe to Send/Sync across threads.
// - All Py<T> fields rely on PyO3's delayed reference count mechanism for safe
//   cross-thread drops (see https://docs.rs/pyo3/0.28.0/pyo3/struct.Py.html#impl-Drop-for-Py%3CT%3E)
// - SampledWaveformPy and IntegrationWeightPy are themselves Send/Sync
// - All other fields are standard Send/Sync types (String, Vec, HashMap, Option, etc.)
// - No thread-local storage or thread-specific state
// This enables passing compiled results to async executors and background threads.
unsafe impl Send for AwgCodeGenerationResultPy {}
unsafe impl Sync for AwgCodeGenerationResultPy {}

impl AwgCodeGenerationResultPy {
    #[allow(clippy::too_many_arguments)]
    pub fn create(
        py: Python,
        result: AwgCodeGenerationResult<WaveformSamplerPy>,
        id_store: &NamedIdStore,
    ) -> PyResult<Self> {
        let sampled_waveforms: Vec<Py<SampledWaveformPy>> = result
            .sampled_waveforms
            .into_iter()
            .map(|sampled| {
                Py::new(
                    py,
                    SampledWaveformPy {
                        signals: sampled
                            .signals
                            .iter()
                            .map(|s| id_store.resolve_unchecked(*s).to_string())
                            .collect(),
                        signature_string: Arc::clone(&sampled.signature_string),
                        signature: sampled.signature.signature,
                    },
                )
                .unwrap()
            })
            .collect();
        let integration_weights: Vec<Py<IntegrationWeightPy>> = result
            .integration_weights
            .into_iter()
            .map(|weight| {
                Py::new(
                    py,
                    IntegrationWeightPy {
                        signals: weight
                            .signals
                            .iter()
                            .map(|s| id_store.resolve_unchecked(*s).to_string())
                            .collect(),
                        samples_i: weight.samples_i,
                        samples_q: weight.samples_q,
                        downsampling_factor: weight.downsampling_factor,
                        basename: weight.basename,
                    },
                )
                .unwrap()
            })
            .collect();
        let integration_lengths = result
            .integration_lengths
            .into_iter()
            .map(|(k, v)| {
                (
                    id_store.resolve_unchecked(k).to_string(),
                    Py::new(
                        py,
                        SignalIntegrationInfoPy {
                            is_play: v.is_play,
                            length: v.length,
                        },
                    )
                    .unwrap(),
                )
            })
            .collect();
        let shf_sweeper_config = if let Some(config) = result.shf_sweeper_config {
            Some(Py::new(py, PpcSweeperConfigPy { inner: config })?)
        } else {
            None
        };
        let parameter_phase_increment_map = result.parameter_phase_increment_map.map(|p| {
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
        let signal_delays = result
            .signal_delays
            .into_iter()
            .map(|(k, v)| (id_store.resolve_unchecked(k).to_string(), v))
            .collect();
        // We take a detour via a Vec to ensure the order of wave indices is preserved
        let wave_indices = result
            .wave_indices
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
            local: result.feedback_register_config.local,
            source_feedback_register: result.feedback_register_config.source_feedback_register,
            register_index_select: result.feedback_register_config.register_index_select,
            codeword_bitshift: result.feedback_register_config.codeword_bitshift,
            codeword_bitmask: result.feedback_register_config.codeword_bitmask,
            command_table_offset: result.feedback_register_config.command_table_offset,
            target_feedback_register: result.feedback_register_config.target_feedback_register,
        };
        let channel_properties = result
            .channel_properties
            .into_iter()
            .map(|properties| ChannelPropertiesPy {
                channel: properties.channel,
                marker_mode: properties.marker_mode.map(|marker_mode| match marker_mode {
                    MarkerMode::Trigger => "TRIGGER".to_string(),
                    MarkerMode::Marker => "MARKER".to_string(),
                }),
            })
            .collect();
        let output = AwgCodeGenerationResultPy {
            seqc: result.seqc,
            wave_indices,
            command_table: result.command_table,
            shf_sweeper_config,
            sampled_waveforms,
            integration_weights,
            signal_delays,
            integration_lengths,
            parameter_phase_increment_map,
            feedback_register_config,
            channel_properties,
        };
        Ok(output)
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
            channel_properties: vec![],
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
    #[pyo3(get)]
    result_handle_maps: HashMap<ResultSourcePy, Vec<Vec<String>>>,
    #[pyo3(get)]
    measurements: Vec<MeasurementPy>,
    #[pyo3(get)]
    integration_unit_allocations: Vec<IntegrationUnitAllocationPy>,
}

impl SeqCGenOutputPy {
    pub fn new(
        py: Python,
        results: SeqCGenOutput<WaveformSamplerPy>,
        id_store: &NamedIdStore,
    ) -> Self {
        let awg_results: Vec<Py<AwgCodeGenerationResultPy>> = results
            .awg_results
            .into_iter()
            .map(|result| {
                Py::new(
                    py,
                    AwgCodeGenerationResultPy::create(py, result, id_store).unwrap(),
                )
                .expect("Failed to create AwgCodeGenerationResultPy")
            })
            .collect();
        let result_handle_maps = results
            .result_handle_maps
            .into_iter()
            .map(|(result_source, map)| {
                (
                    ResultSourcePy {
                        device_id: result_source.device_id,
                        awg_id: result_source.awg_id,
                        integrator_idx: result_source.integrator_idx,
                    },
                    map,
                )
            })
            .collect();
        let measurements = results
            .measurements
            .into_iter()
            .map(|m| MeasurementPy {
                device: m.device,
                channel: m.channel,
                length: m.length,
            })
            .collect();
        let mut integration_unit_allocations = Vec::new();
        for alloc in results.integration_unit_allocations {
            integration_unit_allocations.push(IntegrationUnitAllocationPy {
                signal: id_store.resolve_unchecked(alloc.signal).to_string(),
                integrator_channels: alloc.channels.into_iter().map(|c| c as i64).collect(),
                kernel_count: alloc.kernel_count,
            });
        }
        SeqCGenOutputPy {
            awg_results,
            total_execution_time: results.total_execution_time,
            result_handle_maps,
            measurements,
            integration_unit_allocations,
        }
    }
}

#[pymethods]
impl SeqCGenOutputPy {
    #[getter]
    fn awg_results(&self) -> &Vec<Py<AwgCodeGenerationResultPy>> {
        &self.awg_results
    }
}

#[pyclass(name = "Measurement")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MeasurementPy {
    pub device: DeviceUid,
    #[pyo3(get)]
    pub channel: u16,
    #[pyo3(get)]
    pub length: Samples,
}

#[pymethods]
impl MeasurementPy {
    #[getter]
    fn device(&self) -> &str {
        &self.device
    }
}

#[pyclass(name = "ResultSource")]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct ResultSourcePy {
    #[pyo3(get)]
    pub device_id: String,
    #[pyo3(get)]
    pub awg_id: u16,
    #[pyo3(get)]
    pub integrator_idx: Option<u8>,
}

#[pyclass(name = "IntegrationUnitAllocation", eq, hash, frozen)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct IntegrationUnitAllocationPy {
    #[pyo3(get)]
    pub signal: String,
    #[pyo3(get)]
    pub integrator_channels: Vec<i64>,
    #[pyo3(get)]
    pub kernel_count: u8,
}

#[pyclass(name = "ChannelProperties")]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct ChannelPropertiesPy {
    #[pyo3(get)]
    pub channel: ChannelIndex,
    #[pyo3(get)]
    pub marker_mode: Option<String>,
}

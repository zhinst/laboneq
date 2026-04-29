// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

//! Result module for the code generator.
//!
//! This module defines the [`AwgCodeGenerationResultPy`] class, which is used to
//! represent the result of the code generation process for an AWG.

use pyo3::{IntoPyObjectExt, prelude::*};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use laboneq_common::named_id::NamedIdStore;
use laboneq_dsl::types::{PumpCancellationSource, Quantity, Unit};
use laboneq_units::duration::{Duration, Second};

use codegenerator::ir::compilation_job::ChannelIndex;
use codegenerator::ir::compilation_job::DeviceUid;
use codegenerator::ir::{Samples, compilation_job::AwgKind};
use codegenerator::result::SignalType;
use codegenerator::result::{AwgCodeGenerationResult, MarkerMode};
use codegenerator::result::{FixedValueOrParameter, ParameterPhaseIncrement};
use codegenerator::result::{PpcSettings, RoutedOutput, SeqCGenOutput};

use crate::waveform_sampler::WaveformSamplerPy;

#[pyo3::pyclass(name = "FeedbackRegisterConfig", skip_from_py_object)]
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

#[pyclass(name = "IntegrationKernel")]
#[derive(Debug)]
pub(crate) struct IntegrationKernelPy {
    #[pyo3(get)]
    pub signals: Vec<String>,
    #[pyo3(get)]
    pub samples_i: Py<PyAny>,
    #[pyo3(get)]
    pub samples_q: Py<PyAny>,
    #[pyo3(get)]
    pub downsampling_factor: Option<u8>,
    #[pyo3(get)]
    pub basename: String,
}

// SAFETY: Safe to Send/Sync across threads.
// - Py<PyAny> fields (samples_i, samples_q) use PyO3's thread-safe drop mechanism
// - Other fields (HashSet, String, Option<usize>) are inherently Send/Sync
// - No thread-local storage or thread-specific state
unsafe impl Send for IntegrationKernelPy {}
unsafe impl Sync for IntegrationKernelPy {}

#[pyclass(name = "SignalIntegrationInfo")]
#[derive(Debug)]
pub(crate) struct SignalIntegrationInfoPy {
    #[pyo3(get)]
    pub is_play: bool,
    #[pyo3(get)]
    pub length: Samples,
}

#[pyclass(name = "PpcSettings", frozen)]
struct PpcSettingsPy {
    #[pyo3(get)]
    device: String,
    #[pyo3(get)]
    channel: u16,

    #[pyo3(get)]
    alc_on: bool,
    #[pyo3(get)]
    pump_on: bool,
    #[pyo3(get)]
    pump_filter_on: bool,
    #[pyo3(get)]
    pump_power: Option<Py<PyAny>>, // Can be either a float or a parameter reference (string)
    #[pyo3(get)]
    pump_frequency: Option<Py<PyAny>>, // Can be either a float or a parameter reference (string)

    #[pyo3(get)]
    probe_on: bool,
    #[pyo3(get)]
    probe_power: Option<Py<PyAny>>, // Can be either a float or a parameter reference (string)
    #[pyo3(get)]
    probe_frequency: Option<Py<PyAny>>, // Can be either a float or a parameter reference (string)

    #[pyo3(get)]
    cancellation_on: bool,
    #[pyo3(get)]
    cancellation_phase: Option<Py<PyAny>>, // Can be either a float or a parameter reference (string)
    #[pyo3(get)]
    cancellation_attenuation: Option<Py<PyAny>>, // Can be either a float or a parameter reference (string)
    #[pyo3(get)]
    cancellation_source: String,
    #[pyo3(get)]
    cancellation_source_frequency: Option<f64>,

    // Sweep config JSON
    #[pyo3(get)]
    sweep_config: Option<String>,
}

/// Result structure for single AWG code generation.
#[pyclass(name = "AwgCodeGenerationResult", frozen)]
pub(crate) struct AwgCodeGenerationResultPy {
    #[pyo3(get)]
    awg_properties: AwgPropertiesPy,
    #[pyo3(get)]
    seqc: Py<SeqCProgramPy>,
    #[pyo3(get)]
    wave_indices: Vec<(String, (u32, String))>,
    #[pyo3(get)]
    command_table: Option<String>,
    #[pyo3(get)]
    sampled_waveforms: Vec<Py<SampledWaveformPy>>,
    #[pyo3(get)]
    integration_kernels: Vec<Py<IntegrationKernelPy>>,
    #[pyo3(get)]
    signal_delays: HashMap<String, f64>,
    #[pyo3(get)]
    integration_lengths: HashMap<String, Py<SignalIntegrationInfoPy>>,
    #[pyo3(get)]
    parameter_phase_increment_map: Option<HashMap<String, Vec<i64>>>,
    #[pyo3(get)]
    feedback_register_config: FeedbackRegisterConfigPy,
    #[pyo3(get)]
    channel_properties: Vec<Py<ChannelPropertiesPy>>,
    #[pyo3(get)]
    integration_weights: Vec<IntegrationWeightPy>,
    #[pyo3(get)]
    integration_unit_allocations: Vec<Py<IntegrationUnitAllocationPy>>,
}

#[pyclass(name = "DeviceProperties", skip_from_py_object, eq, frozen)]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct DevicePropertiesPy {
    #[pyo3(get)]
    pub uid: String,
    #[pyo3(get)]
    pub device_type: String,
    #[pyo3(get)]
    pub sampling_rate: Option<f64>,
}

#[pyclass(name = "AwgProperties", skip_from_py_object)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AwgPropertiesPy {
    awg_id: i64,
    device_uid: DeviceUid,
    kind: AwgKind,
}

#[pyclass(name = "QuantityPy", skip_from_py_object)]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct QuantityPy {
    #[pyo3(get)]
    value: f64,
    #[pyo3(get)]
    unit: Option<String>,
}

#[pyclass(name = "RoutedOutput", skip_from_py_object)]
#[derive(Debug)]
pub(crate) struct RoutedOutputPy {
    #[pyo3(get)]
    source_channel: u8,
    #[pyo3(get)]
    amplitude_scaling: Option<Py<PyAny>>, // Can be either a float or a parameter reference (string)
    #[pyo3(get)]
    phase_shift: Option<Py<PyAny>>, // Can be either a float or a parameter reference (string)
}

#[pymethods]
impl AwgPropertiesPy {
    #[getter]
    fn key(&self) -> (&str, i64) {
        (&self.device_uid, self.awg_id)
    }

    #[getter]
    fn signal_type(&self) -> &str {
        match self.kind {
            AwgKind::SINGLE => "SINGLE",
            AwgKind::DOUBLE => "DOUBLE",
            AwgKind::IQ => "IQ",
        }
    }
}

// SAFETY: Safe to Send/Sync across threads.
// - All Py<T> fields rely on PyO3's delayed reference count mechanism for safe
//   cross-thread drops (see https://docs.rs/pyo3/0.28.0/pyo3/struct.Py.html#impl-Drop-for-Py%3CT%3E)
// - SampledWaveformPy and IntegrationKernelPy are themselves Send/Sync
// - All other fields are standard Send/Sync types (String, Vec, HashMap, Option, etc.)
// - No thread-local storage or thread-specific state
// This enables passing compiled results to async executors and background threads.
unsafe impl Send for AwgCodeGenerationResultPy {}
unsafe impl Sync for AwgCodeGenerationResultPy {}

impl AwgCodeGenerationResultPy {
    pub(crate) fn create(
        py: Python,
        mut result: AwgCodeGenerationResult<WaveformSamplerPy>,
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
        let integration_kernels: Vec<Py<IntegrationKernelPy>> = result
            .integration_kernels
            .into_iter()
            .map(|weight| {
                Py::new(
                    py,
                    IntegrationKernelPy {
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
        let command_table = result
            .command_table
            .as_mut()
            .map(|ct| std::mem::take(&mut ct.src));
        let parameter_phase_increment_map = result.command_table.map(|ct| {
            ct.parameter_phase_increment_map
                .into_iter()
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

        let mut channel_properties = Vec::with_capacity(
            result.output_channel_properties.len() + result.input_channel_properties.len(),
        ); // Mixed both input and output

        for output in result.output_channel_properties.into_iter() {
            channel_properties.push(
                ChannelPropertiesPy {
                    signal: id_store.resolve(output.signal).unwrap().to_string(),
                    channel: output.channel,
                    direction: "OUT".to_string(),
                    marker_mode: output.marker_mode.map(|marker_mode| match marker_mode {
                        MarkerMode::Trigger => "TRIGGER".to_string(),
                        MarkerMode::Marker => "MARKER".to_string(),
                    }),
                    hw_oscillator_index: output.hw_oscillator_index,
                    amplitude: output.amplitude.map(|amp| {
                        fixed_value_or_parameterf64_to_pyany(py, &amp, id_store).unwrap()
                    }),
                    voltage_offset: Some(
                        fixed_value_or_parameterf64_to_pyany(py, &output.voltage_offset, id_store)
                            .unwrap(),
                    ),
                    gains: output.gains.map(|gains| {
                        Gains {
                            diagonal: fixed_value_or_parameterf64_to_pyany(
                                py,
                                &gains.diagonal,
                                id_store,
                            )
                            .unwrap(),
                            off_diagonal: fixed_value_or_parameterf64_to_pyany(
                                py,
                                &gains.off_diagonal,
                                id_store,
                            )
                            .unwrap(),
                        }
                        .into_pyobject(py)
                        .unwrap()
                        .unbind()
                    }),
                    port_mode: output.port_mode.map(|pm| pm.to_string().to_lowercase()),
                    port_delay: output.port_delay.map(|pd| {
                        fixed_value_or_parameter_duration_to_pyany(py, &pd, id_store).unwrap()
                    }),
                    range: output.range.map(|r| quantity_to_py(py, &r).unwrap()),
                    lo_frequency: output
                        .lo_frequency
                        .map(|lf| fixed_value_or_parameterf64_to_pyany(py, &lf, id_store).unwrap()),
                    routed_outputs: output
                        .routed_outputs
                        .iter()
                        .map(|o| routed_output_to_py(py, o, id_store).unwrap())
                        .collect(),
                }
                .into_pyobject(py)
                .unwrap()
                .unbind(),
            );
        }

        for properties in result.input_channel_properties.into_iter() {
            channel_properties.push(
                ChannelPropertiesPy {
                    signal: id_store.resolve(properties.signal).unwrap().to_string(),
                    channel: properties.channel,
                    direction: "IN".to_string(),
                    marker_mode: None, // Input channels don't have marker modes
                    hw_oscillator_index: properties.hw_oscillator_index,
                    amplitude: None, // Input channels don't have amplitude settings
                    voltage_offset: None, // Input channels don't have voltage offset
                    gains: None,     // Input channels don't have gain settings
                    port_mode: properties.port_mode.map(|pm| pm.to_string().to_lowercase()),
                    port_delay: properties.port_delay.map(|pd| {
                        fixed_value_or_parameter_duration_to_pyany(py, &pd, id_store).unwrap()
                    }),
                    range: properties.range.map(|r| quantity_to_py(py, &r).unwrap()),
                    lo_frequency: properties
                        .lo_frequency
                        .map(|lf| fixed_value_or_parameterf64_to_pyany(py, &lf, id_store).unwrap()),
                    routed_outputs: Vec::new(), // Input channels don't have routed outputs
                }
                .into_pyobject(py)
                .unwrap()
                .unbind(),
            );
        }

        let integration_unit_allocations = result
            .integrator_allocations
            .into_iter()
            .map(|alloc| {
                let alloc = IntegrationUnitAllocationPy {
                    signal: id_store.resolve_unchecked(alloc.signal).to_string(),
                    integration_units: alloc
                        .integration_units
                        .into_iter()
                        .map(|c| c as u16)
                        .collect(),
                    kernel_count: alloc.kernel_count.get(),
                    thresholds: alloc.thresholds,
                };
                Py::new(py, alloc).expect("Failed to create IntegrationUnitAllocationPy")
            })
            .collect();

        let output = AwgCodeGenerationResultPy {
            awg_properties: AwgPropertiesPy {
                awg_id: result.awg.key.index() as i64,
                device_uid: result.awg.key.device_name().clone(),
                kind: result.awg.kind,
            },
            seqc: {
                Py::new(
                    py,
                    SeqCProgramPy {
                        src: result.seqc.src,
                        sequencer: result.seqc.sequencer.to_string(),
                        dev_type: result.seqc.dev_type,
                        dev_opts: result.seqc.dev_opts,
                        awg_index: result.seqc.awg_index,
                        sampling_rate: result.seqc.sampling_rate,
                    },
                )
                .unwrap()
            },
            wave_indices,
            command_table,
            sampled_waveforms,
            integration_kernels,
            signal_delays,
            integration_lengths,
            parameter_phase_increment_map,
            feedback_register_config,
            channel_properties,
            integration_weights: result
                .integration_weights
                .into_iter()
                .map(|w| IntegrationWeightPy {
                    integration_units: w.integration_units.into_iter().map(|c| c as u16).collect(),
                    basename: w.basename,
                    downsampling_factor: w.downsampling_factor,
                })
                .collect(),
            integration_unit_allocations,
        };
        Ok(output)
    }
}

fn routed_output_to_py(
    py: Python,
    routed: &RoutedOutput,
    id_store: &NamedIdStore,
) -> PyResult<Py<RoutedOutputPy>> {
    let output = RoutedOutputPy {
        source_channel: routed.source_channel,
        amplitude_scaling: routed
            .amplitude_scaling
            .as_ref()
            .map(|amp| fixed_value_or_parameterf64_to_pyany(py, amp, id_store).unwrap()),
        phase_shift: routed
            .phase_shift
            .as_ref()
            .map(|ps| fixed_value_or_parameterf64_to_pyany(py, ps, id_store).unwrap()),
    };
    Py::new(py, output)
}

fn quantity_to_py(py: Python, quantity: &Quantity) -> PyResult<Py<QuantityPy>> {
    let value = quantity.value;
    let unit_str = match quantity.unit {
        Some(Unit::Volt) => Some("volt".to_string()),
        Some(Unit::Dbm) => Some("dBm".to_string()),
        _ => None,
    };
    Py::new(
        py,
        QuantityPy {
            value,
            unit: unit_str,
        },
    )
}

fn fixed_value_or_parameter_duration_to_pyany(
    py: Python,
    value: &FixedValueOrParameter<Duration<Second>>,
    id_store: &NamedIdStore,
) -> PyResult<Py<PyAny>> {
    match value {
        FixedValueOrParameter::Value(v) => v.value().into_py_any(py),
        FixedValueOrParameter::Parameter(p) => {
            let param_string = id_store.resolve(*p).unwrap();
            param_string.to_string().into_py_any(py)
        }
    }
}

fn fixed_value_or_parameterf64_to_pyany(
    py: Python,
    value: &FixedValueOrParameter<f64>,
    id_store: &NamedIdStore,
) -> PyResult<Py<PyAny>> {
    match value {
        FixedValueOrParameter::Value(v) => v.into_py_any(py),
        FixedValueOrParameter::Parameter(p) => {
            let param_string = id_store.resolve(*p).unwrap();
            param_string.to_string().into_py_any(py)
        }
    }
}

#[pyclass(name = "SeqCGenOutput")]
pub struct SeqCGenOutputPy {
    #[pyo3(get)]
    device_properties: Vec<Py<DevicePropertiesPy>>,
    #[pyo3(get)]
    awg_results: Vec<Py<AwgCodeGenerationResultPy>>,
    #[pyo3(get)]
    total_execution_time: f64,
    #[pyo3(get)]
    result_handle_maps: HashMap<ResultSourcePy, Vec<Vec<String>>>,
    #[pyo3(get)]
    measurements: Vec<MeasurementPy>,
    #[pyo3(get)]
    ppc_settings: Vec<Py<PpcSettingsPy>>,
}

impl SeqCGenOutputPy {
    pub fn new(
        py: Python,
        mut results: SeqCGenOutput<WaveformSamplerPy>,
        id_store: &NamedIdStore,
    ) -> Self {
        let awg_results: Vec<Py<AwgCodeGenerationResultPy>> = results
            .awg_results
            .drain(..)
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
            .drain()
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
            .drain(..)
            .map(|m| MeasurementPy {
                device: m.device,
                channel: m.channel,
                length: m.length,
            })
            .collect();

        let device_properties = create_device_properties(py, &results, id_store);

        SeqCGenOutputPy {
            device_properties,
            awg_results,
            total_execution_time: results.total_execution_time,
            result_handle_maps,
            measurements,
            ppc_settings: convert_ppc_settings(py, results.ppc_settings, id_store),
        }
    }
}

fn create_device_properties(
    py: Python,
    output: &SeqCGenOutput<WaveformSamplerPy>,
    id_store: &NamedIdStore,
) -> Vec<Py<DevicePropertiesPy>> {
    let mut devices = output
        .device_properties
        .iter()
        .map(|dp| {
            let dp_py = DevicePropertiesPy {
                uid: dp.uid.to_string(),
                device_type: dp.kind.as_str().to_string(),
                sampling_rate: dp.sampling_rate,
            };
            Py::new(py, dp_py).unwrap()
        })
        .collect::<Vec<_>>();

    let auxiliary_devices = output
        .auxiliary_device_properties
        .iter()
        .map(|dp| {
            let dp_py = DevicePropertiesPy {
                uid: id_store.resolve(dp.uid()).unwrap().to_string(),
                device_type: dp.kind().to_string(),
                sampling_rate: None, // Auxiliary devices don't have sampling rates
            };
            Py::new(py, dp_py).unwrap()
        })
        .collect::<Vec<_>>();

    devices.extend(auxiliary_devices);
    devices
}

fn convert_ppc_settings(
    py: Python,
    settings: Vec<PpcSettings>,
    id_store: &NamedIdStore,
) -> Vec<Py<PpcSettingsPy>> {
    settings
        .into_iter()
        .map(|s| {
            let ppc_settings_py = PpcSettingsPy {
                device: id_store.resolve(s.device).unwrap().to_string(),
                channel: s.channel,
                alc_on: s.alc_on,
                pump_on: s.pump_on,
                pump_filter_on: s.pump_filter_on,
                pump_power: s
                    .pump_power
                    .map(|pp| fixed_value_or_parameterf64_to_pyany(py, &pp, id_store).unwrap()),
                pump_frequency: s
                    .pump_frequency
                    .map(|pf| fixed_value_or_parameterf64_to_pyany(py, &pf, id_store).unwrap()),
                probe_on: s.probe_on,
                probe_power: s
                    .probe_power
                    .map(|pp| fixed_value_or_parameterf64_to_pyany(py, &pp, id_store).unwrap()),
                probe_frequency: s
                    .probe_frequency
                    .map(|pf| fixed_value_or_parameterf64_to_pyany(py, &pf, id_store).unwrap()),
                cancellation_on: s.cancellation_on,
                cancellation_phase: s
                    .cancellation_phase
                    .map(|cp| fixed_value_or_parameterf64_to_pyany(py, &cp, id_store).unwrap()),
                cancellation_attenuation: s
                    .cancellation_attenuation
                    .map(|ca| fixed_value_or_parameterf64_to_pyany(py, &ca, id_store).unwrap()),
                cancellation_source: match s.cancellation_source {
                    PumpCancellationSource::Internal => "INTERNAL".to_string(),
                    PumpCancellationSource::External => "EXTERNAL".to_string(),
                },
                cancellation_source_frequency: s.cancellation_source_frequency,
                sweep_config: s.sweep_config,
            };
            Py::new(py, ppc_settings_py).unwrap()
        })
        .collect()
}

#[pymethods]
impl SeqCGenOutputPy {
    #[getter]
    fn awg_results(&self) -> &Vec<Py<AwgCodeGenerationResultPy>> {
        &self.awg_results
    }
}

#[pyclass(name = "Measurement", skip_from_py_object)]
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

#[pyclass(name = "ResultSource", skip_from_py_object)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct ResultSourcePy {
    #[pyo3(get)]
    pub device_id: String,
    #[pyo3(get)]
    pub awg_id: u16,
    #[pyo3(get)]
    pub integrator_idx: Option<u8>,
}

#[pyclass(
    name = "IntegrationUnitAllocation",
    eq,
    ord,
    frozen,
    skip_from_py_object
)]
#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub(crate) struct IntegrationUnitAllocationPy {
    #[pyo3(get)]
    pub signal: String,
    #[pyo3(get)]
    pub integration_units: Vec<u16>,
    #[pyo3(get)]
    pub kernel_count: u8,
    #[pyo3(get)]
    pub thresholds: Vec<f64>,
}

#[pyclass(name = "ChannelProperties", skip_from_py_object)]
#[derive(Debug)]
pub(crate) struct ChannelPropertiesPy {
    #[pyo3(get)]
    pub signal: String,
    #[pyo3(get)]
    pub channel: ChannelIndex,
    #[pyo3(get)]
    pub direction: String,
    #[pyo3(get)]
    pub marker_mode: Option<String>,
    #[pyo3(get)]
    pub hw_oscillator_index: Option<u16>,
    #[pyo3(get)]
    pub amplitude: Option<Py<PyAny>>, // Can be either a float or a parameter reference (string)
    #[pyo3(get)]
    pub voltage_offset: Option<Py<PyAny>>, // Can be either a float or a parameter reference (string)
    #[pyo3(get)]
    pub gains: Option<Py<Gains>>, // Internal field to hold the Gains struct
    #[pyo3(get)]
    pub port_mode: Option<String>,
    #[pyo3(get)]
    pub port_delay: Option<Py<PyAny>>, // Can be either a float or a parameter reference (string)
    #[pyo3(get)]
    pub range: Option<Py<QuantityPy>>,
    #[pyo3(get)]
    pub lo_frequency: Option<Py<PyAny>>, // Can be either a float or a parameter reference (string)
    #[pyo3(get)]
    pub routed_outputs: Vec<Py<RoutedOutputPy>>,
}

#[pyclass(name = "Gains", skip_from_py_object)]
#[derive(Debug)]
pub(crate) struct Gains {
    #[pyo3(get)]
    pub diagonal: Py<PyAny>, // Can be either a float or a parameter reference (string)
    #[pyo3(get)]
    pub off_diagonal: Py<PyAny>, // Can be either a float or a parameter reference (string)
}

#[pyclass(name = "SeqCProgram", skip_from_py_object)]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct SeqCProgramPy {
    #[pyo3(get)]
    src: String,
    #[pyo3(get)]
    dev_type: String,
    #[pyo3(get)]
    dev_opts: Vec<String>,
    #[pyo3(get)]
    awg_index: u16,
    #[pyo3(get)]
    sequencer: String,
    #[pyo3(get)]
    sampling_rate: Option<f64>,
}

#[pyclass(name = "IntegrationWeight", skip_from_py_object)]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct IntegrationWeightPy {
    #[pyo3(get)]
    pub integration_units: Vec<u16>,
    #[pyo3(get)]
    pub basename: String,
    #[pyo3(get)]
    pub downsampling_factor: u8,
}

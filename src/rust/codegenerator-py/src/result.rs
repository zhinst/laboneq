// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

//! Result module for the code generator.
//!
//! This module defines the [`AwgCodeGenerationResultPy`] class, which is used to
//! represent the result of the code generation process for an AWG.

use codegenerator::pulse_map::PulseMap;
use laboneq_py_utils::py_export::{complex_or_float_to_py, pulse_parameters_to_py_dict};
use laboneq_py_utils::py_object_interner::PyObjectInterner;
use numpy::PyArray1;
use pyo3::types::{PyDict, PyList};
use pyo3::{IntoPyObjectExt, intern, prelude::*};
use std::collections::HashMap;

use laboneq_common::named_id::NamedIdStore;
use laboneq_dsl::types::{ExternalParameterUid, PumpCancellationSource, Quantity, Unit};
use laboneq_units::duration::{Duration, Second};

use codegenerator::ir::compilation_job::ChannelIndex;
use codegenerator::ir::compilation_job::DeviceUid;
use codegenerator::ir::{Samples, compilation_job::AwgKind};
use codegenerator::result::{AwgCodeGenerationResult, MarkerMode};
use codegenerator::result::{ChannelOscillator, CodegenWaveform, SignalType};
use codegenerator::result::{FixedValueOrParameter, ParameterPhaseIncrement};
use codegenerator::result::{PpcSettings, RoutedOutput, SeqCGenOutput};
use codegenerator::waveform_sampler::{SampleBuffer, WaveformStore};

use crate::common_types::PortModePy;
use crate::utils::mixer_type_to_py;

type DeviceUidString = String;
type SignalUidString = String;

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

/// Compare two optional Python values for equality, honouring their Python
/// `__eq__` semantics.
fn py_opt_eq<T>(py: Python<'_>, a: &Option<Py<T>>, b: &Option<Py<T>>) -> PyResult<bool> {
    match (a, b) {
        (None, None) => Ok(true),
        (Some(a), Some(b)) => a.bind(py).as_any().eq(b.bind(py).as_any()),
        _ => Ok(false),
    }
}

#[pymethods]
impl PpcSettingsPy {
    // The linker compares PPC settings across real-time iterations to ensure
    // they are static. Without an explicit `__eq__`, distinct instances would
    // compare unequal by identity, so identical configurations across
    // near-time steps would spuriously fail compilation.
    fn __eq__(&self, py: Python<'_>, other: &PpcSettingsPy) -> PyResult<bool> {
        Ok(self.device == other.device
            && self.channel == other.channel
            && self.alc_on == other.alc_on
            && self.pump_on == other.pump_on
            && self.pump_filter_on == other.pump_filter_on
            && py_opt_eq(py, &self.pump_power, &other.pump_power)?
            && py_opt_eq(py, &self.pump_frequency, &other.pump_frequency)?
            && self.probe_on == other.probe_on
            && py_opt_eq(py, &self.probe_power, &other.probe_power)?
            && py_opt_eq(py, &self.probe_frequency, &other.probe_frequency)?
            && self.cancellation_on == other.cancellation_on
            && py_opt_eq(py, &self.cancellation_phase, &other.cancellation_phase)?
            && py_opt_eq(
                py,
                &self.cancellation_attenuation,
                &other.cancellation_attenuation,
            )?
            && self.cancellation_source == other.cancellation_source
            && self.cancellation_source_frequency == other.cancellation_source_frequency
            && self.sweep_config == other.sweep_config)
    }
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
    #[pyo3(get)]
    result_length: Option<usize>,
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

#[pyclass(name = "QuantityPy", eq, skip_from_py_object)]
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
impl RoutedOutputPy {
    fn __eq__(&self, py: Python<'_>, other: &RoutedOutputPy) -> PyResult<bool> {
        Ok(self.source_channel == other.source_channel
            && py_opt_eq(py, &self.amplitude_scaling, &other.amplitude_scaling)?
            && py_opt_eq(py, &self.phase_shift, &other.phase_shift)?)
    }
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

impl AwgCodeGenerationResultPy {
    pub(crate) fn create(
        py: Python,
        mut result: AwgCodeGenerationResult,
        id_store: &NamedIdStore,
    ) -> PyResult<Self> {
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
                    port_mode: output
                        .port_mode
                        .map(|pm| PortModePy::from(pm).into_pyobject(py).unwrap().unbind()),
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
                    scheduler_delay: output.scheduler_delay.into(),
                    output_mute_enable: output.output_mute_enable,
                    hardware_oscillator: output
                        .oscillator
                        .as_ref()
                        .map(|osc| oscillator_to_py(py, osc, id_store).unwrap()),
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
                    marker_mode: None,    // Input channels don't have marker modes
                    amplitude: None,      // Input channels don't have amplitude settings
                    voltage_offset: None, // Input channels don't have voltage offset
                    gains: None,          // Input channels don't have gain settings
                    port_mode: properties
                        .port_mode
                        .map(|pm| PortModePy::from(pm).into_pyobject(py).unwrap().unbind()),
                    port_delay: properties.port_delay.map(|pd| {
                        fixed_value_or_parameter_duration_to_pyany(py, &pd, id_store).unwrap()
                    }),
                    range: properties.range.map(|r| quantity_to_py(py, &r).unwrap()),
                    lo_frequency: properties
                        .lo_frequency
                        .map(|lf| fixed_value_or_parameterf64_to_pyany(py, &lf, id_store).unwrap()),
                    routed_outputs: Vec::new(), // Input channels don't have routed outputs
                    scheduler_delay: properties.scheduler_delay.into(),
                    output_mute_enable: false, // Input channels don't have output mute
                    hardware_oscillator: properties
                        .oscillator
                        .as_ref()
                        .map(|osc| oscillator_to_py(py, osc, id_store).unwrap()),
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
            result_length: result.result_length,
        };
        Ok(output)
    }
}

fn oscillator_to_py(
    py: Python,
    oscillator: &ChannelOscillator,
    id_store: &NamedIdStore,
) -> PyResult<Py<HardwareOscillatorPy>> {
    let hw_oscillator_py = HardwareOscillatorPy {
        uid: oscillator.uid.clone(),
        index: oscillator.index,
        frequency: fixed_value_or_parameterf64_to_pyany(py, &oscillator.frequency, id_store)?,
    };
    Py::new(py, hw_oscillator_py)
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
    /// Map of device ID to list of signals that require long readout
    #[pyo3(get)]
    requires_long_readout: Py<PyDict>,
    #[pyo3(get)]
    waves: Py<PyDict>,
    #[pyo3(get)]
    pulse_map: Py<PyDict>,
}

impl SeqCGenOutputPy {
    pub fn new(
        py: Python,
        mut results: SeqCGenOutput,
        id_store: &NamedIdStore,
        py_object_store: &PyObjectInterner<ExternalParameterUid>,
    ) -> PyResult<Self> {
        let result_handle_maps = results
            .awg_results
            .iter()
            .flat_map(|result| {
                result.result_handle_maps.iter().map(move |(source, map)| {
                    (
                        ResultSourcePy {
                            device_id: source.device_id.to_string(),
                            awg_id: source.awg_id,
                            integrator_idx: source.integrator_idx,
                        },
                        map.iter()
                            .map(|handles| {
                                handles
                                    .iter()
                                    .map(|handle| handle.to_string())
                                    .collect::<Vec<String>>()
                            })
                            .collect::<Vec<Vec<String>>>(),
                    )
                })
            })
            .collect();
        let requires_long_readout = collect_requires_long_readout(py, &results, id_store)?;

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

        let output = SeqCGenOutputPy {
            device_properties,
            awg_results,
            total_execution_time: results.total_execution_time,
            result_handle_maps,
            measurements,
            ppc_settings: convert_ppc_settings(py, results.ppc_settings, id_store),
            requires_long_readout: requires_long_readout.into(),
            waves: create_waves(py, results.waveforms, results.waveform_store).unwrap(),
            pulse_map: pulse_map_to_py(py, results.pulse_map, id_store, py_object_store)?.unbind(),
        };
        Ok(output)
    }
}

fn collect_requires_long_readout<'py>(
    py: Python<'py>,
    results: &SeqCGenOutput,
    id_store: &NamedIdStore,
) -> PyResult<Bound<'py, PyDict>> {
    fn insert_to_mapping(
        mapping: &Bound<'_, PyDict>,
        signal: &str,
        device_uid: &str,
    ) -> PyResult<()> {
        if let Some(signals) = mapping.get_item(device_uid)? {
            signals.cast_into::<PyList>()?.append(signal)?;
        } else {
            mapping.set_item(device_uid, vec![signal])?;
        }
        Ok(())
    }

    // Build directly into a PyDict to preserve order of devices
    let mapping = PyDict::new(py);
    for result in &results.awg_results {
        let device_id = result.awg.key.device_name().to_string();
        for ch in &result.output_channel_properties {
            if ch.requires_long_readout {
                let signal = id_store.resolve(ch.signal).unwrap().to_string();
                insert_to_mapping(&mapping, &signal, &device_id)?;
            }
        }
        for ch in &result.input_channel_properties {
            if ch.requires_long_readout {
                let signal = id_store.resolve(ch.signal).unwrap().to_string();
                insert_to_mapping(&mapping, &signal, &device_id)?;
            }
        }
    }
    Ok(mapping)
}

fn create_device_properties(
    py: Python,
    output: &SeqCGenOutput,
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
                device: id_store
                    .resolve(s.ppc_channel.device_uid())
                    .unwrap()
                    .to_string(),
                channel: s.ppc_channel.channel(),
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
    pub device_id: DeviceUidString,
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
    pub signal: SignalUidString,
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
    pub signal: SignalUidString,
    #[pyo3(get)]
    pub channel: ChannelIndex,
    #[pyo3(get)]
    pub direction: String,
    #[pyo3(get)]
    pub marker_mode: Option<String>,
    #[pyo3(get)]
    pub amplitude: Option<Py<PyAny>>, // Can be either a float or a parameter reference (string)
    #[pyo3(get)]
    pub voltage_offset: Option<Py<PyAny>>, // Can be either a float or a parameter reference (string)
    #[pyo3(get)]
    pub gains: Option<Py<Gains>>, // Internal field to hold the Gains struct
    #[pyo3(get)]
    pub port_mode: Option<Py<PortModePy>>,
    #[pyo3(get)]
    pub port_delay: Option<Py<PyAny>>, // Can be either a float or a parameter reference (string)
    #[pyo3(get)]
    pub range: Option<Py<QuantityPy>>,
    #[pyo3(get)]
    pub lo_frequency: Option<Py<PyAny>>, // Can be either a float or a parameter reference (string)
    #[pyo3(get)]
    pub routed_outputs: Vec<Py<RoutedOutputPy>>,
    #[pyo3(get)]
    pub scheduler_delay: f64,
    #[pyo3(get)]
    output_mute_enable: bool,
    #[pyo3(get)]
    hardware_oscillator: Option<Py<HardwareOscillatorPy>>,
}

#[pymethods]
impl ChannelPropertiesPy {
    fn __eq__(&self, py: Python<'_>, other: &ChannelPropertiesPy) -> PyResult<bool> {
        if self.routed_outputs.len() != other.routed_outputs.len() {
            return Ok(false);
        }
        for (a, b) in self.routed_outputs.iter().zip(other.routed_outputs.iter()) {
            if !a.bind(py).as_any().eq(b.bind(py).as_any())? {
                return Ok(false);
            }
        }
        Ok(self.signal == other.signal
            && self.channel == other.channel
            && self.direction == other.direction
            && self.marker_mode == other.marker_mode
            && py_opt_eq(py, &self.amplitude, &other.amplitude)?
            && py_opt_eq(py, &self.voltage_offset, &other.voltage_offset)?
            && py_opt_eq(py, &self.gains, &other.gains)?
            && py_opt_eq(py, &self.port_mode, &other.port_mode)?
            && py_opt_eq(py, &self.port_delay, &other.port_delay)?
            && py_opt_eq(py, &self.range, &other.range)?
            && py_opt_eq(py, &self.lo_frequency, &other.lo_frequency)?
            && self.scheduler_delay == other.scheduler_delay
            && self.output_mute_enable == other.output_mute_enable
            && py_opt_eq(py, &self.hardware_oscillator, &other.hardware_oscillator)?)
    }
}

#[pyclass(name = "Gains", skip_from_py_object)]
#[derive(Debug)]
pub(crate) struct Gains {
    #[pyo3(get)]
    pub diagonal: Py<PyAny>, // Can be either a float or a parameter reference (string)
    #[pyo3(get)]
    pub off_diagonal: Py<PyAny>, // Can be either a float or a parameter reference (string)
}

#[pymethods]
impl Gains {
    fn __eq__(&self, py: Python<'_>, other: &Gains) -> PyResult<bool> {
        Ok(self.diagonal.bind(py).eq(other.diagonal.bind(py))?
            && self.off_diagonal.bind(py).eq(other.off_diagonal.bind(py))?)
    }
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

#[pyclass(name = "Oscillator", skip_from_py_object)]
#[derive(Debug)]
pub(crate) struct HardwareOscillatorPy {
    #[pyo3(get)]
    pub uid: String,
    #[pyo3(get)]
    pub index: u16,
    #[pyo3(get)]
    pub frequency: Py<PyAny>, // Can be either a float or a parameter reference (string)
}

#[pymethods]
impl HardwareOscillatorPy {
    fn __eq__(&self, py: Python<'_>, other: &HardwareOscillatorPy) -> PyResult<bool> {
        Ok(self.uid == other.uid
            && self.index == other.index
            && self.frequency.bind(py).eq(other.frequency.bind(py))?)
    }
}

fn create_waves(
    py: Python,
    waveforms: Vec<CodegenWaveform>,
    store: WaveformStore,
) -> PyResult<Py<PyDict>> {
    let waves_dict = PyDict::new(py);
    for waveform in waveforms.into_iter() {
        let key = waveform.filename();
        let buffer = store
            .get(waveform.wave_key())
            .expect("WaveformStore missing entry for CodegenWaveform key");
        let py_waveform = waveform_to_py(py, waveform, buffer)?;
        waves_dict.set_item(key, py_waveform)?;
    }
    Ok(waves_dict.into())
}

fn waveform_to_py(
    py: Python,
    waveform: CodegenWaveform,
    buffer: &SampleBuffer,
) -> PyResult<Py<PyAny>> {
    let py_class = py
        .import(intern!(py, "laboneq.data.scheduled_experiment"))?
        .getattr(intern!(py, "CodegenWaveform"))?;

    let kwargs = PyDict::new(py);
    if let Some(compression_properties) = &waveform.compression_properties {
        kwargs.set_item(intern!(py, "hold_start"), compression_properties.hold_start)?;
        kwargs.set_item(
            intern!(py, "hold_length"),
            compression_properties.hold_length,
        )?;
    }
    kwargs.set_item(
        intern!(py, "downsampling_factor"),
        waveform.downsampling_factor,
    )?;
    kwargs.set_item(intern!(py, "samples"), samples_to_py(py, buffer)?)?;
    Ok(py_class.call((), Some(&kwargs))?.into())
}

fn samples_to_py(py: Python, samples: &SampleBuffer) -> PyResult<Py<PyAny>> {
    match samples {
        SampleBuffer::Float64(v) => PyArray1::from_slice(py, v).into_py_any(py),
        SampleBuffer::Complex64(v) => PyArray1::from_slice(py, v).into_py_any(py),
        SampleBuffer::U8(v) => PyArray1::from_slice(py, v).into_py_any(py),
    }
}

fn pulse_map_to_py<'py>(
    py: Python<'py>,
    pulse_map: PulseMap,
    id_store: &NamedIdStore,
    py_object_store: &PyObjectInterner<ExternalParameterUid>,
) -> PyResult<Bound<'py, PyDict>> {
    let pulse_map_entry_cls = py
        .import(intern!(py, "laboneq.data.scheduled_experiment"))?
        .getattr(intern!(py, "PulseMapEntry"))?;

    let pulse_waveform_map_cls = py
        .import(intern!(py, "laboneq.data.scheduled_experiment"))?
        .getattr(intern!(py, "PulseWaveformMap"))?;

    let pulse_instance_cls = py
        .import(intern!(py, "laboneq.data.scheduled_experiment"))?
        .getattr(intern!(py, "PulseInstance"))?;

    let dict = PyDict::new(py);

    let pulse_map = pulse_map.into_map();
    for (entry_id, entry) in pulse_map {
        let pulse_uid = id_store
            .resolve(entry_id)
            .expect("Internal error: Failed to resolve pulse UID");
        let waveforms = PyDict::new(py);
        let waveforms_compressed = PyDict::new(py);

        for (waveform_signature, pulse_waveform_map) in entry {
            let pulse_waveform_map_kwargs = PyDict::new(py);

            pulse_waveform_map_kwargs.set_item(
                intern!(py, "sampling_rate"),
                pulse_waveform_map.sampling_rate,
            )?;
            pulse_waveform_map_kwargs.set_item(
                intern!(py, "length_samples"),
                pulse_waveform_map.length_samples,
            )?;
            pulse_waveform_map_kwargs.set_item(
                intern!(py, "signal_type"),
                if pulse_waveform_map.iq_modulation {
                    "iq"
                } else {
                    "single"
                },
            )?;
            pulse_waveform_map_kwargs.set_item(
                intern!(py, "mixer_type"),
                pulse_waveform_map
                    .mixer_type
                    .as_ref()
                    .map(|mixer| mixer_type_to_py(py, mixer))
                    .transpose()?,
            )?;

            let instances = pulse_waveform_map
                .instances
                .into_iter()
                .map(|instance| {
                    let kwargs = PyDict::new(py);
                    kwargs.set_item(intern!(py, "offset_samples"), instance.offset_samples)?;
                    kwargs.set_item(
                        intern!(py, "amplitude"),
                        instance
                            .amplitude
                            .map(|amp| complex_or_float_to_py(py, &amp))
                            .transpose()?,
                    )?;
                    kwargs.set_item(intern!(py, "length"), instance.length)?;
                    kwargs.set_item(intern!(py, "iq_phase"), instance.iq_phase)?;
                    kwargs.set_item(
                        intern!(py, "modulation_frequency"),
                        instance.modulation_frequency,
                    )?;
                    kwargs.set_item(intern!(py, "channel"), instance.channel)?;
                    kwargs.set_item(intern!(py, "needs_conjugate"), instance.needs_conjugate)?;

                    let play_pulse_parameters = pulse_parameters_to_py_dict(
                        py,
                        &instance.parameters,
                        id_store,
                        py_object_store,
                    )?;
                    let pulse_pulse_parameters = pulse_parameters_to_py_dict(
                        py,
                        &instance.pulse_parameters,
                        id_store,
                        py_object_store,
                    )?;

                    kwargs.set_item(intern!(py, "play_pulse_parameters"), play_pulse_parameters)?;
                    kwargs.set_item(
                        intern!(py, "pulse_pulse_parameters"),
                        pulse_pulse_parameters,
                    )?;

                    kwargs.set_item(intern!(py, "has_marker1"), instance.has_marker1)?;
                    kwargs.set_item(intern!(py, "has_marker2"), instance.has_marker2)?;
                    kwargs.set_item(intern!(py, "can_compress"), instance.can_compress)?;
                    pulse_instance_cls.call((), Some(&kwargs))
                })
                .collect::<PyResult<Vec<_>>>()?;

            pulse_waveform_map_kwargs.set_item(intern!(py, "instances"), instances)?;

            let is_compressed = pulse_waveform_map.compressed;
            let pulse_waveform_map =
                pulse_waveform_map_cls.call((), Some(&pulse_waveform_map_kwargs))?;
            if is_compressed {
                waveforms_compressed
                    .set_item(waveform_signature.to_string(), pulse_waveform_map)?;
            } else {
                waveforms.set_item(waveform_signature.to_string(), pulse_waveform_map)?;
            }
        }

        if !waveforms.is_empty() {
            let pulse_map_entry = pulse_map_entry_cls.call1((waveforms,))?;
            dict.set_item(pulse_uid, pulse_map_entry)?;
        }

        if !waveforms_compressed.is_empty() {
            let pulse_map_entry = pulse_map_entry_cls.call1((waveforms_compressed,))?;
            // let pulse_uid = format!("{}_compr_", pulse_uid);
            dict.set_item(pulse_uid, pulse_map_entry)?;
        }
    }
    Ok(dict)
}

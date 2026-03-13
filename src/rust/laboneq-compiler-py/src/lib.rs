// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for laboneq compiler.

pub mod error;

use std::collections::HashMap;
use std::collections::HashSet;
use std::num::NonZeroU32;
use std::str::FromStr;
use std::sync::Arc;

use crate::error::{Error, Result, create_error_message};
use crate::experiment::Experiment;
use crate::experiment_context::experiment_context_from_experiment;
use crate::experiment_processor::process_experiment;
use crate::experiment_validation::validate_experiment;
use crate::parameter_store::create_parameter_store;
use crate::py_awg::AwgInfoPy;
use crate::py_conversion::ExperimentBuilder;
use crate::py_conversion::register_experiment_signals;
use crate::py_device::DevicePy;
use crate::py_experiment::ExperimentPy;
use crate::py_signal::AmplifierPumpPy;
use crate::py_signal::SweepParameterPy;
use crate::py_signal::{OscillatorPy, py_signal_to_signal};
use crate::qccs_feedback_calculator::QccsFeedbackCalculator;
use crate::setup_processor::process_setup;
use crate::signal_properties::SignalProperties;

use laboneq_common::device_options::DeviceOptions;
use laboneq_common::named_id::{NamedIdStore, resolve_ids};
use laboneq_common::types::AuxiliaryDeviceKind;
use laboneq_common::types::DeviceKind;
use laboneq_dsl::device_setup::AuxiliaryDevice;
use laboneq_ir::ExperimentIr;
use laboneq_ir::awg::AwgCore;
use laboneq_ir::pulse_sheet_schedule::PulseSheetSchedule;
use laboneq_ir::system::AwgDevice;
use laboneq_ir::system::DeviceSetup;
use laboneq_py_utils::experiment_ir::ExperimentIrPy;
use laboneq_scheduler::{ChunkingInfo, ExperimentContext as SchedulerContext, schedule_experiment};
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

#[cfg(test)]
mod test_py;

pub(crate) mod capnp_deserializer;
use capnp_deserializer::fix_derived_parameters;
pub(crate) mod capnp_serializer;
mod experiment_equivalence;
mod py_conversion;
mod py_helpers;
use py_conversion::build_experiment;
mod py_signal;
use py_signal::SignalPy;
pub(crate) mod experiment;
mod experiment_context;
mod experiment_processor;
mod experiment_validation;
mod parameter_store;
mod py_awg;
mod py_device;
mod qccs_feedback_calculator;
mod signal_view;
use signal_view::signal_views;
mod py_experiment;
mod setup_processor;
mod signal_properties;

pub(crate) struct SetupProperties {
    signals: Vec<SignalProperties>,
    awg_devices: Vec<AwgDevice>,
    auxiliary_devices: Vec<AuxiliaryDevice>,
    awgs: Vec<AwgCore>,
}

/// Convert the Python experiment definition into internal representation.
///
/// Arguments:
/// - `experiment`: The Python experiment definition (DSL)
/// - `signals`: List of intermediate Python signal representations.
/// - `devices`: List of intermediate Python device representations.
/// - `awgs`: List of intermediate Python AWG representations.
pub(crate) fn experiment_py_to_experiment(
    experiment: &Bound<'_, PyAny>,
    signals: Vec<Bound<'_, SignalPy>>,
    devices: Vec<Bound<'_, DevicePy>>,
    awgs: Vec<Bound<'_, AwgInfoPy>>,
) -> Result<(Experiment, SetupProperties)> {
    let mut builder = ExperimentBuilder::new(experiment.py());
    // Register experiment signals first to get consistent UID ordering.
    register_experiment_signals(experiment, &mut builder)?;

    // Build devices
    // NOTE: SHFQC split is already done at this point by the Python lib,
    // shall be moved here later when the split logic is implemented in Rust.
    let mut awg_devices = Vec::new();
    let mut auxiliary_devices = Vec::new();
    for device in devices {
        let device = device.borrow();
        if let Ok(device_kind) = DeviceKind::from_str(&device.kind) {
            let mut device_builder = AwgDevice::builder(
                builder.id_store.get_or_insert(&device.uid).into(),
                device.physical_device_uid.into(),
                device_kind,
            )
            .shfqc(device.is_shfqc)
            .options(DeviceOptions::new(device.options.clone()));
            if let Some(ref_clk) = &device.reference_clock {
                device_builder =
                    device_builder.reference_clock(ref_clk.parse().map_err(Error::new)?)
            }
            let device = device_builder.build();
            awg_devices.push(device);
        } else if let Ok(device_kind) = AuxiliaryDeviceKind::from_str(&device.kind) {
            // For auxiliary device, we ignore the rest of the fields. They are
            // processed separately in the Controller.
            let device = AuxiliaryDevice::new(
                builder.id_store.get_or_insert(&device.uid).into(),
                device_kind,
            );
            auxiliary_devices.push(device);
        } else {
            return Err(Error::new(format!("Unknown device: '{}'", device.kind)));
        }
    }

    let signals = signals
        .into_iter()
        .map(|s| {
            let signal = py_signal_to_signal(s.py(), &s.borrow(), &mut builder)?;
            Ok(signal)
        })
        .collect::<Result<_>>()?;

    let awg_mapping = awgs.into_iter().map(|awg| {
        let awg = awg.borrow();
        AwgCore::new(awg.uid, awg.number.clone().into())
    });

    build_experiment(experiment, &mut builder)?;
    Ok((
        Experiment {
            root: builder.root,
            id_store: builder.id_store.into(),
            parameters: builder.parameters,
            pulses: builder.pulses,
            py_object_store: builder.py_object_store.into(),
        },
        SetupProperties {
            signals,
            awg_devices,
            auxiliary_devices,
            awgs: awg_mapping.collect(),
        },
    ))
}

/// Serialize a Python experiment object to Cap'n Proto bytes.
#[pyfunction(name = "serialize_experiment", signature = (experiment, packed=false))]
fn serialize_experiment_py(experiment: &Bound<'_, PyAny>, packed: bool) -> Result<Vec<u8>> {
    capnp_serializer::serialize_experiment(experiment, packed)
}

/// Build an experiment from Cap'n Proto bytes plus device/signal configuration.
#[pyfunction(name = "build_experiment_capnp", signature = (capnp_data, signals, devices, awgs, desktop_setup, packed=false))]
fn build_experiment_capnp_py(
    py: Python<'_>,
    capnp_data: &[u8],
    signals: Vec<Bound<'_, SignalPy>>,
    devices: Vec<Bound<'_, DevicePy>>,
    awgs: Vec<Bound<'_, AwgInfoPy>>,
    desktop_setup: bool,
    packed: bool,
) -> Result<ExperimentPy> {
    // 1. Deserialize the experiment tree from Cap'n Proto.
    let deserialized = capnp_deserializer::deserialize_experiment(py, capnp_data, packed)?;

    // 2. Use the deserialized NamedIdStore as the shared store for all ID resolution.
    //    Create an ExperimentBuilder with the deserialized store so that signal/device
    //    processing registers IDs into the same namespace.
    let mut builder = ExperimentBuilder::new(py);
    builder.id_store = deserialized.id_store;
    builder.available_signals = deserialized.available_signals;
    builder.parameters = deserialized.parameters;
    builder.pulses = deserialized.pulses;
    builder.root = deserialized.root;
    for (uid, value) in deserialized.external_parameter_values {
        builder.py_object_store.insert(uid, value);
    }

    // Build devices
    // NOTE: SHFQC split is already done at this point by the Python lib,
    // shall be moved here later when the split logic is implemented in Rust.
    let mut awg_devices = Vec::new();
    let mut auxiliary_devices = Vec::new();
    for device in devices {
        let device = device.borrow();
        if let Ok(device_kind) = DeviceKind::from_str(&device.kind) {
            let mut device_builder = AwgDevice::builder(
                builder.id_store.get_or_insert(&device.uid).into(),
                device.physical_device_uid.into(),
                device_kind,
            )
            .shfqc(device.is_shfqc)
            .options(DeviceOptions::new(device.options.clone()));
            if let Some(ref_clk) = &device.reference_clock {
                device_builder =
                    device_builder.reference_clock(ref_clk.parse().map_err(Error::new)?)
            }
            let device = device_builder.build();
            awg_devices.push(device);
        } else if let Ok(device_kind) = AuxiliaryDeviceKind::from_str(&device.kind) {
            // For auxiliary device, we ignore the rest of the fields. They are
            // processed separately in the Controller.
            let device = AuxiliaryDevice::new(
                builder.id_store.get_or_insert(&device.uid).into(),
                device_kind,
            );
            auxiliary_devices.push(device);
        } else {
            return Err(Error::new(format!("Unknown device: '{}'", device.kind)));
        }
    }

    let signals = signals
        .into_iter()
        .map(|s| {
            let signal = py_signal_to_signal(s.py(), &s.borrow(), &mut builder)?;
            Ok(signal)
        })
        .collect::<Result<_>>()?;

    let awg_mapping = awgs.into_iter().map(|awg| {
        let awg = awg.borrow();
        AwgCore::new(awg.uid, awg.number.clone().into())
    });

    // Post-pass: add derived sweep parameters that were registered during signal processing.
    // Signal processing (py_signal_to_signal) populates builder.driving_parameters; this pass
    // uses that map to augment sweep parameter lists — mirroring collect_derived_parameters in
    // py_conversion.rs.
    fix_derived_parameters(&mut builder.root, &builder.driving_parameters);

    let mut experiment = Experiment {
        root: builder.root,
        id_store: builder.id_store.into(),
        parameters: builder.parameters,
        pulses: builder.pulses,
        py_object_store: builder.py_object_store.into(),
    };
    let setup_properties = SetupProperties {
        signals,
        awg_devices,
        auxiliary_devices,
        awgs: awg_mapping.collect(),
    };

    let context = experiment_context_from_experiment(&experiment).map_err(|e| {
        let msg = create_error_message(e);
        Error::new(resolve_ids(&msg, &experiment.id_store))
    })?;

    let processed_setup = process_setup(setup_properties, desktop_setup, &experiment.id_store)
        .map_err(|e| {
            let msg = create_error_message(e);
            Error::new(resolve_ids(&msg, &experiment.id_store))
        })?;
    let device_setup = DeviceSetup::new(
        processed_setup
            .signals
            .into_iter()
            .map(|s| (s.uid, s))
            .collect(),
        processed_setup
            .devices
            .into_iter()
            .map(|d| (d.uid(), d))
            .collect(),
        processed_setup.awgs,
    )
    .map_err(Error::new)?;

    process_experiment(&mut experiment, &device_setup, &context).map_err(|e| {
        let msg = create_error_message(e);
        Error::new(resolve_ids(&msg, &experiment.id_store))
    })?;
    validate_experiment(&experiment, &device_setup).map_err(|e| {
        let msg = create_error_message(e);
        Error::new(resolve_ids(&msg, &experiment.id_store))
    })?;
    Ok(ExperimentPy {
        inner: experiment,
        device_setup: Arc::new(device_setup),
        context,
        delay_compensation: processed_setup.on_device_delays,
    })
}

#[pyfunction(name = "build_experiment")]
fn build_experiment_py(
    experiment: &Bound<'_, PyAny>,
    signals: Vec<Bound<'_, SignalPy>>,
    devices: Vec<Bound<'_, DevicePy>>,
    awgs: Vec<Bound<'_, AwgInfoPy>>,
    desktop_setup: bool,
) -> Result<ExperimentPy> {
    let (mut experiment, setup_properties) =
        experiment_py_to_experiment(experiment, signals, devices, awgs).map_err(|e| {
            // NOTE: The error message here does not resolve IDs, as the experiment's ID store is not yet built.
            // and `experiment_py_to_experiment` has access to the actual UIDs before interning.
            let msg = create_error_message(e);
            Error::new(msg)
        })?;

    let context = experiment_context_from_experiment(&experiment).map_err(|e| {
        let msg = create_error_message(e);
        Error::new(resolve_ids(&msg, &experiment.id_store))
    })?;

    let processed_setup = process_setup(setup_properties, desktop_setup, &experiment.id_store)
        .map_err(|e| {
            let msg = create_error_message(e);
            Error::new(resolve_ids(&msg, &experiment.id_store))
        })?;
    let device_setup = DeviceSetup::new(
        processed_setup
            .signals
            .into_iter()
            .map(|s| (s.uid, s))
            .collect(),
        processed_setup
            .devices
            .into_iter()
            .map(|d| (d.uid(), d))
            .collect(),
        processed_setup.awgs,
    )
    .map_err(Error::new)?;

    process_experiment(&mut experiment, &device_setup, &context).map_err(|e| {
        let msg = create_error_message(e);
        Error::new(resolve_ids(&msg, &experiment.id_store))
    })?;
    validate_experiment(&experiment, &device_setup).map_err(|e| {
        let msg = create_error_message(e);
        Error::new(resolve_ids(&msg, &experiment.id_store))
    })?;
    Ok(ExperimentPy {
        inner: experiment,
        device_setup: Arc::new(device_setup),
        context,
        delay_compensation: processed_setup.on_device_delays,
    })
}

#[pyclass(name = "ScheduleResult", frozen)]
struct ScheduleResult {
    /// Parameters used in the experiment
    #[pyo3(get)]
    used_parameters: HashSet<String>,
    #[pyo3(get)]
    experiment_ir: Py<ExperimentIrPy>,
}

#[pyfunction(name = "schedule_experiment")]
fn schedule_experiment_py(
    py: Python,
    experiment: &ExperimentPy,
    parameters: HashMap<String, f64>,
    chunking_info: Option<(usize, usize)>, // Current chunk index, total chunk count
) -> Result<ScheduleResult> {
    // The NT / RT boundaries should have been resolved at this point.
    // This means only 1 NT root.
    let chunking_info = if let Some((index, count)) = chunking_info {
        let count = NonZeroU32::new(count as u32)
            .ok_or_else(|| Error::new("Chunk count must be a positive integer".to_string()))?;
        Some(ChunkingInfo { index, count })
    } else {
        None
    };
    let context = &experiment.context;
    let device_setup = &experiment.device_setup;
    let experiment = &experiment.inner;
    let mut parameter_store = create_parameter_store(parameters, &experiment.id_store);
    let views = signal_views(device_setup);
    let feedback_calculator = QccsFeedbackCalculator::new(py, views.values().cloned())?;
    let result = schedule_experiment(
        &experiment.root,
        SchedulerContext {
            id_store: &experiment.id_store,
            parameters: experiment.parameters.clone(),
            signals: &views,
            handle_to_signal: &context.handle_to_signal,
        },
        &parameter_store,
        chunking_info,
        Some(&feedback_calculator),
    )
    .map_err(|e| {
        let msg = create_error_message(e);
        Error::new(resolve_ids(&msg, &experiment.id_store))
    })?;
    let ir = ExperimentIr {
        root: result.root,
        parameters: result.parameters.values().cloned().collect(),
        acquisition_type: context.acquisition_type,
        id_store: Arc::clone(&experiment.id_store),
        device_setup: Arc::clone(device_setup),
    };
    let ir_py = ExperimentIrPy {
        inner: ir,
        py_object_store: Arc::clone(&experiment.py_object_store),
        pulses: experiment.pulses.values().cloned().collect(),
    };
    let out = ScheduleResult {
        used_parameters: parameter_store
            .empty_queries()
            .iter()
            .map(|p| experiment.id_store.resolve(p.0).unwrap().to_string())
            .collect(),
        experiment_ir: ir_py.into_pyobject(py)?.into(),
    };
    Ok(out)
}

/// Generate a schedule (event list + metadata) from an IR tree.
///
/// This function is used by the Python compiler to generate the event list
/// for the Pulse Sheet Viewer (PSV).
///
/// # Arguments
/// * `ir_py` - The experiment IR from Python
/// * `expand_loops` - Whether to expand compressed loops (EXPAND_LOOPS_FOR_SCHEDULE flag)
/// * `max_events` - Maximum number of events to generate (MAX_EVENTS_TO_PUBLISH setting)
///
/// # Returns
/// A Python dict containing:
/// - event_list: List of scheduler events
/// - section_info: Section metadata with preorder map
/// - section_signals_with_children: Signal hierarchy per section
#[pyfunction(name = "generate_pulse_sheet_schedule")]
fn generate_pulse_sheet_schedule<'py>(
    py: Python<'py>,
    ir_py: &ExperimentIrPy,
    expand_loops: bool,
    max_events: usize,
) -> Result<Bound<'py, PyAny>> {
    let ir = &ir_py.inner;

    // Generate the schedule
    let schedule = PulseSheetSchedule::generate(
        &ir.root,
        ir.device_setup
            .signals()
            .map(|s| {
                (
                    ir.device_setup.device_by_uid(&s.device_uid).unwrap().kind(),
                    s.sampling_rate,
                )
            })
            .collect(),
        expand_loops,
        max_events,
        &ir.id_store,
    );

    // Convert to JSON then to Python dict
    let json_value = serde_json::to_value(&schedule)
        .map_err(|e| Error::new(format!("Failed to serialize schedule: {}", e)))?;

    // Convert JSON to Python object
    json_to_py(py, &json_value)
}

/// Helper function to convert serde_json::Value to Python objects
fn json_to_py<'py>(py: Python<'py>, value: &serde_json::Value) -> Result<Bound<'py, PyAny>> {
    use pyo3::IntoPyObject;
    use pyo3::types::{PyDict, PyList};
    use serde_json::Value;

    match value {
        Value::Null => Ok(py.None().into_bound(py)),
        Value::Bool(b) => {
            let py_bool = b
                .into_pyobject(py)
                .map_err(|e| Error::new(format!("Failed to convert bool: {}", e)))?;
            Ok(py_bool.to_owned().into_any())
        }
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                let py_int = i
                    .into_pyobject(py)
                    .map_err(|e| Error::new(format!("Failed to convert i64: {}", e)))?;
                Ok(py_int.to_owned().into_any())
            } else if let Some(u) = n.as_u64() {
                let py_int = u
                    .into_pyobject(py)
                    .map_err(|e| Error::new(format!("Failed to convert u64: {}", e)))?;
                Ok(py_int.to_owned().into_any())
            } else if let Some(f) = n.as_f64() {
                let py_float = f
                    .into_pyobject(py)
                    .map_err(|e| Error::new(format!("Failed to convert f64: {}", e)))?;
                Ok(py_float.to_owned().into_any())
            } else {
                Err(Error::new("Invalid JSON number"))
            }
        }
        Value::String(s) => {
            let py_str = s
                .into_pyobject(py)
                .map_err(|e| Error::new(format!("Failed to convert string: {}", e)))?;
            Ok(py_str.to_owned().into_any())
        }
        Value::Array(arr) => {
            let py_list = PyList::empty(py);
            for item in arr {
                py_list.append(json_to_py(py, item)?)?;
            }
            Ok(py_list.into_any())
        }
        Value::Object(obj) => {
            let py_dict = PyDict::new(py);
            for (key, val) in obj {
                py_dict.set_item(key, json_to_py(py, val)?)?;
            }
            Ok(py_dict.into_any())
        }
    }
}

/// Assert that two built experiments are semantically equivalent.
/// Resolves all NamedId fields to strings before comparing.
/// Panics with a descriptive message on mismatch.
#[pyfunction(name = "assert_experiment_equivalent")]
fn assert_experiment_equivalent_py(
    lhs: &Bound<'_, ExperimentPy>,
    rhs: &Bound<'_, ExperimentPy>,
) -> PyResult<()> {
    experiment_equivalence::assert_experiment_equivalent(&lhs.borrow().inner, &rhs.borrow().inner);
    Ok(())
}

#[pyfunction(name = "init_logging")]
fn init_logging_py(log_level: i64) -> PyResult<()> {
    laboneq_py_utils::logging::init_logging_py(log_level)
}

pub fn create_py_module<'py>(py: Python<'py>, name: &str) -> Result<Bound<'py, PyModule>> {
    use crate::py_signal::{
        BounceCompensationPy, ExponentialCompensationPy, FirCompensationPy, HighPassCompensationPy,
        OutputRoutePy, PrecompensationPy,
    };
    use py_awg::AwgInfoPy;

    let m = PyModule::new(py, name)?;
    m.add_function(wrap_pyfunction!(schedule_experiment_py, &m)?)?;
    m.add_function(wrap_pyfunction!(build_experiment_py, &m)?)?;
    m.add_function(wrap_pyfunction!(serialize_experiment_py, &m)?)?;
    m.add_function(wrap_pyfunction!(build_experiment_capnp_py, &m)?)?;
    m.add_function(wrap_pyfunction!(generate_pulse_sheet_schedule, &m)?)?;
    m.add_function(wrap_pyfunction!(assert_experiment_equivalent_py, &m)?)?;
    m.add_function(wrap_pyfunction!(init_logging_py, &m)?)?;
    // Intermediate migration objects, shall be removed later
    m.add_class::<SignalPy>()?;
    m.add_class::<DevicePy>()?;
    m.add_class::<AwgInfoPy>()?;
    m.add_class::<OscillatorPy>()?;
    m.add_class::<SweepParameterPy>()?;
    m.add_class::<AmplifierPumpPy>()?;
    m.add_class::<PrecompensationPy>()?;
    m.add_class::<HighPassCompensationPy>()?;
    m.add_class::<FirCompensationPy>()?;
    m.add_class::<ExponentialCompensationPy>()?;
    m.add_class::<BounceCompensationPy>()?;
    m.add_class::<OutputRoutePy>()?;
    Ok(m)
}

// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for `laboneq-scheduler`.
use std::collections::HashMap;
use std::collections::HashSet;
use std::num::NonZeroU32;
use std::sync::Arc;

use crate::error::{Error, Result, create_error_message};
use crate::scheduler::experiment::DeviceSetup;
use crate::scheduler::experiment::Experiment;
use crate::scheduler::experiment_context::ExperimentContext;
use crate::scheduler::experiment_context::experiment_context_from_experiment;
use crate::scheduler::experiment_processor::process_experiment;
use crate::scheduler::experiment_validation::validate_experiment;
use crate::scheduler::parameter_store::create_parameter_store;
use crate::scheduler::py_conversion::ExperimentBuilder;
use crate::scheduler::py_device::DevicePy;
use crate::scheduler::py_device::py_device_to_device;
use crate::scheduler::py_signal::AmplifierPumpPy;
use crate::scheduler::py_signal::SweepParameterPy;
use crate::scheduler::py_signal::{OscillatorPy, py_signal_to_signal};
use crate::scheduler::qccs_feedback_calculator::QccsFeedbackCalculator;
use laboneq_common::named_id::{NamedIdStore, resolve_ids};
use laboneq_ir::ExperimentIr;
use laboneq_ir::schedule::Schedule;
use laboneq_py_utils::experiment_ir::ExperimentIrPy;
use laboneq_scheduler::{ChunkingInfo, ExperimentContext as SchedulerContext, schedule_experiment};
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

#[cfg(test)]
mod test_py;

mod py_conversion;
use py_conversion::build_experiment;
mod py_signal;
use py_signal::SignalPy;
pub(crate) mod experiment;
mod experiment_context;
mod experiment_processor;
mod experiment_validation;
mod parameter_store;
mod py_device;
mod qccs_feedback_calculator;
mod signal_view;
use signal_view::signal_views;

#[pyclass(name = "Experiment", frozen)]
struct ExperimentPy {
    pub inner: Experiment,
    pub device_setup: DeviceSetup,
    pub context: ExperimentContext,
}

fn build_device_setup(
    builder: &mut ExperimentBuilder,
    signals: Vec<Bound<'_, SignalPy>>,
    devices: Vec<Bound<'_, DevicePy>>,
) -> Result<DeviceSetup> {
    let devices = devices
        .into_iter()
        .map(|d| {
            let device = py_device_to_device(&d.borrow(), &mut builder.id_store)?;
            Ok((device.uid, device))
        })
        .collect::<Result<_>>()?;
    let signals = signals
        .into_iter()
        .map(|s| {
            let signal = py_signal_to_signal(s.py(), &s.borrow(), builder)?;
            Ok((signal.uid, signal))
        })
        .collect::<Result<_>>()?;
    DeviceSetup::new(signals, devices)
}

pub(crate) fn experiment_py_to_experiment(
    experiment: &Bound<'_, PyAny>,
    signals: Vec<Bound<'_, SignalPy>>,
    devices: Vec<Bound<'_, DevicePy>>,
) -> Result<(Experiment, DeviceSetup)> {
    let mut builder = ExperimentBuilder::new(experiment.py());
    let device_setup = build_device_setup(&mut builder, signals, devices)?;
    build_experiment(experiment, &mut builder)?;
    Ok((
        Experiment {
            root: builder.root,
            id_store: builder.id_store.into(),
            parameters: builder.parameters,
            pulses: builder.pulses,
            py_object_store: builder.py_object_store.into(),
        },
        device_setup,
    ))
}

#[pyfunction(name = "build_experiment")]
fn build_experiment_py(
    experiment: &Bound<'_, PyAny>,
    signals: Vec<Bound<'_, SignalPy>>,
    devices: Vec<Bound<'_, DevicePy>>,
) -> Result<ExperimentPy> {
    let (mut experiment, device_setup) = experiment_py_to_experiment(experiment, signals, devices)
        .map_err(|e| {
            // NOTE: The error message here does not resolve IDs, as the experiment's ID store is not yet built.
            // and `experiment_py_to_experiment` has access to the actual UIDs before interning.
            let msg = create_error_message(e);
            Error::new(msg)
        })?;
    let context = experiment_context_from_experiment(&experiment).map_err(|e| {
        let msg = create_error_message(e);
        Error::new(resolve_ids(&msg, &experiment.id_store))
    })?;
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
        device_setup,
        context,
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
///
/// Note: sampling_rates should be added separately by Python (depends on SamplingRateTracker).
#[pyfunction(name = "generate_schedule")]
fn generate_schedule_py<'py>(
    py: Python<'py>,
    ir_py: &ExperimentIrPy,
    expand_loops: bool,
    max_events: usize,
) -> Result<Bound<'py, PyAny>> {
    let ir = &ir_py.inner;

    // Generate the schedule
    let schedule = Schedule::generate(&ir.root, expand_loops, max_events, &ir.id_store);

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

pub(crate) fn create_py_module<'py>(py: Python<'py>, name: &str) -> Result<Bound<'py, PyModule>> {
    let m = PyModule::new(py, name)?;
    m.add_function(wrap_pyfunction!(schedule_experiment_py, &m)?)?;
    m.add_function(wrap_pyfunction!(build_experiment_py, &m)?)?;
    m.add_function(wrap_pyfunction!(generate_schedule_py, &m)?)?;
    // Intermediate migration objects, shall be removed later
    m.add_class::<SignalPy>()?;
    m.add_class::<DevicePy>()?;
    m.add_class::<OscillatorPy>()?;
    m.add_class::<SweepParameterPy>()?;
    m.add_class::<AmplifierPumpPy>()?;
    Ok(m)
}

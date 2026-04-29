// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for laboneq compiler.

pub mod error;

use laboneq_dsl::device_setup::DeviceSignal;
use pyo3::types::PyDict;
use std::collections::HashMap;
use std::collections::HashSet;
use std::num::NonZeroU32;
use std::sync::Arc;

use laboneq_opentelemetry_python::attach_otel_context;
use laboneq_tracing::tracing_is_enabled;
use laboneq_tracing::with_tracing;

use crate::compiler_backend::CompilerBackend;
use crate::compiler_backend::ExperimentView;
use crate::compiler_backend::PreprocessedBackendData;
use crate::error::{Error, Result, create_error_message};
use crate::experiment::Experiment;
use crate::experiment_context::ExperimentContext;
use crate::experiment_context::experiment_context_from_experiment;
use crate::experiment_processor::process_experiment;
use crate::experiment_validation::validate_experiment;
use crate::parameter_store::create_parameter_store;
use crate::py_device_setup_capnp::DeviceSetupCapnpBuilderPy;
use crate::py_experiment::ExperimentPy;
use crate::py_experiment_ir::ExperimentIrPy;
use crate::py_signal::AmplifierPumpPy;
use crate::qccs_feedback_calculator::QccsFeedbackCalculator;
use crate::setup_processor::DelayRegistry;
use crate::setup_processor::process_setup;

use laboneq_common::compiler_settings::CompilerSettings;
use laboneq_common::named_id::{NamedIdStore, resolve_ids};
use laboneq_dsl::device_setup::AuxiliaryDevice;
use laboneq_dsl::types::ExternalParameterUid;
use laboneq_ir::ExperimentIr;
use laboneq_ir::pulse_sheet_schedule::PulseSheetSchedule;
use laboneq_ir::system::AwgDevice;
use laboneq_ir::system::DeviceSetup;
use laboneq_py_utils::py_object_interner::PyObjectInterner;
use laboneq_scheduler::{ChunkingInfo, ExperimentContext as SchedulerContext, schedule_experiment};

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

#[cfg(test)]
mod test_py;

pub(crate) mod capnp_deserializer;
pub(crate) mod capnp_serializer;
pub(crate) mod experiment;
mod experiment_context;
mod experiment_processor;
mod experiment_validation;
mod parameter_store;
mod py_conversion;
mod py_helpers;
mod py_signal;
mod qccs_feedback_calculator;
mod signal_view;
use signal_view::signal_views;
pub mod compiler_backend;
mod py_device_setup_capnp;
pub mod py_experiment;
pub mod py_experiment_ir;
mod setup_processor;

pub(crate) struct SetupProperties {
    signals: Vec<DeviceSignal>,
    awg_devices: Vec<AwgDevice>,
    auxiliary_devices: Vec<AuxiliaryDevice>,
}

/// Serialize a Python experiment object to Cap'n Proto bytes.
#[pyfunction(name = "serialize_experiment", signature = (experiment, device_setup, packed=false))]
fn serialize_experiment_py(
    experiment: &Bound<'_, PyAny>,
    device_setup: Bound<'_, DeviceSetupCapnpBuilderPy>,
    packed: bool,
) -> Result<Vec<u8>> {
    capnp_serializer::serialize_experiment(experiment, device_setup, packed)
}

/// Build an experiment from Cap'n Proto bytes plus device/signal configuration.
pub fn build_experiment_with_backend_capnp_py<B: CompilerBackend>(
    py: Python<'_>,
    capnp_data: &[u8],
    desktop_setup: bool,
    packed: bool,
    compiler_settings: Option<Bound<'_, PyDict>>,
    backend: B,
) -> PyResult<ExperimentPy>
where
    <B as CompilerBackend>::Output: Send + Sync + 'static,
{
    let compiler_settings = compiler_settings
        .as_ref()
        .map(|dict| py_helpers::compiler_settings_from_py_dict(dict))
        .transpose()?
        .unwrap_or_default();

    let processed = build_experiment_capnp(py, capnp_data, desktop_setup, packed, backend)?;

    let exp_py = ExperimentPy {
        inner: processed.inner,
        device_setup: processed.device_setup,
        context: processed.context,
        delay_compensation: processed.delay_compensation,
        compiler_settings: compiler_settings.clone(),
        backend_data: Arc::new(processed.backend_data),
    };
    Ok(exp_py)
}

pub struct ProcessedExperiment<D> {
    inner: Experiment,
    // NOTE: The usage of Arc here is to allow sharing the id_store across Python bindings
    // Remove when Python bindings are no longer needed
    device_setup: Arc<DeviceSetup>,
    context: ExperimentContext,
    /// Delay compensation for signals on devices.
    delay_compensation: DelayRegistry,
    backend_data: D,
}

/// Build an experiment from Cap'n Proto bytes plus device/signal configuration.
fn build_experiment_capnp<B: CompilerBackend>(
    py: Python<'_>,
    capnp_data: &[u8],
    desktop_setup: bool,
    packed: bool,
    backend: B,
) -> Result<ProcessedExperiment<impl PreprocessedBackendData>> {
    // 1. Deserialize the experiment tree from Cap'n Proto.
    let mut deserialized = capnp_deserializer::deserialize_experiment(py, capnp_data, packed)?;

    // Register the PyObjects
    let mut py_object_store = PyObjectInterner::<ExternalParameterUid>::new();
    for (uid, value) in deserialized.external_parameter_values {
        py_object_store.insert(uid, value);
    }

    let backend_processed = backend.preprocess_experiment(ExperimentView::new(
        &deserialized.root,
        &mut deserialized.id_store,
        &deserialized.parameters,
        &deserialized.pulses,
        &deserialized.awg_devices,
        &deserialized.auxiliary_devices,
        &deserialized.signals,
    ))?;
    deserialized
        .signals
        .extend(backend_processed.additional_signals().iter().cloned());

    let mut experiment = Experiment {
        root: deserialized.root,
        id_store: deserialized.id_store.into(),
        parameters: deserialized.parameters,
        pulses: deserialized.pulses,
        py_object_store: py_object_store.into(),
    };
    let setup_properties = SetupProperties {
        signals: deserialized.signals,
        awg_devices: deserialized.awg_devices,
        auxiliary_devices: deserialized.auxiliary_devices,
    };

    let context = experiment_context_from_experiment(&experiment).map_err(|e| {
        let msg = create_error_message(e);
        Error::new(resolve_ids(&msg, &experiment.id_store))
    })?;

    let processed_setup = process_setup(
        setup_properties,
        &backend_processed,
        desktop_setup,
        &experiment.id_store,
        &context,
    )
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
        processed_setup.devices,
        processed_setup.auxiliary_devices,
        desktop_setup,
    )
    .map_err(Error::new)?;

    process_experiment(&mut experiment, &device_setup).map_err(|e| {
        let msg = create_error_message(e);
        Error::new(resolve_ids(&msg, &experiment.id_store))
    })?;
    validate_experiment(&experiment, &device_setup).map_err(|e| {
        let msg = create_error_message(e);
        Error::new(resolve_ids(&msg, &experiment.id_store))
    })?;

    let res = ProcessedExperiment {
        inner: experiment,
        device_setup: Arc::new(device_setup),
        context,
        delay_compensation: processed_setup.on_device_delays,
        backend_data: backend_processed,
    };
    Ok(res)
}

#[pyclass(name = "ScheduleResult", frozen)]
struct ScheduleResult {
    /// Parameters used in the experiment
    #[pyo3(get)]
    used_parameters: HashSet<String>,
    #[pyo3(get)]
    experiment_ir: Py<ExperimentIrPy>,
    #[pyo3(get)]
    pulse_sheet_schedule: Option<Py<PyAny>>,
}

#[pyfunction(name = "schedule_experiment")]
fn schedule_experiment_py(
    py: Python,
    experiment: &ExperimentPy,
    parameters: HashMap<String, f64>,
    chunking_info: Option<(usize, usize)>, // Current chunk index, total chunk count
) -> Result<ScheduleResult> {
    let _context_guard = tracing_is_enabled()
        .then(|| attach_otel_context(py))
        .transpose()?;
    with_tracing(|| schedule_experiment_py_impl(py, experiment, parameters, chunking_info))
}

fn schedule_experiment_py_impl(
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
    let compiler_settings = &experiment.compiler_settings;
    let context = &experiment.context;
    let device_setup = &experiment.device_setup;
    let inner_experiment = &experiment.inner;
    let mut parameter_store = create_parameter_store(parameters, &inner_experiment.id_store);
    let views = signal_views(device_setup);
    let feedback_calculator = QccsFeedbackCalculator::new(py, views.values().cloned())?;
    let result = schedule_experiment(
        &inner_experiment.root,
        SchedulerContext {
            id_store: &inner_experiment.id_store,
            parameters: inner_experiment.parameters.clone(),
            signals: &views,
            handle_to_signal: context.handle_to_signal(),
        },
        &parameter_store,
        chunking_info,
        Some(&feedback_calculator),
    )
    .map_err(|e| {
        let msg = create_error_message(e);
        Error::new(resolve_ids(&msg, &inner_experiment.id_store))
    })?;

    let ir = ExperimentIr {
        root: result.root,
        parameters: result.parameters.values().cloned().collect(),
        pulses: inner_experiment.pulses.values().cloned().collect(),
        acquisition_type: *context.acquisition_type(),
        id_store: Arc::clone(&inner_experiment.id_store),
        device_setup: Arc::clone(device_setup),
    };

    let pulse_sheet_schedule = prepare_schedule(&ir, compiler_settings);

    let ir_py = ExperimentIrPy {
        inner: ir,
        py_object_store: Arc::clone(&experiment.inner.py_object_store),
        compiler_settings: experiment.compiler_settings.clone(),
        backend_data: Arc::clone(&experiment.backend_data),
    };

    let out = ScheduleResult {
        used_parameters: parameter_store
            .empty_queries()
            .iter()
            .map(|p| inner_experiment.id_store.resolve(p.0).unwrap().to_string())
            .collect(),
        experiment_ir: ir_py.into_pyobject(py)?.into(),
        pulse_sheet_schedule: pulse_sheet_schedule
            .as_ref()
            .map(|s| schedule_to_py(py, s))
            .transpose()?
            .map(|s| s.unbind()),
    };
    Ok(out)
}

/// Generate a schedule (event list + metadata) from an IR tree.
///
/// This function is used by the Python compiler to generate the event list
/// for the Pulse Sheet Viewer (PSV).
fn prepare_schedule(
    ir: &ExperimentIr,
    compiler_settings: &CompilerSettings,
) -> Option<PulseSheetSchedule> {
    if !compiler_settings.output_extras {
        return None;
    }
    let out = PulseSheetSchedule::generate(
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
        compiler_settings.expand_loops_for_schedule,
        compiler_settings.max_events_to_publish,
        &ir.id_store,
    );
    Some(out)
}

/// Convert a [`PulseSheetSchedule`] to a Python dict for consumption in Python.
///
/// # Returns
/// A Python dict containing:
/// - event_list: List of scheduler events
/// - event_list_truncated: Whether event generation hit the max_events limit
/// - section_info: Section metadata with preorder map
/// - section_signals_with_children: Signal hierarchy per section
fn schedule_to_py<'py>(
    py: Python<'py>,
    schedule: &PulseSheetSchedule,
) -> Result<Bound<'py, PyAny>> {
    // Convert to JSON then to Python dict
    let json_value = serde_json::to_value(schedule)
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

#[pyfunction(name = "init_logging")]
fn init_logging_py(log_level: i64) -> PyResult<()> {
    laboneq_py_utils::logging::init_logging_py(log_level)
}

pub fn create_py_module<'py>(py: Python<'py>, name: &str) -> Result<Bound<'py, PyModule>> {
    use crate::py_signal::{
        BounceCompensationPy, ExponentialCompensationPy, FirCompensationPy, HighPassCompensationPy,
        PrecompensationPy,
    };
    use laboneq_opentelemetry_python::span_buffer::SpanBufferPy;
    use py_device_setup_capnp::DeviceSetupCapnpBuilderPy;

    let m = PyModule::new(py, name)?;
    m.add_function(wrap_pyfunction!(schedule_experiment_py, &m)?)?;
    m.add_function(wrap_pyfunction!(serialize_experiment_py, &m)?)?;
    m.add_function(wrap_pyfunction!(init_logging_py, &m)?)?;
    m.add_class::<SpanBufferPy>()?;
    // Intermediate migration objects, shall be removed later
    m.add_class::<AmplifierPumpPy>()?;
    m.add_class::<PrecompensationPy>()?;
    m.add_class::<HighPassCompensationPy>()?;
    m.add_class::<FirCompensationPy>()?;
    m.add_class::<ExponentialCompensationPy>()?;
    m.add_class::<BounceCompensationPy>()?;
    m.add_class::<DeviceSetupCapnpBuilderPy>()?;
    Ok(m)
}

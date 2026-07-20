// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for laboneq compiler.

pub mod error;

use laboneq_dsl::types::NumericLiteral;
use laboneq_dsl::types::ParameterUid;
use laboneq_error::LabOneQError;
use pyo3::types::PyDict;
use std::collections::HashMap;
use std::collections::HashSet;
use std::sync::Arc;

use crate::capnp_py_types::DeviceSetupCapnpPy;
use crate::capnp_py_types::ExperimentCapnpPy;
use crate::compiler::run_compilation;
use crate::compiler_backend::CompilerBackend;
use crate::compiler_backend::ExperimentView;
use crate::error::{Error, Result as CompilerResult, create_error_message};
use crate::experiment::Experiment;
use crate::experiment_context::ExperimentContext;
use crate::experiment_context::experiment_context_from_experiment;
use crate::experiment_processor::process_experiment;
use crate::experiment_validation::validate_experiment;
use crate::py_experiment::ExperimentPy;
use crate::result_shape::ResultShapes;
use crate::result_shape::extract_result_shapes;
use crate::rt_compiler::RealTimeCompilerInput;
use crate::rt_compiler::compile_realtime;
use crate::setup_processor::DelayRegistry;
use crate::setup_processor::SetupProperties;
use crate::setup_processor::process_setup;
use laboneq_opentelemetry_python::attach_otel_context;
use laboneq_tracing::tracing_is_enabled;
use laboneq_tracing::with_tracing;

use laboneq_common::compiler_settings::CompilerSettings;
use laboneq_common::named_id::{NamedIdStore, resolve_ids};
use laboneq_dsl::types::ExternalParameterUid;
use laboneq_ir::ExperimentIr;
use laboneq_ir::pulse_sheet_schedule::PulseSheetSchedule;
use laboneq_ir::system::DeviceSetup;
use laboneq_py_utils::py_object_interner::PyObjectInterner;

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use tracing::instrument;

#[cfg(test)]
mod test_py;

pub(crate) mod capnp_deserializer;
mod capnp_py_types;
pub(crate) mod capnp_serializer;
mod chunking_mode;
mod compiler;
pub mod compiler_backend;
mod execution;
pub(crate) mod experiment;
mod experiment_context;
mod experiment_processor;
mod experiment_validation;
mod py_conversion;
mod py_execution;
pub mod py_experiment;
mod py_helpers;
mod py_result_shape;
mod qccs_feedback_calculator;
mod result_shape;
mod rt_compiler;
mod setup_processor;
mod signal_view;

/// Serialize a Python experiment object to Cap'n Proto bytes.
#[pyfunction(name = "serialize_experiment", signature = (experiment, device_setup, packed=false))]
fn serialize_experiment_py(
    py: Python,
    experiment: ExperimentCapnpPy,
    device_setup: DeviceSetupCapnpPy,
    packed: bool,
) -> CompilerResult<Vec<u8>> {
    let _context_guard = tracing_is_enabled()
        .then(|| attach_otel_context(py))
        .transpose()?;
    with_tracing(|| capnp_serializer::serialize_experiment(py, experiment, device_setup, packed))
}

/// Build an experiment from Cap'n Proto bytes plus device/signal configuration.
pub fn compile_experiment<'py, B>(
    py: Python<'py>,
    capnp_data: &[u8],
    packed: bool,
    compiler_settings: Option<Bound<'_, PyDict>>,
    backend: B,
) -> PyResult<Bound<'py, PyAny>>
where
    B: CompilerBackend + Send + Sync + 'static,
    B::Output: Send + Sync + 'static,
    B::CodeGenArtifact: Send + Sync + 'static,
{
    let _context_guard = tracing_is_enabled()
        .then(|| attach_otel_context(py))
        .transpose()?;

    let compiler_settings = compiler_settings
        .as_ref()
        .map(|dict| py_helpers::compiler_settings_from_py_dict(dict))
        .transpose()?
        .unwrap_or_default();

    with_tracing(|| {
        let processed = build_experiment_capnp(py, capnp_data, packed, &backend)?;
        let id_store = Arc::clone(&processed.inner.id_store);
        let scheduled_experiment = run_compilation(py, backend, processed, compiler_settings)
            .map_err(|e| e.to_pyerr(|s| resolve_ids(&s, &id_store)))?;
        Ok(scheduled_experiment)
    })
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
    result_shapes: ResultShapes,
    device_setup_fingerprint: String,
}

/// Build an experiment from Cap'n Proto bytes plus device/signal configuration.
#[instrument(
    level = "debug",
    name = "laboneq.compiler.build_experiment_capnp",
    skip_all
)]
fn build_experiment_capnp<B: CompilerBackend>(
    py: Python<'_>,
    capnp_data: &[u8],
    packed: bool,
    backend: &B,
) -> Result<ProcessedExperiment<B::Output>, LabOneQError> {
    // 1. Deserialize the experiment tree from Cap'n Proto.
    let mut deserialized = capnp_deserializer::deserialize_experiment(py, capnp_data, packed)?;

    // Register the PyObjects
    let mut py_object_store = PyObjectInterner::<ExternalParameterUid>::new();
    for (uid, value) in deserialized.external_parameter_values {
        py_object_store.insert(uid, value);
    }
    let backend_processed = backend
        .preprocess_experiment(ExperimentView::new(
            &deserialized.root,
            &mut deserialized.id_store,
            deserialized
                .parameters
                .iter()
                .map(|(k, v)| (*k, v.inner_values().as_ref()))
                .collect(),
            &deserialized.pulses,
            deserialized.experiment_signals.clone(),
            deserialized.setup_description,
        ))
        .map_err(|e| {
            let msg = create_error_message(e);
            Error::new(resolve_ids(&msg, &deserialized.id_store))
        })?;

    let mut experiment = Experiment {
        root: deserialized.root,
        id_store: deserialized.id_store.into(),
        parameters: deserialized.parameters,
        pulses: deserialized.pulses,
        py_object_store: py_object_store.into(),
    };

    let context = experiment_context_from_experiment(&experiment).map_err(|e| {
        let msg = create_error_message(e);
        Error::new(resolve_ids(&msg, &experiment.id_store))
    })?;

    let processed_setup = process_setup(
        SetupProperties {
            signals: backend_processed.device_signals,
            awg_devices: backend_processed.awg_devices,
        },
        &backend_processed.backend_data,
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
    let result_shapes = extract_result_shapes(
        &experiment.root,
        experiment.parameters.values(),
        Arc::get_mut(&mut experiment.id_store).expect("Expected no additional ID stores"),
    )
    .map_err(|e| {
        let msg = create_error_message(e);
        Error::new(resolve_ids(&msg, &experiment.id_store))
    })?;

    let res = ProcessedExperiment {
        inner: experiment,
        device_setup: Arc::new(device_setup),
        context,
        delay_compensation: processed_setup.on_device_delays,
        backend_data: backend_processed.backend_data,
        result_shapes,
        device_setup_fingerprint: backend_processed.device_setup_fingerprint,
    };
    Ok(res)
}

#[pyclass(name = "RealTimeCompilerOutput", frozen)]
struct RealTimeCompilerOutputPy {
    /// Parameters used in the experiment
    #[pyo3(get)]
    used_parameters: HashSet<String>,
    #[pyo3(get)]
    pulse_sheet_schedule: Option<Py<PyAny>>,
    codegen_output: Py<PyAny>,
}

#[pymethods]
impl RealTimeCompilerOutputPy {
    fn codegen_output<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Ok(self.codegen_output.bind(py).into())
    }
}

#[pyfunction(name = "compile_realtime")]
fn compile_realtime_py(
    py: Python,
    experiment: &ExperimentPy,
    parameters: HashMap<String, f64>,
    chunking_info: Option<(usize, usize)>, // Current chunk index, total chunk count
) -> PyResult<RealTimeCompilerOutputPy> {
    let rt_input = RealTimeCompilerInput {
        experiment: &experiment.inner,
        parameters: parameters
            .into_iter()
            .map(|(k, v)| {
                let uid = experiment.inner.id_store.get(&k).ok_or_else(|| {
                    laboneq_error::laboneq_error!("Parameter {k} not found in experiment ID store")
                })?;
                Ok((ParameterUid(uid), NumericLiteral::Float(v)))
            })
            .collect::<Result<HashMap<ParameterUid, NumericLiteral>, LabOneQError>>()?,
        chunking_info,
        context: &experiment.context,
        backend: experiment.backend.as_ref(),
        backend_data: experiment.backend_data.as_ref(),
        device_setup: &experiment.device_setup,
        compiler_settings: &experiment.compiler_settings,
    };

    let result = compile_realtime(rt_input)?;

    let result_py = RealTimeCompilerOutputPy {
        used_parameters: result
            .used_parameters
            .iter()
            .map(|p| experiment.inner.id_store.resolve(p.0).unwrap().to_string())
            .collect(),
        pulse_sheet_schedule: result
            .pulse_sheet_schedule
            .as_ref()
            .map(|s| schedule_to_py(py, s))
            .transpose()
            .map_err(|e| laboneq_error::laboneq_error!("{e}"))?
            .map(|s| s.unbind()),
        codegen_output: result.codegen_output.to_python(py)?.into(),
    };
    Ok(result_py)
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
        ir.id_store,
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
) -> CompilerResult<Bound<'py, PyAny>> {
    // Convert to JSON then to Python dict
    let json_value = serde_json::to_value(schedule)
        .map_err(|e| Error::new(format!("Failed to serialize schedule: {}", e)))?;
    // Convert JSON to Python object
    json_to_py(py, &json_value)
}

/// Helper function to convert serde_json::Value to Python objects
fn json_to_py<'py>(
    py: Python<'py>,
    value: &serde_json::Value,
) -> CompilerResult<Bound<'py, PyAny>> {
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

pub fn create_py_module<'py>(py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyModule>> {
    use laboneq_opentelemetry_python::span_buffer::SpanBufferPy;

    let m = PyModule::new(py, name)?;
    m.add_function(wrap_pyfunction!(compile_realtime_py, &m)?)?;
    m.add_function(wrap_pyfunction!(serialize_experiment_py, &m)?)?;
    m.add_function(wrap_pyfunction!(init_logging_py, &m)?)?;
    m.add_class::<SpanBufferPy>()?;
    Ok(m)
}

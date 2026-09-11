// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::{PyTypeInfo, intern};

use laboneq_common::compiler_settings::CompilerSettings;
use laboneq_common::named_id::resolve_ids;
use laboneq_dsl::types::{HandleUid, SignalUid};
use laboneq_error::{
    LabOneQError, PyErrorWithContext, ResourceLimitationError, WithContext, bail_resource_usage,
};

use crate::capnp_compiled_experiment;
use crate::chunking_mode::{ChunkingMode, collect_chunking_mode};
use crate::compiled_experiment::{
    CompiledExperiment, ExecutionTiming, Metadata, RealTimeProperties, ResultShapes,
};
use crate::compiler_backend::{CombinedOutput, CompilerBackend};
use crate::error::ResolveErr;
use crate::execution::{Execution, create_execution};
use crate::preprocessor::{PreprocessedExperiment, preprocess_experiment};
use crate::py_execution::create_py_execution;
use crate::py_experiment::ExperimentPy;
use crate::py_pulse_sheet_schedule::PulseSheetSchedulePy;
use crate::real_time_loop::real_time_loop_properties;

/// Compile an experiment from Cap'n Proto bytes into serialized `CompiledExperiment` bytes.
///
/// `packed` describes `capnp_data` only; the returned bytes are always unpacked.
pub(crate) fn compile_experiment<B: CompilerBackend>(
    py: Python<'_>,
    capnp_data: &[u8],
    packed: bool,
    compiler_settings: CompilerSettings,
    backend: B,
) -> PyResult<Vec<u8>> {
    let processed = preprocess_experiment(py, capnp_data, packed, &backend)?;
    let id_store = Arc::clone(&processed.inner.id_store);

    let compiled_experiment = run_compilation(py, backend, processed, compiler_settings)
        .map_err(|e| e.to_pyerr(|s| resolve_ids(&s, &id_store)))?;

    let serialized_payload =
        capnp_compiled_experiment::serialize_compiled_experiment(&compiled_experiment)
            .resolve_err(&id_store)?;

    Ok(serialized_payload)
}

fn run_compilation<B: CompilerBackend>(
    py: Python<'_>,
    backend: B,
    processed: PreprocessedExperiment<B::Output, B::EmulatedHqcsProgram>,
    compiler_settings: CompilerSettings,
) -> Result<CompiledExperiment<B::CompilerArtifact>, LabOneQError> {
    let execution = create_execution(&processed.inner)?;
    let chunking_mode = collect_chunking_mode(&processed.inner)?;

    let compiled_experiment = compile_whole_or_with_chunks(
        py,
        backend,
        processed,
        compiler_settings,
        execution,
        chunking_mode,
    )?;

    Ok(compiled_experiment)
}

const RESOURCE_LIMIT_MSG: &str = "Compilation error - resource limitation exceeded.\n\
    To circumvent this, try one or more of the following:\n\
    - Double check the integrity of your experiment (look for unexpectedly long pulses, large number of sweep steps, etc.)\n\
    - Use length of pulses that are multiple of 8 ns to improve memory effficiency\n\
    - Reduce the number of sweep steps\n\
    - Reduce the number of variations in the pulses that are being played\n\
    - Enable chunking for a sweep\n\
    - If chunking is already enabled, increase the chunk count or switch to automatic chunking";

const AUTO_CHUNK_EXHAUSTED_MSG: &str = "Automatic chunking was not able to find a chunk count to circumvent resource limitations.\n\
    This means that one iteration of a sweep is too large and cannot be executed.\n\
    To circumvent this, try one or more of the following:\n\
    - Use length of pulses that are multiple of 8 ns to improve memory effficiency\n\
    - Chunking another sweep (e.g. in case of nested sweeps, enable chunking for the inner one)\n\
    - Find ways suitable for your use case to reduce the size of the program in one iteration";

fn compile_whole_or_with_chunks<B: CompilerBackend>(
    py: Python<'_>,
    backend: B,
    processed: PreprocessedExperiment<B::Output, B::EmulatedHqcsProgram>,
    compiler_settings: CompilerSettings,
    execution: Execution,
    chunking_mode: Option<ChunkingMode>,
) -> Result<CompiledExperiment<B::CompilerArtifact>, LabOneQError> {
    let backend_name = backend.name();

    let backend_typed = backend.clone();
    let rt_properties = real_time_loop_properties(&processed.inner.root)?;
    let id_store = Arc::clone(&processed.inner.id_store);
    let py_object_store = Arc::clone(&processed.inner.py_object_store);
    let sweep_parameters = processed.inner.parameters.clone();
    let coprocessors = processed.coprocessors;

    let raw_acquisitions: Vec<(SignalUid, HandleUid)> =
        processed.result_shapes.raw_acquisitions().collect();

    // Prepare the Python objects
    let exp_py = ExperimentPy {
        inner: processed.inner,
        signals: processed.signals,
        compiler_settings: compiler_settings.clone(),
        backend_data: Arc::new(processed.backend_data),
        backend: Arc::new(backend),
    };

    let execution_py = create_py_execution(
        py,
        &execution,
        &sweep_parameters,
        &id_store,
        &py_object_store,
    )?
    .into_pyobject(py)
    .map_err(LabOneQError::from_err)?;

    let compiler_settings = create_compiler_settings_py(py, &exp_py.compiler_settings)?;
    let bound_exp_py = exp_py.into_pyobject(py)?;

    // Run the compilation
    let (compiled_output, chunk_count) = match chunking_mode {
        None => call_compile(
            py,
            &bound_exp_py,
            &execution_py,
            None,
            backend_name,
            &compiler_settings,
        )
        .map(|result| (result, None))
        .map_err(|e| wrap_resource_limitation_error(e, RESOURCE_LIMIT_MSG)),
        Some(ChunkingMode::Manual { chunk_count }) => {
            let chunk_count = sanitize_chunk_count(chunk_count.get());
            call_compile(
                py,
                &bound_exp_py,
                &execution_py,
                chunk_count,
                backend_name,
                &compiler_settings,
            )
            .map(|result| (result, chunk_count))
            .map_err(|e| wrap_resource_limitation_error(e, RESOURCE_LIMIT_MSG))
        }
        Some(ChunkingMode::Auto(mut auto_chunking)) => loop {
            let raw_chunk_count = auto_chunking.initial_chunk_count.get();
            let chunk_count = sanitize_chunk_count(raw_chunk_count);
            laboneq_log::debug!("Attempting to compile with {} chunks", raw_chunk_count);
            let result = call_compile(
                py,
                &bound_exp_py,
                &execution_py,
                chunk_count,
                backend_name,
                &compiler_settings,
            );

            match result {
                Ok(result) => {
                    laboneq_log::info!(
                        "Auto-chunked sweep divided into {} chunks",
                        raw_chunk_count
                    );
                    break Ok((result, chunk_count));
                }
                Err(LabOneQError::ResourceExhaustion(e)) => {
                    laboneq_log::debug!(
                        "The attempt with {} chunks failed with resource exhaustion: {}",
                        raw_chunk_count,
                        e
                    );
                    if raw_chunk_count == auto_chunking.iterations.get() {
                        return Err(wrap_resource_limitation_error(
                            LabOneQError::ResourceExhaustion(e),
                            AUTO_CHUNK_EXHAUSTED_MSG,
                        ));
                    }
                    let multiplier = e.usage.unwrap_or(2.0).ceil().max(2.0) as u32;
                    let new_requested = raw_chunk_count.saturating_mul(multiplier);
                    auto_chunking = auto_chunking.resize(
                        new_requested
                            .try_into()
                            .expect("Internal error: Expected chunk count to be non-zero"),
                    );
                }
                Err(e) => return Err(e),
            }
        },
    }?;

    let combined_output = backend_typed.combined_output_from_py(&compiled_output.artifacts)?;

    // Resolve the raw acquisition lengths and result shapes from the combined output.
    let raw_acquisition_lengths =
        combined_output.raw_acquisition_lengths(&raw_acquisitions, &id_store)?;
    let handle_result_shapes = processed
        .result_shapes
        .get_shapes(raw_acquisition_lengths.into_iter())?;
    let execution_timing = ExecutionTiming {
        total_execution_time: combined_output.total_execution_time()?,
        max_step_execution_time: combined_output.max_execution_time_per_step()?,
    };

    let backend_output = backend_typed.finalize(combined_output, processed.hqcs_program)?;

    let output = CompiledExperiment {
        device_setup_fingerprint: processed.device_setup_fingerprint,
        artifacts: backend_output.artifacts,
        execution_timing,
        metadata: Metadata {
            producer_name: backend_typed.name(),
            producer_version: env!("CARGO_PKG_VERSION").to_string(),
        },
        pulse_sheet_schedule: compiled_output
            .schedule
            .map(|obj| {
                obj.extract::<PyRefMut<PulseSheetSchedulePy>>()
                    .map_err(PyErr::from)
            })
            .transpose()?
            .map(|mut s| s.take_inner()),
        execution,
        real_time_properties: RealTimeProperties {
            acquisition_type: rt_properties.acquisition_type,
            averaging_mode: rt_properties.averaging_mode,
            shots: rt_properties.shots,
            chunk_count,
        },
        result_shapes: ResultShapes {
            handle_result_shapes,
        },
        sweep_parameters,
        acquisition_handles: processed.acquisition_handles,
        coprocessors,
        id_store: Arc::clone(&id_store),
        object_store: Arc::clone(&py_object_store),
    };
    Ok(output)
}

fn create_compiler_settings_py<'py>(
    py: Python<'py>,
    compiler_settings: &CompilerSettings,
) -> PyResult<Bound<'py, PyAny>> {
    let compiler_settings_py = py
        .import(intern!(py, "laboneq.compiler.common.compiler_settings"))?
        .getattr(intern!(py, "CompilerSettings"))?;

    let kwargs = PyDict::new(py);
    kwargs.set_item(intern!(py, "LOG_REPORT"), compiler_settings.log_report)?;
    kwargs.set_item(
        intern!(py, "IGNORE_RESOURCE_LIMITATION_ERRORS"),
        compiler_settings.ignore_resource_exhaustion,
    )?;
    let compiled_settings = compiler_settings_py.call((), Some(&kwargs))?;
    Ok(compiled_settings)
}

/// A chunk count of 1 is equivalent to not chunking at all.
fn sanitize_chunk_count(chunk_count: u32) -> Option<u32> {
    Some(chunk_count).filter(|&c| c != 1)
}

fn wrap_resource_limitation_error(error: LabOneQError, error_msg: &'static str) -> LabOneQError {
    if let LabOneQError::ResourceExhaustion(e) = error {
        laboneq_error::laboneq_error!("{}", error_msg).with_context(|| e.to_string())
    } else {
        error
    }
}

fn call_compile<'py>(
    py: Python<'py>,
    bound_exp_py: &Bound<'py, ExperimentPy>,
    execution: &Bound<'_, PyAny>,
    chunk_count: Option<u32>,
    backend_name: &str,
    compiler_settings: &Bound<'_, PyAny>,
) -> Result<CompilationOutputPy<'py>, LabOneQError> {
    let compile_fn = py
        .import(intern!(py, "laboneq.compiler.workflow.compiler"))?
        .getattr(intern!(py, "compile_whole_or_with_chunks"))?;

    let kwargs = PyDict::new(py);
    kwargs.set_item(intern!(py, "experiment"), bound_exp_py)?;
    kwargs.set_item(intern!(py, "execution"), execution)?;
    kwargs.set_item(intern!(py, "chunk_count"), chunk_count)?;
    kwargs.set_item(intern!(py, "backend"), backend_name)?;
    kwargs.set_item(intern!(py, "compiler_settings"), compiler_settings)?;
    let result = compile_fn.call((), Some(&kwargs));

    match result {
        Ok(result) => Ok(result.extract()?),
        Err(e) if e.is_instance(py, &ResourceLimitationError::type_object(py)) => {
            let usage = e
                .value(py)
                .getattr(intern!(py, "usage"))
                .ok()
                .and_then(|v: Bound<'_, PyAny>| v.extract().ok());
            bail_resource_usage!("{}", e.to_string(), usage = usage);
        }
        Err(e) => {
            if let Some(cause) = e.cause(py) {
                // Remove original Python error cause and rebuild the error chain.
                //
                // This is necessary since the error chain is:
                // Python (this) - Rust (rt compiler) - Python (pulse sampler)
                //
                // We need to convert the first Python error to Rust error for chain to become:
                // Rust - Python (pulse sampler)
                // For proper error reporting from custom Python pulse sampling functions.
                // TODO: Python is removed as middle man, this workaround is no longer needed.
                e.set_cause(py, None);
                let mut cause: PyErrorWithContext = cause.into();
                cause.add_context(|| e.to_string());
                return Err(LabOneQError::PythonException(cause));
            }
            Err(e.into())
        }
    }
}

/// The output of the Python function `compile_whole_or_with_chunks`.
#[derive(FromPyObject, Debug)]
struct CompilationOutputPy<'py> {
    artifacts: Bound<'py, PyAny>,
    schedule: Option<Bound<'py, PyAny>>,
}

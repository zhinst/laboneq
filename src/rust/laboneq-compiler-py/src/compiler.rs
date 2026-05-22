// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::{PyTypeInfo, intern};

use laboneq_common::compiler_settings::CompilerSettings;
use laboneq_error::{
    LabOneQError, PyErrorWithContext, ResourceLimitationError, WithContext, bail_resource_usage,
};

use crate::ProcessedExperiment;
use crate::chunking_mode::{ChunkingMode, collect_chunking_mode};
use crate::compiler_backend::CompilerBackend;
use crate::execution::{Statement, create_execution};
use crate::py_execution::create_py_execution;
use crate::py_experiment::ExperimentPy;

pub(crate) fn run_compilation<B>(
    py: Python<'_>,
    backend: B,
    processed: ProcessedExperiment<B::Output>,
    compiler_settings: CompilerSettings,
) -> Result<Bound<'_, PyAny>, LabOneQError>
where
    B: CompilerBackend + Send + Sync + 'static,
    B::Output: Send + Sync + 'static,
    B::CodeGenArtifact: Send + Sync + 'static,
{
    let execution = create_execution(&processed.inner)?;
    let chunking_mode = collect_chunking_mode(&processed.inner)?;
    compile_whole_or_with_chunks(
        py,
        backend,
        processed,
        compiler_settings,
        execution,
        chunking_mode,
    )
}

const RESOURCE_LIMIT_MSG: &str = "Compilation error - resource limitation exceeded.\n\
    To circumvent this, try one or more of the following:\n\
    - Double check the integrity of your experiment (look for unexpectedly long pulses, large number of sweep steps, etc.)\n\
    - Reduce the number of sweep steps\n\
    - Reduce the number of variations in the pulses that are being played\n\
    - Enable chunking for a sweep\n\
    - If chunking is already enabled, increase the chunk count or switch to automatic chunking";

const AUTO_CHUNK_EXHAUSTED_MSG: &str = "Automatic chunking was not able to find a chunk count to circumvent resource limitations.\n\
    This means that one iteration of a sweep is too large and cannot be executed.\n\
    To circumvent this, try one or more of the following:\n\
    - Chunking another sweep (e.g. in case of nested sweeps, enable chunking for the inner one)\n\
    - Find ways suitable for your use case to reduce the size of the program in one iteration";

fn compile_whole_or_with_chunks<'py, B>(
    py: Python<'py>,
    backend: B,
    processed: ProcessedExperiment<B::Output>,
    compiler_settings: CompilerSettings,
    execution: Vec<Statement>,
    chunking_mode: Option<ChunkingMode>,
) -> Result<Bound<'py, PyAny>, LabOneQError>
where
    B: CompilerBackend + Send + Sync + 'static,
    B::Output: Send + Sync + 'static,
    B::CodeGenArtifact: Send + Sync + 'static,
{
    // Prepare the Python objects
    let device_class = backend.device_class();

    let exp_py = ExperimentPy {
        inner: processed.inner,
        device_setup: processed.device_setup,
        context: processed.context,
        delay_compensation: processed.delay_compensation,
        compiler_settings: compiler_settings.clone(),
        backend_data: Arc::new(processed.backend_data),
        backend: Arc::new(backend),
        result_shapes: processed.result_shapes,
    };

    let execution = create_py_execution(
        py,
        execution,
        &exp_py.inner.id_store,
        &exp_py.inner.py_object_store,
    )?
    .into_pyobject(py)
    .map_err(LabOneQError::from_err)?;

    let compiler_settings = create_compiler_settings_py(py, &exp_py.compiler_settings)?;
    let bound_exp_py = exp_py.into_pyobject(py)?;

    // Run the compilation
    match chunking_mode {
        None => call_compile(
            py,
            &bound_exp_py,
            &execution,
            None,
            device_class,
            &compiler_settings,
        )
        .map_err(|e| wrap_resource_limitation_error(e, RESOURCE_LIMIT_MSG)),
        Some(ChunkingMode::Manual { chunk_count }) => call_compile(
            py,
            &bound_exp_py,
            &execution,
            Some(chunk_count.get()),
            device_class,
            &compiler_settings,
        )
        .map_err(|e| wrap_resource_limitation_error(e, RESOURCE_LIMIT_MSG)),
        Some(ChunkingMode::Auto(mut auto_chunking)) => loop {
            let chunk_count = auto_chunking.initial_chunk_count.get();
            laboneq_log::debug!("Attempting to compile with {} chunks", chunk_count);
            let result = call_compile(
                py,
                &bound_exp_py,
                &execution,
                Some(chunk_count),
                device_class,
                &compiler_settings,
            );

            match result {
                Ok(result) => {
                    laboneq_log::info!("Auto-chunked sweep divided into {} chunks", chunk_count);
                    return Ok(result);
                }
                Err(LabOneQError::ResourceExhaustion(e)) => {
                    laboneq_log::debug!(
                        "The attempt with {} chunks failed with resource exhaustion: {}",
                        chunk_count,
                        e
                    );
                    if chunk_count == auto_chunking.iterations.get() {
                        return Err(wrap_resource_limitation_error(
                            LabOneQError::ResourceExhaustion(e),
                            AUTO_CHUNK_EXHAUSTED_MSG,
                        ));
                    }
                    let multiplier = e.usage.ceil().max(2.0) as u32;
                    let new_requested = chunk_count.saturating_mul(multiplier);
                    auto_chunking = auto_chunking.resize(
                        new_requested
                            .try_into()
                            .expect("Internal error: Expected chunk count to be non-zero"),
                    );
                }
                Err(e) => return Err(e),
            }
        },
    }
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
    device_class: usize,
    compiler_settings: &Bound<'_, PyAny>,
) -> Result<Bound<'py, PyAny>, LabOneQError> {
    let compile_fn = py
        .import(intern!(py, "laboneq.compiler.workflow.compiler"))?
        .getattr(intern!(py, "compile_whole_or_with_chunks"))?;

    let kwargs = PyDict::new(py);
    kwargs.set_item(intern!(py, "experiment"), bound_exp_py)?;
    kwargs.set_item(intern!(py, "execution"), execution)?;
    kwargs.set_item(intern!(py, "chunk_count"), chunk_count)?;
    kwargs.set_item(intern!(py, "device_class"), device_class)?;
    kwargs.set_item(intern!(py, "compiler_settings"), compiler_settings)?;
    let result = compile_fn.call((), Some(&kwargs));

    match result {
        Ok(result) => Ok(result),
        Err(e) if e.is_instance(py, &ResourceLimitationError::type_object(py)) => {
            let usage: f64 = e
                .value(py)
                .getattr(intern!(py, "usage"))
                .ok()
                .and_then(|v: Bound<'_, PyAny>| v.extract().ok())
                .expect("ResourceLimitationError without usage information");
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
                return Err(LabOneQError::PulseSamplerCallback(cause));
            }
            Err(e.into())
        }
    }
}

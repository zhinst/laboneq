// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use laboneq_common::compiler_settings::CompilerSettings;
use laboneq_dsl::types::{HandleUid, SignalUid};
use laboneq_error::{LabOneQError, WithContext};

use crate::capnp_compiled_experiment;
use crate::chunking_mode::{ChunkingMode, collect_chunking_mode};
use crate::compiled_experiment::{
    CompiledExperiment, ExecutionTiming, Metadata, RealTimeProperties, ResultShapes,
};
use crate::compiler_backend::{CombinedOutput, CompilerBackend};
use crate::error::{CompilationError, ResolveErr};
use crate::execution::{Execution, create_execution};
use crate::preprocessor::{PreprocessedExperiment, preprocess_experiment};
use crate::py_compile_pass::CompilePass;
use crate::real_time_loop::real_time_loop_properties;

/// Compile an experiment from Cap'n Proto bytes into serialized `CompiledExperiment` bytes.
///
/// `packed` describes `capnp_data` only; the returned bytes are always unpacked.
pub(crate) fn compile_experiment<B: CompilerBackend>(
    capnp_data: &[u8],
    packed: bool,
    compiler_settings: CompilerSettings,
    backend: B,
) -> Result<Vec<u8>, CompilationError> {
    let processed = preprocess_experiment(capnp_data, packed, &backend)?;
    let id_store = Arc::clone(&processed.inner.id_store);

    let compiled_experiment = run_compilation(backend, processed, compiler_settings)
        .map_err(|e| CompilationError::new(e, Arc::clone(&id_store)))?;

    let serialized_payload =
        capnp_compiled_experiment::serialize_compiled_experiment(&compiled_experiment)
            .resolve_err(&id_store)?;

    Ok(serialized_payload)
}

fn run_compilation<B: CompilerBackend>(
    backend: B,
    processed: PreprocessedExperiment<B::Output, B::EmulatedHqcsProgram>,
    compiler_settings: CompilerSettings,
) -> Result<CompiledExperiment<B::CompilerArtifact>, LabOneQError> {
    let execution = create_execution(&processed.inner)?;
    let chunking_mode = collect_chunking_mode(&processed.inner)?;

    compile_whole_or_with_chunks(
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
    backend: B,
    processed: PreprocessedExperiment<B::Output, B::EmulatedHqcsProgram>,
    compiler_settings: CompilerSettings,
    execution: Execution,
    chunking_mode: Option<ChunkingMode>,
) -> Result<CompiledExperiment<B::CompilerArtifact>, LabOneQError> {
    // The compile pass takes `processed.inner` and hands it to Python for the length of
    // the compilation, so what the compiled experiment still needs comes out first.
    let rt_properties = real_time_loop_properties(&processed.inner.root)?;
    let id_store = Arc::clone(&processed.inner.id_store);
    let external_parameters = Arc::clone(&processed.inner.external_parameters);
    let sweep_parameters = processed.inner.parameters.clone();

    let raw_acquisitions: Vec<(SignalUid, HandleUid)> =
        processed.result_shapes.raw_acquisitions().collect();

    let pass = CompilePass::new(
        backend.clone(),
        processed.inner,
        processed.signals,
        processed.backend_data,
        compiler_settings.clone(),
        &execution,
    )?;

    let compile = |chunk_count| -> Result<_, LabOneQError> {
        let mut output = pass.compile(chunk_count)?;
        backend.finish_attempt(&mut output.combined_output, &compiler_settings, chunk_count)?;
        Ok(output)
    };
    let (output, chunk_count) = run_chunked(chunking_mode, compile)?;

    // Resolve the raw acquisition lengths and result shapes from the combined output.
    let raw_acquisition_lengths = output
        .combined_output
        .raw_acquisition_lengths(&raw_acquisitions, &id_store)?;
    let handle_result_shapes = processed
        .result_shapes
        .get_shapes(raw_acquisition_lengths.into_iter())?;
    let execution_timing = ExecutionTiming {
        total_execution_time: output.combined_output.total_execution_time()?,
        max_step_execution_time: output.combined_output.max_execution_time_per_step()?,
    };

    let backend_output = backend.produce_output(output.combined_output, processed.hqcs_program)?;

    Ok(CompiledExperiment {
        device_setup_fingerprint: processed.device_setup_fingerprint,
        artifacts: backend_output.artifacts,
        execution_timing,
        metadata: Metadata {
            producer_name: backend.name(),
            producer_version: env!("CARGO_PKG_VERSION").to_string(),
        },
        pulse_sheet_schedule: output.pulse_sheet_schedule,
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
        coprocessors: processed.coprocessors,
        id_store,
        external_parameters,
    })
}

/// Run the compilation pass, retrying with more chunks while auto-chunking is
/// still allowed to grow. Returns the output and the chunk count that produced it.
fn run_chunked<T>(
    chunking_mode: Option<ChunkingMode>,
    compile: impl Fn(Option<u32>) -> Result<T, LabOneQError>,
) -> Result<(T, Option<u32>), LabOneQError> {
    let compile_once = |chunk_count| compile(chunk_count).map(|output| (output, chunk_count));
    let mut auto_chunking = match chunking_mode {
        None => {
            return compile_once(None).map_err(|e| with_hint(e, RESOURCE_LIMIT_MSG));
        }
        Some(ChunkingMode::Manual { chunk_count }) => {
            return compile_once(sanitize_chunk_count(chunk_count.get()))
                .map_err(|e| with_hint(e, RESOURCE_LIMIT_MSG));
        }
        Some(ChunkingMode::Auto(auto_chunking)) => auto_chunking,
    };

    loop {
        let requested = auto_chunking.initial_chunk_count.get();
        laboneq_log::debug!("Attempting to compile with {} chunks", requested);
        match compile_once(sanitize_chunk_count(requested)) {
            Ok(result) => {
                laboneq_log::info!("Auto-chunked sweep divided into {} chunks", requested);
                return Ok(result);
            }
            Err(LabOneQError::ResourceExhaustion(e)) => {
                laboneq_log::debug!(
                    "The attempt with {} chunks failed with resource exhaustion: {}",
                    requested,
                    e
                );
                if requested == auto_chunking.iterations.get() {
                    return Err(with_hint(
                        LabOneQError::ResourceExhaustion(e),
                        AUTO_CHUNK_EXHAUSTED_MSG,
                    ));
                }
                let multiplier = e.usage.unwrap_or(2.0).ceil().max(2.0) as u32;
                auto_chunking = auto_chunking.resize(
                    requested
                        .saturating_mul(multiplier)
                        .try_into()
                        .expect("Internal error: Expected chunk count to be non-zero"),
                );
            }
            Err(e) => return Err(e),
        }
    }
}

/// A chunk count of 1 is equivalent to not chunking at all.
fn sanitize_chunk_count(chunk_count: u32) -> Option<u32> {
    Some(chunk_count).filter(|&c| c != 1)
}

fn with_hint(error: LabOneQError, hint: &'static str) -> LabOneQError {
    if let LabOneQError::ResourceExhaustion(e) = error {
        laboneq_error::laboneq_error!("{}", hint).with_context(|| e.to_string())
    } else {
        error
    }
}

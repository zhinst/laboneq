// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::num::NonZeroU32;
use std::sync::Arc;

use laboneq_common::compiler_settings::CompilerSettings;
use laboneq_dsl::types::{HandleUid, SignalUid};
use laboneq_error::LabOneQError;

use crate::capnp_compiled_experiment;
use crate::compiled_experiment::{
    CompiledExperiment, ExecutionTiming, Metadata, RealTimeProperties, ResultShapes,
};
use crate::compiler_backend::{CombinedOutput, CompilerBackend};
use crate::error::{CompilationError, ResolveErr};
use crate::execution::create_execution;
use crate::preprocessor::{PreprocessedExperiment, preprocess_experiment};
use crate::real_time_loop::real_time_loop_properties;
use crate::rt_jobs::compile_rt_jobs;

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
    let rt_properties = real_time_loop_properties(&processed.inner.root)?;
    let execution = create_execution(&processed.inner.root)?;
    let raw_acquisitions: Vec<(SignalUid, HandleUid)> =
        processed.result_shapes.raw_acquisitions().collect();

    let output = compile_rt_jobs(
        &backend,
        &processed.inner,
        &processed.signals,
        &processed.backend_data,
        &compiler_settings,
    )?;
    drop(processed.inner.root);
    drop(processed.inner.pulses);

    // Resolve the raw acquisition lengths and result shapes from the combined output.
    let raw_acquisition_lengths = output
        .combined_output
        .raw_acquisition_lengths(&raw_acquisitions, &processed.inner.id_store)?;
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
            chunk_count: output.chunk_count.map(NonZeroU32::get),
        },
        result_shapes: ResultShapes {
            handle_result_shapes,
        },
        sweep_parameters: processed.inner.parameters,
        acquisition_handles: processed.acquisition_handles,
        coprocessors: processed.coprocessors,
        id_store: processed.inner.id_store,
        external_parameters: processed.inner.external_parameters,
    })
}

// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

//! Preprocessing stage: turns serialized input into a validated experiment
//! ready for compilation.
//!
//! Each backend contributes its hardware-specific share via
//! [`CompilerBackend::preprocess_experiment`].

use std::sync::Arc;

use pyo3::prelude::*;
use tracing::instrument;

use laboneq_dsl::types::HandleUid;
use laboneq_error::LabOneQError;
use laboneq_ir::signal::Signal;

use crate::capnp_deserializer;
use crate::compiler_backend::{CompilerBackend, ExperimentView};
use crate::error::ResolveErr;
use crate::experiment::Experiment;
use crate::experiment_processor::process_experiment;
use crate::experiment_validation::validate_experiment;
use crate::result_shape::{ResultShapes, extract_result_shapes};
use crate::setup_processor::process_setup;

pub(crate) struct PreprocessedExperiment<D, H> {
    pub(crate) inner: Experiment,
    // NOTE: The usage of Arc here is to allow sharing the id_store across Python bindings
    // Remove when Python bindings are no longer needed
    pub(crate) signals: Arc<[Signal]>,
    pub(crate) backend_data: D,
    pub(crate) result_shapes: ResultShapes,
    pub(crate) device_setup_fingerprint: String,
    /// Pre-declared acquisition handles, in stable index order.
    pub(crate) acquisition_handles: Vec<HandleUid>,
    pub(crate) hqcs_program: H,
    pub(crate) coprocessors: laboneq_dsl::hqcs::CoprocessorTable,
}

/// Preprocess an experiment from Cap'n Proto bytes plus device/signal configuration.
#[instrument(
    level = "debug",
    name = "laboneq.compiler.preprocess_experiment",
    skip_all
)]
pub(crate) fn preprocess_experiment<B: CompilerBackend>(
    py: Python<'_>,
    capnp_data: &[u8],
    packed: bool,
    backend: &B,
) -> Result<PreprocessedExperiment<B::Output, B::EmulatedHqcsProgram>, LabOneQError> {
    let mut deserialized = capnp_deserializer::deserialize_experiment(py, capnp_data, packed)?;
    let acquisition_handles = deserialized.handles;

    let hqcs_program = backend
        .process_hqcs(
            &mut deserialized.root,
            &deserialized.hqcs,
            &deserialized.id_store,
            &deserialized.parameters,
        )
        .resolve_err(&deserialized.id_store)?;
    let coprocessors = laboneq_dsl::hqcs::CoprocessorTable::try_from(&deserialized.hqcs)?;

    let backend_processed = backend
        .preprocess_experiment(ExperimentView::new(
            &deserialized.root,
            &mut deserialized.id_store,
            deserialized
                .parameters
                .iter()
                .map(|p| (p.uid, p.inner_values().as_ref()))
                .collect(),
            &deserialized.pulses,
            deserialized.experiment_signals.clone(),
            deserialized.setup_description,
        ))
        .resolve_err(&deserialized.id_store)?;

    let mut experiment = Experiment {
        root: deserialized.root,
        id_store: deserialized.id_store.into(),
        parameters: deserialized.parameters,
        pulses: deserialized.pulses,
        py_object_store: deserialized.external_parameter_values.into(),
    };

    let signals = process_setup(
        backend_processed.device_signals,
        &backend_processed.backend_data,
        &experiment.id_store,
    )
    .resolve_err(&experiment.id_store)?;

    process_experiment(&mut experiment, &signals).resolve_err(&experiment.id_store)?;
    validate_experiment(
        &experiment,
        &signals,
        backend,
        &backend_processed.backend_data,
    )
    .resolve_err(&experiment.id_store)?;
    let result_shapes = extract_result_shapes(
        &experiment.root,
        experiment.parameters.iter(),
        Arc::get_mut(&mut experiment.id_store).expect("Expected no additional ID stores"),
        backend,
    )
    .resolve_err(&experiment.id_store)?;

    let res = PreprocessedExperiment {
        inner: experiment,
        signals: signals.into(),
        backend_data: backend_processed.backend_data,
        result_shapes,
        device_setup_fingerprint: backend_processed.device_setup_fingerprint,
        acquisition_handles,
        hqcs_program,
        coprocessors,
    };
    Ok(res)
}

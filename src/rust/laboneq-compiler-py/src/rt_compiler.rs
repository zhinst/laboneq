// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};
use std::num::NonZeroU32;

use laboneq_common::compiler_settings::CompilerSettings;
use laboneq_dsl::types::{NumericLiteral, ParameterUid};
use laboneq_error::LabOneQError;
use laboneq_ir::ExperimentIr;
use laboneq_ir::pulse_sheet_schedule::PulseSheetSchedule;
use laboneq_ir::system::DeviceSetup;
use laboneq_scheduler::{
    ChunkingInfo, ExperimentContext as SchedulerContext, ParameterStore, ParameterStoreBuilder,
    schedule_experiment,
};

use crate::compiler_backend::{CodeGenArtifact, DynCompilerBackend, PreprocessedBackendData};
use crate::experiment::Experiment;
use crate::experiment_context::ExperimentContext;
use crate::prepare_schedule;
use crate::signal_view::signal_views;

pub(crate) struct RealTimeCompilerInput<'a> {
    pub experiment: &'a Experiment,
    pub device_setup: &'a DeviceSetup,
    pub compiler_settings: &'a CompilerSettings,
    pub context: &'a ExperimentContext,
    pub parameters: HashMap<ParameterUid, NumericLiteral>,
    pub chunking_info: Option<(usize, usize)>,
    pub backend: &'a dyn DynCompilerBackend,
    pub backend_data: &'a (dyn PreprocessedBackendData + Send + Sync),
}

pub(crate) struct RealTimeCompilerOutput {
    pub codegen_output: Box<dyn CodeGenArtifact + Send + Sync>,
    pub used_parameters: HashSet<ParameterUid>,
    pub pulse_sheet_schedule: Option<PulseSheetSchedule>,
}

pub(crate) fn compile_realtime(
    input: RealTimeCompilerInput<'_>,
) -> Result<RealTimeCompilerOutput, LabOneQError> {
    let chunking_info = if let Some((index, count)) = input.chunking_info {
        let count = NonZeroU32::new(count as u32).ok_or_else(|| {
            laboneq_error::laboneq_error!("Chunk count must be a positive integer")
        })?;
        Some(ChunkingInfo { index, count })
    } else {
        None
    };

    let views = signal_views(input.device_setup);

    let feedback_calculator = input
        .backend
        .feedback_calculator(
            &views.values().cloned().collect::<Vec<_>>(),
            input.compiler_settings,
        )
        .map_err(|e| laboneq_error::laboneq_error!("{e}"))?;

    let _t = laboneq_log::StageTiming::start("Schedule");
    let mut parameter_store = create_parameter_store(input.parameters);
    let experiment = input.experiment;
    let result = schedule_experiment(
        &experiment.root,
        SchedulerContext {
            id_store: &experiment.id_store,
            parameters: experiment.parameters.clone(),
            signals: &views,
            handle_to_signal: input.context.handle_to_signal(),
        },
        &parameter_store,
        chunking_info,
        feedback_calculator.as_ref(),
    )
    .map_err(|e| laboneq_error::laboneq_error!("{e}"))?;
    drop(_t);

    let ir = ExperimentIr {
        root: result.root,
        parameters: result.parameters.values().cloned().collect(),
        pulses: experiment.pulses.values().cloned().collect(),
        acquisition_type: *input.context.acquisition_type(),
        id_store: &experiment.id_store,
        device_setup: input.device_setup,
    };
    let pulse_sheet_schedule = prepare_schedule(&ir, input.compiler_settings);

    let _t = laboneq_log::StageTiming::start("Code generation");
    let code_gen_output = input.backend.generate_code_dyn(
        ir,
        input.compiler_settings,
        &input.experiment.py_object_store,
        input.backend_data,
    )?;
    drop(_t);

    let out = RealTimeCompilerOutput {
        used_parameters: parameter_store.empty_queries(),
        pulse_sheet_schedule,
        codegen_output: code_gen_output,
    };
    Ok(out)
}

fn create_parameter_store(parameters: HashMap<ParameterUid, NumericLiteral>) -> ParameterStore {
    let mut builder = ParameterStoreBuilder::new();
    for (uid, value) in parameters.iter() {
        builder = builder.with_parameter(*uid, *value);
    }
    builder.build()
}

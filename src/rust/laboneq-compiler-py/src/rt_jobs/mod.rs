// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

//! The near-time loop of the compilation. A real-time job is one near-time step,
//! narrowed to one chunk when the experiment chunks a sweep.

mod cache;
mod chunking;
mod environment;
mod list;

use std::num::NonZeroU32;
use std::time::Instant;

use laboneq_common::compiler_settings::CompilerSettings;
use laboneq_dsl::ExperimentNode;
use laboneq_dsl::operation::Operation;
use laboneq_error::LabOneQError;
use laboneq_ir::pulse_sheet_schedule::PulseSheetSchedule;
use laboneq_ir::signal::Signal;
use laboneq_ir::{ExperimentIr, ParameterLookup};
use laboneq_scheduler::{ChunkingInfo, ExperimentContext as SchedulerContext, schedule_experiment};

use crate::compiler_backend::{CompilerBackend, FeedbackCalculator, Linker, SignalDelays};
use crate::error::Error;
use crate::experiment::Experiment;
use crate::signal_view::signal_views;
use cache::VariantCache;
use chunking::{collect_chunking_mode, run_chunked};
use environment::RecordingLookup;
use list::{count_rt_jobs, for_each_rt_job};

/// The real-time jobs of an experiment, compiled and linked.
pub(crate) struct CompiledRtJobs<C> {
    pub(crate) combined_output: C,
    /// The schedule of the first program compiled.
    pub(crate) pulse_sheet_schedule: Option<PulseSheetSchedule>,
    /// The number of chunks the chunked sweep is split into, or `None` when the experiment
    /// runs unchunked.
    pub(crate) chunk_count: Option<NonZeroU32>,
}

/// Compiles and links the real-time program of every real-time job.
///
/// A job reuses the program of an earlier job in the same chunk when both hold
/// the same values for the near-time parameters that the earlier compilation read.
/// With auto-chunking, the chunked sweep is split into more chunks while the
/// output exceeds a resource limit.
pub(crate) fn compile_rt_jobs<B: CompilerBackend>(
    backend: &B,
    experiment: &Experiment,
    signals: &[Signal],
    backend_data: &B::Output,
    compiler_settings: &CompilerSettings,
) -> Result<CompiledRtJobs<B::CombinedOutput>, LabOneQError> {
    let chunking_mode = collect_chunking_mode(experiment)?;
    let real_time_root = find_real_time_root(&experiment.root)
        .ok_or_else(|| Error::new("Internal error: Experiment has no real-time boundary."))?;
    let signal_delays = signals
        .iter()
        .map(|s| SignalDelays {
            uid: s.uid,
            start_delay: s.start_delay,
            signal_delay: s.signal_delay,
        })
        .collect::<Vec<_>>();
    let feedback_calculator =
        backend.feedback_calculator(&signal_delays, compiler_settings, backend_data)?;
    run_chunked(chunking_mode, |chunk_count| {
        let root = &experiment.root;
        let job_count = count_rt_jobs(root, chunk_count);
        let mut cache = VariantCache::default();
        let mut linker = backend.linker();
        let mut variants = vec![];
        let mut pulse_sheet_schedule = None;
        let mut skipping = false;
        let mut step = 0;
        for_each_rt_job(root, &experiment.parameters, chunk_count, |job| {
            step += 1;
            let start = Instant::now();
            let (variant, compiled) = match cache.get(job.env, job.chunk) {
                Some(variant) => (variant, false),
                None => {
                    let chunk = job
                        .chunk
                        .zip(chunk_count)
                        .map(|(index, count)| ChunkingInfo { index, count });
                    let lookup = RecordingLookup::new(job.env);
                    let ir = schedule(
                        experiment,
                        real_time_root,
                        signals,
                        feedback_calculator.as_ref(),
                        &lookup,
                        chunk,
                    )?;
                    if variants.is_empty() {
                        pulse_sheet_schedule = prepare_schedule(&ir, compiler_settings);
                    }
                    let artifact = {
                        let _timing = laboneq_log::StageTiming::start("Code generation");
                        backend.generate_code_traced(
                            &ir,
                            compiler_settings,
                            &experiment.external_parameters,
                            backend_data,
                        )?
                    };
                    cache.insert(lookup.reads(), job.env, job.chunk, variants.len());
                    variants.push(linker.add_artifact(artifact)?);
                    (variants.len() - 1, true)
                }
            };
            linker.add_job(job.key, &variants[variant], compiled)?;
            if compiled {
                skipping = false;
                laboneq_log::info!(
                    "Completed compilation step {} of {}. [{:.3} s]",
                    step,
                    job_count,
                    start.elapsed().as_secs_f64()
                );
            } else if !skipping {
                laboneq_log::info!("Skipping compilation for next step(s)...");
                skipping = true;
            }
            Ok(())
        })?;
        let mut combined_output = linker.finish()?;
        backend.finish_attempt(&mut combined_output, compiler_settings, chunk_count)?;
        Ok(CompiledRtJobs {
            combined_output,
            pulse_sheet_schedule,
            chunk_count,
        })
    })
}

fn find_real_time_root(node: &ExperimentNode) -> Option<&ExperimentNode> {
    if node.kind == Operation::RealTimeBoundary {
        return Some(node);
    }
    node.children.iter().find_map(find_real_time_root)
}

fn schedule<'a>(
    experiment: &'a Experiment,
    real_time_root: &ExperimentNode,
    signals: &'a [Signal],
    feedback_calculator: Option<&impl FeedbackCalculator>,
    near_time_parameters: &'a dyn ParameterLookup,
    chunk: Option<ChunkingInfo>,
) -> Result<ExperimentIr<'a>, Error> {
    let _timing = laboneq_log::StageTiming::start("Schedule");
    let result = schedule_experiment(
        real_time_root,
        SchedulerContext {
            id_store: &experiment.id_store,
            parameters: experiment
                .parameters
                .iter()
                .map(|p| (p.uid, p.clone()))
                .collect(),
            signals: &signal_views(signals),
        },
        near_time_parameters,
        chunk,
        feedback_calculator,
    )?;
    Ok(ExperimentIr {
        root: result.root,
        parameters: result.parameters.into_values().collect(),
        pulses: &experiment.pulses,
        acquisition_type: result.acquisition_type,
        id_store: &experiment.id_store,
        signals,
        near_time_parameters,
    })
}

fn prepare_schedule(
    ir: &ExperimentIr,
    compiler_settings: &CompilerSettings,
) -> Option<PulseSheetSchedule> {
    if !compiler_settings.output_extras {
        return None;
    }
    let out = PulseSheetSchedule::generate(
        &ir.root,
        ir.signals
            .iter()
            .map(|s| (s.traits.name, s.sampling_rate))
            .collect(),
        compiler_settings.expand_loops_for_schedule,
        compiler_settings.max_events_to_publish,
        ir.id_store,
    );
    Some(out)
}

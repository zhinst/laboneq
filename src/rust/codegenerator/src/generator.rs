// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use crate::CodeGeneratorSettings;
use crate::Result;
use crate::event_list::generate_event_list;
use crate::generate_awg_events::transform_ir_to_awg_events;
use crate::handle_feedback_registers::FeedbackRegisterAllocation;
use crate::handle_feedback_registers::{FeedbackConfig, collect_feedback_config};
use crate::ir::IrNode;
use crate::ir::compilation_job::AwgCore;
use crate::ir::compilation_job::AwgKey;
use crate::ir::compilation_job::SignalKind;
use crate::ir::experiment::AcquisitionType;
use crate::ir::experiment::Handle;
use crate::passes::analyze_awg::analyze_awg_ir;
use crate::passes::analyze_measurements;
use crate::passes::fanout_awg::fanout_for_awg;
use crate::result::AwgCodeGenerationResult;
use crate::result::FeedbackRegisterConfig;
use crate::result::SeqCGenOutput;
use crate::result::SignalIntegrationInfo;
use crate::sample_waveforms::{
    AwgWaveforms, WaveDeclaration, collect_and_finalize_waveforms, collect_integration_kernels,
};
use crate::sampled_event_handler::FeedbackRegisterLayout;
use crate::sampled_event_handler::SeqcResults;
use crate::sampled_event_handler::handle_sampled_events;
use crate::sampled_event_handler::seqc_tracker::FeedbackRegisterIndex;
use crate::sampled_event_handler::seqc_tracker::awg::Awg;
use crate::tinysample::TINYSAMPLE;
use crate::waveform_sampler::SampleWaveforms;

use anyhow::Context;
use laboneq_log::{log, warn};
use rayon::prelude::*;
use std::collections::HashMap;

#[allow(clippy::too_many_arguments)]
fn generate_output(
    node: IrNode,
    awg: &AwgCore,
    wave_declarations: &[WaveDeclaration],
    qa_signals_by_handle: &HashMap<Handle, (String, AwgKey)>,
    emit_timing_comments: bool,
    shf_output_mute_min_duration: Option<f64>,
    has_readout_feedback: bool,
    feedback_register: &Option<FeedbackRegisterIndex>,
    feedback_register_layout: &FeedbackRegisterLayout,
    acquisition_type: &AcquisitionType,
    is_reference_clock_internal: bool,
) -> Result<SeqcResults> {
    let sampled_events = generate_event_list(node, awg)?;
    let awg = Awg {
        signal_kind: awg.kind,
        awg_key: awg.key(),
        play_channels: awg
            .signals
            .iter()
            .find(|s| s.kind != SignalKind::INTEGRATION)
            .map_or_else(Vec::new, |s| s.channels.clone()),
        device_kind: awg.device_kind().clone(),
        sampling_rate: awg.sampling_rate,
        shf_output_mute_min_duration,
        trigger_mode: awg.trigger_mode,
        is_reference_clock_internal,
    };
    let seqc_results = handle_sampled_events(
        sampled_events,
        &awg,
        qa_signals_by_handle,
        wave_declarations,
        *feedback_register,
        feedback_register_layout,
        emit_timing_comments,
        has_readout_feedback,
        acquisition_type,
    )?;
    Ok(seqc_results)
}

#[allow(clippy::too_many_arguments)]
fn generate_code_for_awg<T: SampleWaveforms>(
    root: &IrNode,
    awg: &AwgCore,
    settings: &CodeGeneratorSettings,
    acquisition_type: &AcquisitionType,
    acquisition_config: &FeedbackConfig<'_>,
    feedback_register_layout: &FeedbackRegisterLayout,
    is_reference_clock_internal: bool,
    sampler: &T,
) -> Result<AwgCodeGenerationResult<T>> {
    let root = fanout_for_awg(root, awg);
    let awg_info = analyze_awg_ir(&root);
    let measurement_info = analyze_measurements(&root, awg.device_kind(), awg.sampling_rate)?;
    let mut awg_node = transform_ir_to_awg_events(
        root,
        awg,
        settings,
        &measurement_info
            .delays
            .iter()
            .map(|(signal, delay)| (signal.as_str(), delay.delay_sequencer()))
            .collect(),
    )?;
    let waveforms = if T::supports_waveform_sampling(awg) {
        collect_and_finalize_waveforms(&mut awg_node, sampler, awg)
    } else {
        Ok(AwgWaveforms::default())
    }?;
    let integration_kernels = collect_integration_kernels(&awg_node, awg)?;
    let integration_weights =
        sampler.batch_calculate_integration_weights(awg, integration_kernels)?;
    let qa_signals_by_handle: HashMap<Handle, (String, AwgKey)> = acquisition_config
        .handles()
        .map(|handle| {
            let signal_info = acquisition_config
                .feedback_source(handle)
                .expect("Internal Error: Missing feedback source for handle");
            (
                handle.clone(),
                (
                    signal_info.signal.uid.to_string(),
                    signal_info.awg_key.clone(),
                ),
            )
        })
        .collect();

    let (sampled_waveforms, wave_declarations) = waveforms.into_inner();

    // Feedback registers
    let target_feedback_register = acquisition_config.target_feedback_register(&awg.key());

    let global_feedback_register = match target_feedback_register {
        Some(FeedbackRegisterAllocation::Global { register }) => (*register as u32).into(),
        _ => None,
    };
    let feedback_register = match target_feedback_register {
        Some(FeedbackRegisterAllocation::Global { register }) => (*register as i64).into(),
        Some(FeedbackRegisterAllocation::Local) => (-1).into(),
        None => None,
    };
    let source_feedback_register = if let Some(handle) = awg_info.feedback_handles().first() {
        let source = acquisition_config
            .feedback_source(handle)
            .expect("Internal Error: Missing feedback source for handle");
        acquisition_config.target_feedback_register(&source.awg_key)
    } else {
        None
    };

    let use_automute_playzeros = awg.signals.iter().any(|s| s.automute);
    let shf_output_mute_min_duration = if use_automute_playzeros {
        Some(settings.shf_output_mute_min_duration)
    } else {
        None
    };
    let awg_events = generate_output(
        awg_node,
        awg,
        &wave_declarations,
        &qa_signals_by_handle,
        settings.emit_timing_comments,
        shf_output_mute_min_duration,
        awg_info.has_readout_feedback(),
        &global_feedback_register,
        feedback_register_layout,
        acquisition_type,
        is_reference_clock_internal,
    )?;
    let output = AwgCodeGenerationResult {
        seqc: awg_events.seqc,
        wave_indices: awg_events.wave_indices,
        command_table: awg_events.command_table,
        shf_sweeper_config: awg_events.shf_sweeper_config,
        sampled_waveforms,
        integration_weights,
        signal_delays: measurement_info
            .delays
            .iter()
            .map(|(k, v)| (k.to_string(), v.delay_port().into()))
            .collect(),
        ppc_device: awg_info.ppc_device().cloned(),
        integration_lengths: measurement_info
            .integration_lengths
            .into_iter()
            .map(|x| {
                (
                    x.signal().to_string(),
                    SignalIntegrationInfo {
                        is_play: x.is_play(),
                        length: x.duration(),
                    },
                )
            })
            .collect(),
        parameter_phase_increment_map: awg_events.parameter_phase_increment_map,
        feedback_register_config: FeedbackRegisterConfig {
            local: matches!(
                target_feedback_register,
                Some(FeedbackRegisterAllocation::Local)
            ),
            source_feedback_register: source_feedback_register.map(|sfr| match sfr {
                FeedbackRegisterAllocation::Local => -1,
                FeedbackRegisterAllocation::Global { register } => *register as i64,
            }),
            register_index_select: awg_events.feedback_register_config.register_index_select,
            codeword_bitshift: awg_events.feedback_register_config.codeword_bitshift,
            codeword_bitmask: awg_events.feedback_register_config.codeword_bitmask,
            command_table_offset: awg_events.feedback_register_config.command_table_offset,
            target_feedback_register: feedback_register,
        },
    };
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
fn generate_code_for_multiple_awgs<T: SampleWaveforms + Sync + Send>(
    root: &IrNode,
    awgs: &[AwgCore],
    settings: &CodeGeneratorSettings,
    waveform_sampler: &T,
    acquisition_type: &AcquisitionType,
    acquisition_config: &FeedbackConfig<'_>,
    feedback_register_layout: &FeedbackRegisterLayout,
) -> Result<Vec<AwgCodeGenerationResult<T>>> {
    let awg_results: Vec<AwgCodeGenerationResult<T>> = awgs
        .par_iter()
        .map(|awg| -> Result<AwgCodeGenerationResult<T>> {
            let code = generate_code_for_awg(
                root,
                awg,
                settings,
                acquisition_type,
                acquisition_config,
                feedback_register_layout,
                awg.is_reference_clock_internal,
                waveform_sampler,
            )
            .context(format!(
                "Error while generating code for signals: {}",
                &awg.signals
                    .iter()
                    .map(|s| s.uid.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            ))?;
            Ok(code)
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(awg_results)
}

fn estimate_total_execution_time(root: &IrNode) -> f64 {
    root.data().length() as f64 * TINYSAMPLE
}

pub fn generate_code<T: SampleWaveforms + Sync + Send>(
    root: &IrNode,
    awgs: &[AwgCore],
    acquisition_type: &AcquisitionType,
    feedback_register_layout: &FeedbackRegisterLayout,
    mut settings: CodeGeneratorSettings,
    sampler: &T,
) -> Result<SeqCGenOutput<T>> {
    for msg in settings.sanitize()? {
        warn!(
            "Compiler setting `{}` is sanitized from {} to {}. Reason: {}",
            msg.field.to_uppercase(),
            msg.original,
            msg.sanitized,
            msg.reason
        );
    }
    let total_execution_time = estimate_total_execution_time(root);
    let awg_refs: Vec<&AwgCore> = awgs.iter().collect();
    let feedback_config: FeedbackConfig<'_> = collect_feedback_config(root, &awg_refs)
        .context("Error while processing feedback configuration")?;
    let awg_results = generate_code_for_multiple_awgs(
        root,
        awgs,
        &settings,
        sampler,
        acquisition_type,
        &feedback_config,
        feedback_register_layout,
    )?;
    let result = SeqCGenOutput {
        awg_results,
        total_execution_time,
        simultaneous_acquires: feedback_config.into_acquisitions(),
    };
    Ok(result)
}

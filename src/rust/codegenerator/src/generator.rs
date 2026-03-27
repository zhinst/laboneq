// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use crate::CodeGeneratorSettings;
use crate::CodegenIr;
use crate::Result;
use crate::context::CodeGenContext;
use crate::event_list::generate_event_list;
use crate::generate_awg_events::transform_ir_to_awg_events;
use crate::handle_feedback_registers::FeedbackRegisterAllocation;
use crate::handle_feedback_registers::calculate_feedback_register_layout;
use crate::handle_feedback_registers::collect_feedback_config;
use crate::integration_units::allocate_integration_units;
use crate::ir::IrNode;
use crate::ir::SignalUid;
use crate::ir::compilation_job::AwgCore;
use crate::ir::compilation_job::AwgKey;
use crate::ir::compilation_job::DeviceKind;
use crate::ir::experiment::AcquisitionType;
use crate::ir::experiment::Handle;
use crate::passes::MeasurementAnalysis;
use crate::passes::analyze_awg::AwgCompilationInfo;
use crate::passes::analyze_awg::analyze_awg_ir;
use crate::passes::analyze_measurements;
use crate::passes::fanout_awg::fanout_for_awg;
use crate::result::Acquisition;
use crate::result::AwgCodeGenerationResult;
use crate::result::AwgProperties;
use crate::result::ChannelProperties;
use crate::result::CommandTable;
use crate::result::FeedbackRegisterConfig;
use crate::result::InputChannelProperties;
use crate::result::IntegrationWeight;
use crate::result::IntegratorAllocation;
use crate::result::Measurement;
use crate::result::ResultSource;
use crate::result::SampledWaveform;
use crate::result::SeqCGenOutput;
use crate::result::SeqCProgram;
use crate::result::SequencerType;
use crate::result::ShfPpcSweepJson;
use crate::result::SignalIntegrationInfo;
use crate::sample_waveforms::SampledIntegrationKernel;
use crate::sample_waveforms::{
    AwgWaveforms, WaveDeclaration, collect_and_finalize_waveforms, collect_integration_kernels,
};
use crate::sampled_event_handler::SeqcResults;
use crate::sampled_event_handler::handle_sampled_events;
use crate::sampled_event_handler::seqc_tracker::FeedbackRegisterIndex;
use crate::sampled_event_handler::seqc_tracker::awg::Awg;
use crate::waveform_sampler::SampleWaveforms;

use laboneq_error::LabOneQError;
use laboneq_error::WithContext;
use laboneq_error::resource_usage::ResourceExhaustionError;
use laboneq_error::resource_usage::handle_resource_exhaustion;
use laboneq_log::warn;
use laboneq_units::tinysample::{tiny_samples, tinysamples_to_seconds};
use rayon::prelude::*;
use std::collections::HashMap;

#[allow(clippy::too_many_arguments)]
fn generate_output(
    node: IrNode,
    awg: &AwgCore,
    wave_declarations: &[WaveDeclaration],
    qa_signals_by_handle: &HashMap<Handle, (SignalUid, AwgKey)>,
    emit_timing_comments: bool,
    shf_output_mute_min_duration: Option<f64>,
    has_readout_feedback: bool,
    feedback_register: &Option<FeedbackRegisterIndex>,
    ctx: &CodeGenContext,
) -> Result<SeqcResults> {
    let sampled_events = generate_event_list(node, awg, ctx)?;
    let awg = Awg {
        signal_kind: awg.kind,
        awg_key: awg.key(),
        play_channels: awg
            .signals
            .iter()
            .find(|s| s.is_output())
            .map_or_else(Vec::new, |s| s.channels.clone()),
        device_kind: *awg.device_kind(),
        sampling_rate: awg.sampling_rate,
        shf_output_mute_min_duration,
        trigger_mode: awg.trigger_mode,
    };
    let seqc_results = handle_sampled_events(
        sampled_events,
        &awg,
        qa_signals_by_handle,
        wave_declarations,
        *feedback_register,
        &ctx.feedback_register_layout,
        emit_timing_comments,
        has_readout_feedback,
        &ctx.acquisition_type,
    )?;
    Ok(seqc_results)
}

fn generate_code_for_awg<T: SampleWaveforms>(
    root: &IrNode,
    awg: &AwgCore,
    ctx: &CodeGenContext,
    sampler: &T,
) -> Result<AwgCodeGenerationResult<T>> {
    let root = fanout_for_awg(root, awg);
    let awg_info = analyze_awg_ir(&root, awg)?;
    let measurement_info = analyze_measurements(&root, awg.device_kind(), awg.sampling_rate)?;
    let mut awg_node = transform_ir_to_awg_events(
        root,
        awg,
        &ctx.settings,
        &measurement_info
            .delays
            .iter()
            .map(|(signal, delay)| (*signal, delay.delay_sequencer()))
            .collect(),
    )?;
    let waveforms = if T::supports_waveform_sampling(awg) {
        collect_and_finalize_waveforms(&mut awg_node, sampler, awg)
    } else {
        Ok(AwgWaveforms::default())
    }?;
    let integration_kernels = collect_integration_kernels(&awg_node, awg)?;
    let integration_weights = sampler.sample_integration_kernels(awg, integration_kernels)?;
    let qa_signals_by_handle: HashMap<Handle, (SignalUid, AwgKey)> = ctx
        .feedback_config
        .handles()
        .map(|handle| {
            let signal_info = ctx
                .feedback_config
                .feedback_source(handle)
                .expect("Internal Error: Missing feedback source for handle");
            (
                handle.clone(),
                (signal_info.signal, signal_info.awg_key.clone()),
            )
        })
        .collect();

    let (sampled_waveforms, wave_declarations) = waveforms.into_inner();

    // Feedback registers
    let target_feedback_register = ctx.feedback_config.target_feedback_register(&awg.key());
    let global_feedback_register = match target_feedback_register {
        Some(FeedbackRegisterAllocation::Global { register }) => (*register as u32).into(),
        _ => None,
    };

    let use_automute_playzeros = awg.signals.iter().any(|s| s.automute);
    let shf_output_mute_min_duration = if use_automute_playzeros {
        Some(ctx.settings.shf_output_mute_min_duration)
    } else {
        None
    };
    let awg_events = generate_output(
        awg_node,
        awg,
        &wave_declarations,
        &qa_signals_by_handle,
        ctx.settings.emit_timing_comments,
        shf_output_mute_min_duration,
        awg_info.has_readout_feedback(),
        &global_feedback_register,
        ctx,
    )?;
    let output = construct_awg_result(
        awg,
        awg_events,
        sampled_waveforms,
        integration_weights,
        measurement_info,
        &awg_info,
        ctx,
    );
    Ok(output)
}

fn construct_awg_result<T>(
    awg: &AwgCore,
    awg_events: SeqcResults,
    sampled_waveforms: Vec<SampledWaveform<T::Signature>>,
    integration_weights: Vec<T::SampledIntegrationKernel>,
    measurement_info: MeasurementAnalysis,
    awg_info: &AwgCompilationInfo,
    ctx: &CodeGenContext,
) -> AwgCodeGenerationResult<T>
where
    T: SampleWaveforms,
{
    let target_feedback_register = ctx.feedback_config.target_feedback_register(&awg.key());
    let feedback_register = match target_feedback_register {
        Some(FeedbackRegisterAllocation::Global { register }) => (*register as i64).into(),
        Some(FeedbackRegisterAllocation::Local) => (-1).into(),
        None => None,
    };
    let source_feedback_register = if let Some(handle) = awg_info.feedback_handles().first() {
        let source = ctx
            .feedback_config
            .feedback_source(handle)
            .expect("Internal Error: Missing feedback source for handle");
        ctx.feedback_config
            .target_feedback_register(&source.awg_key)
    } else {
        None
    };

    // Construct channel properties
    let mut channel_properties = vec![];
    for signal in awg.signals.iter().filter(|s| s.is_output()) {
        let signal_properties = ctx.signal_properties(signal.uid).unwrap_or_else(|| {
            panic!(
                "Internal Error: Missing initial signal properties for signal {:?}",
                signal.uid
            )
        });

        let oscillator_index = awg.oscillator_index(&signal.uid);

        let awg_channels = awg.awg_channels_for_signal(&signal.uid).unwrap();
        for channel in awg_channels.iter() {
            channel_properties.push(ChannelProperties {
                signal: signal.uid,
                channel: *channel,
                marker_mode: awg_info.marker_modes.get(channel).cloned(),
                hw_oscillator_index: oscillator_index,
                amplitude: signal_properties.amplitude.clone(),
            });
        }
    }

    // Construct input channel properties
    let mut input_channel_properties = vec![];
    for signal in awg.signals.iter().filter(|s| !s.is_output()) {
        let oscillator_index = awg.oscillator_index(&signal.uid);

        let awg_channels = awg.awg_channels_for_signal(&signal.uid).unwrap();
        for channel in awg_channels.iter() {
            input_channel_properties.push(InputChannelProperties {
                signal: signal.uid,
                channel: *channel,
                hw_oscillator_index: oscillator_index,
            });
        }
    }

    // Create integration weights by their associated integration channels and sort by channels for output consistency
    let mut integration_weight_infos = integration_weights
        .iter()
        .flat_map(|w| {
            w.signals().iter().map(|signal_uid| {
                ctx.integration_units_for_signal(*signal_uid).map(|units| {
                    let mut sorted_units = units.to_vec();
                    sorted_units.sort();
                    IntegrationWeight {
                        integration_units: sorted_units,
                        basename: w.basename().to_string(),
                        downsampling_factor: w.downsampling_factor().unwrap_or(1),
                    }
                })
            })
        })
        .flatten()
        .collect::<Vec<_>>();
    integration_weight_infos.sort_by(|a, b| a.integration_units.cmp(&b.integration_units));

    let awg_properties = AwgProperties {
        key: awg.key(),
        kind: awg.kind,
        sampling_rate: awg.sampling_rate,
        options: awg.options.clone(),
    };
    AwgCodeGenerationResult {
        awg: awg_properties,
        seqc: create_seqc_program(awg, awg_events.seqc),
        wave_indices: awg_events.wave_indices,
        command_table: {
            if let Some(command_table_results) = awg_events.command_table {
                Some(CommandTable {
                    src: serde_json::to_string(&command_table_results.command_table).unwrap(),
                    n_entries: command_table_results.n_entries,
                    max_entries: awg.device_kind().traits().max_ct_entries.unwrap() as usize,
                    parameter_phase_increment_map: command_table_results
                        .parameter_phase_increment_map,
                })
            } else {
                None
            }
        },
        shf_sweeper_config: awg_events
            .shf_sweeper_config
            .map(|config_json| ShfPpcSweepJson {
                ppc_device: awg_info.ppc_device().cloned().unwrap(),
                json: config_json,
            }),
        sampled_waveforms,
        integration_kernels: integration_weights,
        signal_delays: measurement_info
            .delays
            .iter()
            .map(|(k, v)| (*k, v.delay_port().into()))
            .collect(),
        integration_lengths: measurement_info
            .integration_lengths
            .into_iter()
            .map(|x| {
                (
                    *x.signal(),
                    SignalIntegrationInfo {
                        is_play: x.is_play(),
                        length: x.duration(),
                    },
                )
            })
            .collect(),
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
        output_channel_properties: channel_properties,
        input_channel_properties,
        integration_weights: integration_weight_infos,
        integrator_allocations: construct_integration_allocations(awg, ctx),
    }
}

fn generate_code_for_multiple_awgs<T: SampleWaveforms + Sync + Send>(
    root: &IrNode,
    awgs: &[AwgCore],
    ctx: &CodeGenContext,
    waveform_sampler: &T,
) -> Result<Vec<AwgCodeGenerationResult<T>>> {
    let awg_results: Vec<AwgCodeGenerationResult<T>> = awgs
        .par_iter()
        .map(|awg| {
            let code =
                generate_code_for_awg(root, awg, ctx, waveform_sampler).with_context(|| {
                    format!(
                        "Error while generating code for signals: {}",
                        &awg.signals
                            .iter()
                            .map(|s| s.uid.0.to_string())
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                })?;
            Ok(code)
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(awg_results)
}

fn estimate_total_execution_time(root: &IrNode) -> f64 {
    tinysamples_to_seconds(tiny_samples(root.data().length())).into()
}

fn construct_result_handle_maps(
    simultaneous_acquires: Vec<Vec<Acquisition>>,
    awgs: &[AwgCore],
    ctx: &CodeGenContext,
) -> HashMap<ResultSource, Vec<Vec<String>>> {
    fn _awg_has_acquires(awg: &AwgCore, acquires: &[Acquisition]) -> bool {
        for acq in acquires {
            for sig in &awg.signals {
                if sig.uid == acq.signal {
                    return true;
                }
            }
        }
        false
    }
    let simultaneous_awgs = simultaneous_acquires
        .iter()
        .map(|acquires| awgs.iter().filter(|awg| _awg_has_acquires(awg, acquires)));
    let mut result_handle_maps: HashMap<ResultSource, Vec<Vec<String>>> = HashMap::new();
    let mut signal_to_result_source = HashMap::new();
    for (acquires, awgs) in std::iter::zip(simultaneous_acquires.iter(), simultaneous_awgs) {
        let mut result_map_for_this_round: HashMap<ResultSource, Vec<String>> = HashMap::new();
        for awg in awgs {
            for sig in &awg.signals {
                if sig.is_output() {
                    continue;
                }
                let Some(integration_units) = ctx.integration_units_for_signal(sig.uid) else {
                    continue;
                };
                let integrator_idx = match ctx.acquisition_type {
                    AcquisitionType::RAW => None,
                    _ => Some(integration_units[0]),
                };
                let result_source = ResultSource {
                    device_id: awg.device.uid().to_string(),
                    awg_id: awg.uid,
                    integrator_idx,
                };
                result_map_for_this_round
                    .entry(result_source.clone())
                    .or_default();
                signal_to_result_source.insert(&sig.uid, result_source);
            }
        }
        for acq in acquires {
            let _ = result_map_for_this_round
                .entry(signal_to_result_source.get(&acq.signal).unwrap().clone())
                .and_modify(|entry| {
                    entry.push(acq.handle.to_string());
                });
        }
        result_map_for_this_round
            .into_iter()
            .for_each(|(result_source, val)| {
                result_handle_maps
                    .entry(result_source)
                    .or_default()
                    .push(val);
            });
    }

    result_handle_maps
}

pub fn generate_code<T: SampleWaveforms + Sync + Send>(
    codegen_ir: CodegenIr,
    mut settings: CodeGeneratorSettings,
    sampler: &T,
) -> Result<SeqCGenOutput<T>> {
    let root = &codegen_ir.root;
    let mut awgs = codegen_ir.awg_cores;
    let acquisition_type = codegen_ir.acquisition_type;

    awgs.sort_by_key(|a| a.key());
    for msg in settings.sanitize()? {
        warn!(
            "Compiler setting `{}` is sanitized from {} to {}. Reason: {}",
            msg.field.to_uppercase(),
            msg.original,
            msg.sanitized,
            msg.reason
        );
    }
    // Context construction
    let mut feedback_config = collect_feedback_config(root, &awgs)
        .with_context(|| "Error while processing feedback configuration")?;
    let integration_unit_allocation = allocate_integration_units(root, &awgs, &acquisition_type)?;
    let feedback_register_layout =
        calculate_feedback_register_layout(&awgs, &integration_unit_allocation, &feedback_config);
    let simulaneous_acquires = feedback_config.take_acquisitions();
    let ctx = CodeGenContext {
        acquisition_type,
        feedback_register_layout,
        feedback_config,
        integration_unit_allocation,
        settings,
        initial_signal_properties: codegen_ir.initial_signal_properties,
    };

    // Code generation per AWG
    let awg_results = generate_code_for_multiple_awgs(root, &awgs, &ctx, sampler)?;

    // Resource usage evaluation
    handle_resource_exhaustion(
        evaluate_resource_usage(&awg_results),
        ctx.settings.ignore_resource_exhaustion,
    )?;

    // Result construction
    let total_execution_time = estimate_total_execution_time(root);
    let measurements = evaluate_measurement_per_device(&awg_results);
    let result_handle_maps = construct_result_handle_maps(simulaneous_acquires, &awgs, &ctx);

    let result = SeqCGenOutput {
        awg_results,
        total_execution_time,
        result_handle_maps,
        measurements,
    };
    Ok(result)
}

/// Evaluate the measurements per device.
///
/// The measurements are grouped by device and channel, and the maximum length
/// of the measurements is taken for each channel.
fn evaluate_measurement_per_device(
    awg_results: &[AwgCodeGenerationResult<impl SampleWaveforms>],
) -> Vec<Measurement> {
    let mut measurements_by_awg: HashMap<AwgKey, Vec<Measurement>> = HashMap::new();
    for result in awg_results.iter() {
        if result.integration_lengths.is_empty() {
            continue;
        }
        let max = result
            .integration_lengths
            .values()
            .filter_map(|meas| {
                if !meas.is_play {
                    Some(meas.length)
                } else {
                    None
                }
            })
            .max()
            .unwrap_or_default();
        measurements_by_awg
            .entry(result.awg.key.clone())
            .and_modify(|measurement| {
                for meas in measurement.iter_mut() {
                    meas.length = meas.length.max(max);
                }
            })
            .or_default()
            .push(Measurement {
                device: result.awg.key.device_name().clone(),
                channel: result.awg.key.index(),
                length: max,
            });
    }
    let mut measurements = measurements_by_awg
        .into_values()
        .flatten()
        .collect::<Vec<Measurement>>();
    // Sorted for output consistency
    measurements.sort_by(|a, b| a.channel.cmp(&b.channel));
    measurements
}

fn evaluate_resource_usage(
    awg_results: &[AwgCodeGenerationResult<impl SampleWaveforms>],
) -> impl Iterator<Item = Result<(), LabOneQError>> + '_ {
    awg_results.iter().map(|result| {
        if let Some(ct) = &result.command_table {
            let usage = ct.resource_usage_percentage();
            if usage > 1.0 {
                let msg = format!(
                    "Exceeded max number of command table entries on device '{}', AWG({}): Needed: {}, max available: {}",
                    result.awg.key.device_name(),
                    result.awg.key.index(),
                    ct.n_entries,
                    ct.max_entries
                );
                Err(ResourceExhaustionError::new(msg, usage).into())
            } else {
                Ok(())
            }
        } else {
            Ok(())
        }
    })
}

fn construct_integration_allocations(
    awg: &AwgCore,
    ctx: &CodeGenContext,
) -> Vec<IntegratorAllocation> {
    awg.signals
        .iter()
        .filter_map(|s| ctx.integration_unit_allocation_for_signal(s.uid))
        .filter_map(|alloc| {
            let props = ctx.signal_properties(alloc.signal).unwrap_or_else(|| {
                panic!(
                    "Internal Error: Missing initial signal properties for signal {:?}",
                    alloc.signal
                )
            });
            let thresholds = if props.thresholds.is_empty() {
                // Initialize the thresholds with zeros if not provided.
                // The number of thresholds needed for MSD is given by the formula n*(n-1)/2, where n is the number of states.
                let num_states = alloc.kernel_count.get() + 1;
                let thresholds_needed = (num_states * (num_states - 1)) / 2;
                vec![0.0; thresholds_needed as usize]
            } else {
                props.thresholds.clone()
            };
            IntegratorAllocation {
                signal: alloc.signal,
                integration_units: alloc.units.clone(),
                kernel_count: alloc.kernel_count,
                thresholds,
            }
            .into()
        })
        .collect::<Vec<_>>()
}

fn create_seqc_program(awg: &AwgCore, seqc_code: String) -> SeqCProgram {
    let dev_type = awg
        .options
        .first()
        .map(String::as_str)
        .unwrap_or("")
        .to_string();
    let dev_opts = awg.options.iter().skip(1).map(String::to_string).collect();
    SeqCProgram {
        src: seqc_code,
        dev_type,
        dev_opts,
        awg_index: awg.key().index(),
        sequencer: match awg.device_kind() {
            DeviceKind::SHFSG => SequencerType::Sg,
            DeviceKind::SHFQA => SequencerType::Qa,
            _ => SequencerType::Auto,
        },
        sampling_rate: if matches!(awg.device_kind(), DeviceKind::HDAWG) {
            Some(awg.sampling_rate)
        } else {
            None
        },
    }
}

// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use indexmap::IndexMap;

use crate::CodeGeneratorSettings;
use crate::CodegenIr;
use crate::Result;
use crate::context::CodeGenContext;
use crate::context_validation::validate_codegen_ir;
use crate::event_list::generate_event_list;
use crate::generate_awg_events::transform_ir_to_awg_events;
use crate::handle_feedback_registers::FeedbackRegisterAllocation;
use crate::handle_feedback_registers::calculate_feedback_register_layout;
use crate::handle_feedback_registers::collect_feedback_config;
use crate::integration_kernels::ProcessedKernels;
use crate::integration_kernels::SampleIntegrationKernels;
use crate::integration_kernels::collect_and_process_integration_kernels;
use crate::integration_units::allocate_integration_units;
use crate::ir::IrNode;
use crate::ir::SignalUid;
use crate::ir::compilation_job::AwgCore;
use crate::ir::compilation_job::AwgKey;
use crate::ir::compilation_job::ChannelIndex;
use crate::ir::compilation_job::DeviceKind;
use crate::ir::compilation_job::InitialSignalProperties;
use crate::ir::compilation_job::SignalKind;
use crate::ir::experiment::Handle;
use crate::ir_adapter::value_or_parameter_to_fixed;
use crate::par_trace;
use crate::passes::MeasurementAnalysis;
use crate::passes::analyze_awg::AwgCompilationInfo;
use crate::passes::analyze_awg::analyze_awg_ir;
use crate::passes::analyze_measurements;
use crate::passes::fanout_awg::fanout_for_awg;
use crate::passes::handle_result_shapes::AwgMeasurementShapes;
use crate::passes::handle_result_shapes::calculate_measure_shapes;
use crate::pulse_map::PulseMap;
use crate::result::AwgCodeGenerationResult;
use crate::result::AwgProperties;
use crate::result::ChannelOscillator;
use crate::result::ChannelProperties;
use crate::result::CommandTable;
use crate::result::FeedbackRegisterConfig;
use crate::result::FixedValueOrParameter;
use crate::result::Gains;
use crate::result::InputChannelProperties;
use crate::result::IntegratorAllocation;
use crate::result::Measurement;
use crate::result::PpcSettings;
use crate::result::SeqCGenOutput;
use crate::result::SeqCProgram;
use crate::result::SequencerType;
use crate::result::SignalIntegrationInfo;
use crate::sample_waveforms::ProcessedWaveforms;
use crate::sample_waveforms::{WaveDeclaration, collect_and_finalize_waveforms};
use crate::sampled_event_handler::SeqcResults;
use crate::sampled_event_handler::handle_sampled_events;
use crate::sampled_event_handler::seqc_tracker::FeedbackRegisterIndex;
use crate::sampled_event_handler::seqc_tracker::awg::Awg;
use crate::tracing_utils::ParallelTraceContext;
use crate::waveform::CodegenWaveform;
use crate::waveform::WaveKey;
use crate::waveform::WaveformStore;
use crate::waveform_sampler::SampleWaveforms;

use laboneq_common::types::ChannelKey;
use laboneq_dsl::signal_calibration::MixerCalibration;
use laboneq_error::LabOneQError;
use laboneq_error::WithContext;
use laboneq_error::resource_usage::ResourceExhaustionError;
use laboneq_error::resource_usage::handle_resource_exhaustion;
use laboneq_error::resource_usage::intercept_and_collect;
use laboneq_error::{bail, laboneq_error};
use laboneq_log::warn;
use laboneq_units::tinysample::{tiny_samples, tinysamples_to_seconds};
use rayon::prelude::*;
use std::collections::HashMap;
use tracing::instrument;

#[allow(clippy::too_many_arguments)]
fn generate_output(
    node: IrNode,
    awg: &AwgCore,
    wave_declarations: &[WaveDeclaration],
    qa_signals_by_handle: &HashMap<Handle, (SignalUid, AwgKey)>,
    has_readout_feedback: bool,
    feedback_register: &Option<FeedbackRegisterIndex>,
    ctx: &CodeGenContext,
) -> Result<SeqcResults> {
    let sampled_events = generate_event_list(node, awg, ctx)?;

    let shf_output_mute_min_duration = if awg.output_mute_enable {
        Some(ctx.settings.shf_output_mute_min_duration)
    } else {
        None
    };
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
        ctx.settings.emit_timing_comments,
        has_readout_feedback,
        &ctx.acquisition_type,
    )?;
    Ok(seqc_results)
}

fn generate_code_for_awg<T: SampleWaveforms + SampleIntegrationKernels>(
    root: &IrNode,
    awg: &AwgCore,
    ctx: &CodeGenContext,
    sampler: &T,
) -> Result<AwgCodeGenerationResult> {
    let root = fanout_for_awg(root, awg);
    let awg_info = analyze_awg_ir(&root, awg)?;
    let measurement_info = analyze_measurements(&root, awg.device_kind(), awg.sampling_rate)?;
    let measurement_shapes =
        calculate_measure_shapes(&root, awg, &measurement_info.integration_lengths, ctx)?;
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
    let waveforms = collect_and_finalize_waveforms(&mut awg_node, sampler, awg)?;
    let processed_kernels = collect_and_process_integration_kernels(&awg_node, awg, sampler, ctx)?;
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

    // Feedback registers
    let target_feedback_register = ctx.feedback_config.target_feedback_register(&awg.key());
    let global_feedback_register = match target_feedback_register {
        Some(FeedbackRegisterAllocation::Global { register }) => (*register as u32).into(),
        _ => None,
    };

    let awg_events = generate_output(
        awg_node,
        awg,
        &waveforms.wave_declarations,
        &qa_signals_by_handle,
        awg_info.has_readout_feedback(),
        &global_feedback_register,
        ctx,
    )?;
    let output = construct_awg_result(
        awg,
        awg_events,
        waveforms,
        processed_kernels,
        measurement_info,
        measurement_shapes,
        &awg_info,
        ctx,
    )?;
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
fn construct_awg_result(
    awg: &AwgCore,
    awg_events: SeqcResults,
    processed_waveforms: ProcessedWaveforms,
    processed_kernels: ProcessedKernels,
    measurement_info: MeasurementAnalysis,
    measurement_shapes: Option<AwgMeasurementShapes>,
    awg_info: &AwgCompilationInfo,
    ctx: &CodeGenContext,
) -> Result<AwgCodeGenerationResult> {
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

    let waveform_store = WaveformStore::merge([
        processed_kernels.waveform_store,
        processed_waveforms.waveform_store,
    ])?;

    let pulse_map = processed_waveforms.pulse_map;
    let mut waveforms = processed_kernels.waveforms;
    waveforms.extend(processed_waveforms.waveforms);

    let mut long_readout_signals = processed_kernels.long_readout_signals;
    long_readout_signals.extend(processed_waveforms.long_readout_signals);

    // Construct channel properties
    let this_awg_initial_properties = awg
        .signals
        .iter()
        .map(|s| {
            ctx.signal_properties(s.uid).unwrap_or_else(|| {
                panic!(
                    "Internal Error: Missing initial signal properties for signal {:?}",
                    s.uid
                )
            })
        })
        .collect::<Vec<_>>();

    let mut channel_properties = vec![];
    for signal in awg.signals.iter().filter(|s| s.is_output()) {
        let signal_properties = this_awg_initial_properties
            .iter()
            .find(|s| s.uid == signal.uid)
            .unwrap();

        let awg_channels = awg.awg_channels_for_signal(&signal.uid).unwrap();
        let base_channel = awg_channels
            .iter()
            .min()
            .expect("Internal error: Signal has no channels");
        let requires_long_readout = long_readout_signals.contains(&signal.uid);

        for channel in awg_channels.iter() {
            let relative_channel = channel - base_channel;
            let mixer_props = calculate_mixer_properties(
                &signal_properties.mixer_calibration,
                relative_channel,
                awg.device_kind(),
                &signal.kind,
            );

            let voltage_offset = match signal.kind {
                SignalKind::IQ => mixer_props.voltage_offset,
                _ => signal_properties
                    .voltage_offset
                    .clone()
                    .unwrap_or(FixedValueOrParameter::Value(0.0)),
            };

            channel_properties.push(ChannelProperties {
                signal: signal.uid,
                channel: *channel,
                marker_mode: awg_info.marker_modes.get(channel).cloned(),
                scheduler_delay: measurement_info
                    .delays
                    .get(&signal.uid)
                    .map(|d| d.delay_port())
                    .unwrap_or_default(),
                amplitude: signal_properties.amplitude.clone(),
                voltage_offset,
                gains: mixer_props.gains,
                port_mode: signal_properties.port_mode.clone(),
                port_delay: signal_properties.port_delay.clone(),
                range: signal_properties.range.clone(),
                lo_frequency: signal_properties.lo_frequency.clone(),
                routed_outputs: signal_properties.routed_outputs.clone(),
                output_mute_enable: awg.output_mute_enable,
                oscillator: awg
                    .oscillator_index(&signal.uid)
                    .map(|index| ChannelOscillator {
                        uid: signal.oscillator.as_ref().unwrap().uid.clone(),
                        index,
                        frequency: signal_properties
                            .oscillator_frequency
                            .clone()
                            .expect("Expected a frequency for hardware oscillator"),
                    }),
                requires_long_readout,
            });
        }
    }

    // Construct input channel properties
    let mut input_channel_properties = vec![];
    for signal in awg.signals.iter().filter(|s| !s.is_output()) {
        let signal_properties = this_awg_initial_properties
            .iter()
            .find(|s| s.uid == signal.uid)
            .unwrap();

        let awg_channels = awg.awg_channels_for_signal(&signal.uid).unwrap();
        let requires_long_readout = long_readout_signals.contains(&signal.uid);

        for channel in awg_channels.iter() {
            input_channel_properties.push(InputChannelProperties {
                signal: signal.uid,
                channel: *channel,
                scheduler_delay: measurement_info
                    .delays
                    .get(&signal.uid)
                    .map(|d| d.delay_port())
                    .unwrap_or_default(),
                port_mode: signal_properties.port_mode.clone(),
                port_delay: signal_properties.port_delay.clone(),
                range: signal_properties.range.clone(),
                lo_frequency: signal_properties.lo_frequency.clone(),
                oscillator: awg
                    .oscillator_index(&signal.uid)
                    .map(|index| ChannelOscillator {
                        uid: signal.oscillator.as_ref().unwrap().uid.clone(),
                        index,
                        frequency: signal_properties
                            .oscillator_frequency
                            .clone()
                            .expect("Expected a frequency for hardware oscillator"),
                    }),
                requires_long_readout,
            });
        }
    }

    // Post process channel properties
    resolve_port_mode(awg, &mut channel_properties, &mut input_channel_properties)?;

    let awg_properties = AwgProperties {
        key: awg.key(),
        kind: awg.kind,
    };
    let result = AwgCodeGenerationResult {
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
        shf_sweeper_config: awg_events.shf_sweeper_config,
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
        integration_weights: processed_kernels.weights,
        waveforms,
        waveform_store,
        pulse_map,
        integrator_allocations: construct_integration_allocations(awg, ctx),
        result_length: measurement_shapes
            .as_ref()
            .and_then(|shapes| shapes.result_length),
        result_handle_maps: measurement_shapes
            .map(|shapes| shapes.result_handle_maps)
            .unwrap_or_default(),
    };
    Ok(result)
}

fn resolve_port_mode(
    awg: &AwgCore,
    outputs: &mut [ChannelProperties],
    inputs: &mut [InputChannelProperties],
) -> Result<()> {
    if !matches!(awg.device_kind(), DeviceKind::SHFQA) {
        return Ok(());
    }
    for input in inputs.iter_mut() {
        let Some(output) = outputs.iter_mut().find(|o| o.channel == input.channel) else {
            continue;
        };
        if let (Some(input_port_mode), Some(output_port_mode)) =
            (&input.port_mode, &output.port_mode)
            && input_port_mode != output_port_mode
        {
            bail!(
                "Mismatch between input and output port mode on channel {}",
                input.channel
            );
        }
        input.port_mode = input.port_mode.clone().or_else(|| output.port_mode.clone());
        output.port_mode = output.port_mode.clone().or_else(|| input.port_mode.clone());
    }
    Ok(())
}

/// Mixer calibration properties for a single channel
#[derive(Debug, Clone)]
struct MixerChannelProperties {
    voltage_offset: FixedValueOrParameter<f64>,
    gains: Option<Gains>,
}

/// Calculate mixer calibration properties for a channel with device-specific defaults
fn calculate_mixer_properties(
    mixer_calibration: &Option<MixerCalibration>,
    relative_channel: ChannelIndex,
    device_kind: &DeviceKind,
    signal_kind: &SignalKind,
) -> MixerChannelProperties {
    // For non-IQ signals, use defaults.
    // For IQ signals, if mixer calibration is provided, use it, otherwise use defaults.
    // Currently we do not error on mixer calibration on non-IQ signals, but simply ignore it.
    let mixer_calibration = if !matches!(signal_kind, SignalKind::IQ) {
        None
    } else {
        mixer_calibration.as_ref().or(None)
    };
    assert!(
        relative_channel == 0 || relative_channel == 1,
        "Only relative channels 0 (I) and 1 (Q) are supported"
    );
    let is_i_channel = relative_channel == 0;

    // Voltage offsets
    let voltage_offset = mixer_calibration
        .as_ref()
        .and_then(|mc| {
            if is_i_channel {
                mc.voltage_offset_i
            } else {
                mc.voltage_offset_q
            }
        })
        .map(value_or_parameter_to_fixed)
        .unwrap_or(FixedValueOrParameter::Value(0.0));

    // Gains
    let gains = mixer_calibration
        .as_ref()
        .and_then(|mc| {
            let (diagonal, off_diagonal) = if is_i_channel {
                mc.gains_i()
            } else {
                mc.gains_q()
            }?;
            Some(Gains {
                diagonal: value_or_parameter_to_fixed(diagonal),
                off_diagonal: value_or_parameter_to_fixed(off_diagonal),
            })
        })
        .or({
            // Apply device-specific defaults for HDAWG when values are missing
            if matches!(device_kind, DeviceKind::HDAWG) {
                Some(Gains {
                    diagonal: FixedValueOrParameter::Value(1.0),
                    off_diagonal: FixedValueOrParameter::Value(0.0),
                })
            } else {
                None
            }
        });

    MixerChannelProperties {
        voltage_offset,
        gains,
    }
}

fn generate_code_for_multiple_awgs<T: SampleWaveforms + SampleIntegrationKernels + Sync + Send>(
    root: &IrNode,
    awgs: &[AwgCore],
    ctx: &CodeGenContext,
    waveform_sampler: &T,
) -> Result<Vec<AwgCodeGenerationResult>> {
    let trace_ctx = ParallelTraceContext::new();
    let awg_results: Vec<Result<AwgCodeGenerationResult>> = awgs
        .par_iter()
        .map(|awg| -> Result<AwgCodeGenerationResult> {
            par_trace!(trace_ctx, "generate_code_for_awg", {
                generate_code_for_awg(root, awg, ctx, waveform_sampler).with_context(|| {
                    format!(
                        "Error while generating code for signals: {}",
                        awg.signals
                            .iter()
                            .map(|s| s.uid.0.to_string())
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                })
            })
        })
        .collect::<Vec<_>>();
    intercept_and_collect(awg_results.into_iter())
}

fn estimate_total_execution_time(root: &IrNode) -> f64 {
    tinysamples_to_seconds(tiny_samples(root.data().length())).into()
}

#[instrument(name = "laboneq.compiler.generate-code-rs", skip_all)] // Named with `-rs` suffix to distinguish from Python wrapper function
pub fn generate_code<T: SampleWaveforms + SampleIntegrationKernels + Sync + Send>(
    codegen_ir: CodegenIr,
    sampler: &T,
    mut settings: CodeGeneratorSettings,
) -> Result<SeqCGenOutput> {
    validate_codegen_ir(&codegen_ir)?;

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
    let feedback_config = collect_feedback_config(root, &awgs)
        .with_context(|| "Error while processing feedback configuration")?;
    let integration_unit_allocation = allocate_integration_units(root, &awgs, &acquisition_type)?;
    let feedback_register_layout =
        calculate_feedback_register_layout(&awgs, &integration_unit_allocation, &feedback_config);
    let mut ctx = CodeGenContext {
        acquisition_type,
        averaging_mode: codegen_ir.averaging_mode,
        averaging_count: codegen_ir.averaging_count,
        feedback_register_layout,
        feedback_config,
        integration_unit_allocation,
        settings,
        initial_signal_properties: codegen_ir.initial_signal_properties,
    };

    // Code generation per AWG
    let mut awg_results = generate_code_for_multiple_awgs(root, &awgs, &ctx, sampler)?;

    // Resource usage evaluation
    handle_resource_exhaustion(
        evaluate_resource_usage(&awg_results),
        ctx.settings.ignore_resource_exhaustion,
    )?;

    // Result construction
    let total_execution_time = estimate_total_execution_time(root);
    let measurements = evaluate_measurement_per_device(&awg_results);
    let ppc_settings =
        merge_ppc_steps(&awgs, &mut awg_results, &mut ctx.initial_signal_properties)?;
    let (waveforms, waveform_store) = merge_waveforms(&mut awg_results)?;
    let pulse_map = merge_pulse_maps(&mut awg_results);

    let result = SeqCGenOutput {
        device_properties: codegen_ir.awg_devices,
        auxiliary_device_properties: codegen_ir.auxiliary_devices,
        awg_results,
        total_execution_time,
        measurements,
        ppc_settings,
        waveforms,
        waveform_store,
        pulse_map,
    };
    Ok(result)
}

fn merge_pulse_maps(awg_results: &mut [AwgCodeGenerationResult]) -> PulseMap {
    let pulse_maps = awg_results
        .iter_mut()
        .map(|awg_result| std::mem::take(&mut awg_result.pulse_map))
        .collect::<Vec<_>>();
    PulseMap::merge(pulse_maps)
}

/// Merge waveforms across AWGs.
///
/// Drains the waveforms from the AWG results and merges them based on their key.
/// Checks for consistency of waveforms with the same key across different AWGs, and returns an error if inconsistent waveforms are found. Otherwise, returns a unique set of waveforms for output.
fn merge_waveforms(
    awg_results: &mut [AwgCodeGenerationResult],
) -> Result<(Vec<CodegenWaveform>, WaveformStore)> {
    let stores = awg_results
        .iter_mut()
        .map(|r| std::mem::take(&mut r.waveform_store));
    let waveform_store = WaveformStore::merge(stores)?;

    let mut waveform_map: IndexMap<WaveKey, CodegenWaveform> =
        IndexMap::with_capacity(awg_results.len());
    for awg_result in awg_results.iter_mut() {
        for waveform in awg_result.waveforms.drain(..) {
            if let Some(existing_waveform) = waveform_map.get(waveform.wave_key()) {
                if existing_waveform != &waveform {
                    return Err(laboneq_error!(
                        "Internal error: Inconsistent waveforms with the same key '{}' across different AWGs.",
                        waveform.wave_key().signature
                    ));
                }
            } else {
                waveform_map.insert(waveform.wave_key().clone(), waveform);
            }
        }
    }
    Ok((waveform_map.into_values().collect(), waveform_store))
}

/// Merge the PPC steps across signals and AWGs, ensuring consistency of configurations for signals sharing a PPC channel.
///
/// Drains the PPC configurations from the AWG results and initial signal properties, and merges them based on the assigned PPC channels.
/// Checks for consistency of configurations across signals sharing the same PPC channel, and returns an error if incompatible configurations are found.
/// Merge waveforms across AWGs by their keys, ensuring consistency for waveforms with the same key, and return a unique set of waveforms for output.
fn merge_ppc_steps(
    awgs: &[AwgCore],
    awg_results: &mut [AwgCodeGenerationResult],
    initial_signal_properties: &mut [InitialSignalProperties],
) -> Result<Vec<PpcSettings>> {
    let signal_to_ppc_channel = initial_signal_properties
        .iter()
        .filter_map(|signal| {
            signal
                .ppc_settings
                .as_ref()
                .map(|ppc| (signal.uid, ppc.ppc_channel))
        })
        .collect::<HashMap<_, _>>();

    let mut awg_to_ppc_channel = HashMap::<AwgKey, ChannelKey>::new();
    for awg in awgs.iter() {
        for signal in awg.signals.iter() {
            if let Some(ppc_channel) = signal_to_ppc_channel.get(&signal.uid) {
                awg_to_ppc_channel.insert(awg.key(), *ppc_channel);
            }
        }
    }

    // Drain the PPC configurations from the AWG results (real-time) and merge them by their assigned PPC channels, checking for consistency.
    let mut config_by_channel =
        awg_results
            .iter_mut()
            .filter_map(|awg| {
                awg.shf_sweeper_config.take().map(|ppc_config| (awg, ppc_config))
            })
            .try_fold(HashMap::<ChannelKey, String>::new(), |mut acc, (awg, ppc_config)| {
                let ppc_channel = awg_to_ppc_channel.get(&awg.awg.key).ok_or_else(|| {
                    laboneq_error!("Internal Error: Missing PPC channel assignment for AWG with SHFPPC config")
                })?;
                // Check for compatibility with existing config for this device, and merge if compatible.
                if let Some(existing_config) = acc.get(ppc_channel) {
                    if existing_config != &ppc_config {
                        return Err(laboneq_error!("Incompatible configurations for PPC channel '{}' across different signals.", ppc_channel));
                    }
                } else {
                    acc.insert(*ppc_channel, ppc_config);
                }
                Ok(acc)
            })?;

    // Drain the PPC configurations from the signal properties (near-time) and merge them by their assigned PPC channels, checking for consistency.
    let ppc_settings = initial_signal_properties
        .iter_mut()
        .filter_map(|signal| signal.ppc_settings.take())
        .try_fold(HashMap::new(), |mut ppc_settings, ppc_config| {
            let key = ppc_config.ppc_channel;
            if let Some(existing_config) = ppc_settings.get(&key)
                && existing_config != &ppc_config
            {
                Err(laboneq_error!(
                    "Incompatible configurations for PPC channel '{}' across different signals.",
                    key
                ))
            } else {
                ppc_settings.insert(key, ppc_config);
                Ok(ppc_settings)
            }
        })?;

    // Build the merged results
    let mut ppc_settings: Vec<PpcSettings> = ppc_settings
        .into_iter()
        .map(|(key, mut config)| {
            config.sweep_config = config_by_channel.remove(&key);
            config
        })
        .collect();

    // Sorted for output consistency
    ppc_settings.sort_by_key(|settings| settings.ppc_channel);
    Ok(ppc_settings)
}

/// Evaluate the measurements per device.
///
/// The measurements are grouped by device and channel, and the maximum length
/// of the measurements is taken for each channel.
fn evaluate_measurement_per_device(awg_results: &[AwgCodeGenerationResult]) -> Vec<Measurement> {
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
    measurements.sort_by_key(|a| a.channel);
    measurements
}

fn evaluate_resource_usage(
    awg_results: &[AwgCodeGenerationResult],
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

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
use crate::ir::Samples;
use crate::ir::SignalUid;
use crate::ir::compilation_job::AwgCore;
use crate::ir::compilation_job::AwgKey;
use crate::ir::compilation_job::ChannelIndex;
use crate::ir::compilation_job::InitialSignalProperties;
use crate::ir::compilation_job::Signal;
use crate::ir::compilation_job::SignalKind;
use crate::ir_adapter::value_or_parameter_to_fixed;
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
use crate::result::SignalIntegrationInfo;
use crate::sample_waveforms::ProcessedWaveforms;
use crate::sample_waveforms::{WaveDeclaration, collect_and_finalize_waveforms};
use crate::sampled_event_handler::SeqcResults;
use crate::sampled_event_handler::handle_sampled_events;
use crate::sampled_event_handler::seqc_tracker::FeedbackRegisterIndex;
use crate::sampled_event_handler::seqc_tracker::awg::Awg;
use crate::waveform::CodegenWaveform;
use crate::waveform::WaveKey;
use crate::waveform::WaveformStore;
use crate::waveform_sampler::SampleWaveforms;
use codegenerator_utils::par_trace;
use codegenerator_utils::tracing::ParallelTraceContext;

use laboneq_common::types::ChannelKey;
use laboneq_dsl::signal_calibration::MixerCalibration;
use laboneq_error::LabOneQError;
use laboneq_error::WithContext;
use laboneq_error::resource_usage::ResourceExhaustionError;
use laboneq_error::resource_usage::handle_resource_exhaustion;
use laboneq_error::resource_usage::intercept_and_collect;
use laboneq_error::{bail, laboneq_error};
use laboneq_log::warn;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use tracing::instrument;

#[instrument(level = "debug", name = "laboneq.compiler.generate-code-rs", skip_all)] // Named with `-rs` suffix to distinguish from Python wrapper function
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

    let mut awg_results = generate_code_for_multiple_awgs(root, &awgs, &ctx, sampler)?;
    handle_resource_exhaustion(
        evaluate_resource_usage(&awg_results),
        ctx.settings.ignore_resource_exhaustion,
    )?;

    let total_execution_time = awg_results
        .iter()
        .map(|r| r.total_execution_time)
        .fold(0.0, f64::max);
    let measurements = evaluate_measurements_per_awg(&awg_results);
    let ppc_settings =
        merge_ppc_steps(&awgs, &mut awg_results, &mut ctx.initial_signal_properties)?;
    let (waveforms, waveform_store) = merge_waveforms(&mut awg_results)?;
    let pulse_map = merge_pulse_maps(&mut awg_results);

    Ok(SeqCGenOutput {
        device_properties: codegen_ir.awg_devices,
        auxiliary_device_properties: codegen_ir.auxiliary_devices,
        awg_results,
        total_execution_time,
        measurements,
        ppc_settings,
        waveforms,
        waveform_store,
        pulse_map,
    })
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
            par_trace!(trace_ctx, tracing::Level::TRACE, "generate_code_for_awg", {
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
    intercept_and_collect(awg_results)
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
        &awg.signals
            .iter()
            .filter_map(|signal| {
                let delay = measurement_info.delays.get(signal)?;
                Some((signal.uid, delay.delay_sequencer()))
            })
            .collect(),
    )?;
    let waveforms = collect_and_finalize_waveforms(&mut awg_node, sampler, awg)?;
    let kernels = collect_and_process_integration_kernels(&awg_node, awg, sampler, ctx)?;
    let feedback_registers = FeedbackRegisters::resolve(awg, &awg_info, ctx);

    let awg_events = generate_output(
        awg_node,
        awg,
        &waveforms.wave_declarations,
        awg_info.has_readout_feedback(),
        feedback_registers.global_target(),
        ctx,
    )?;
    construct_awg_result(
        awg,
        AwgArtifacts {
            awg_events,
            waveforms,
            kernels,
            measurement_info,
            measurement_shapes,
            awg_info,
            feedback_registers,
        },
        ctx,
    )
}

/// The per-AWG passes' output, ready to be assembled into the AWG result.
struct AwgArtifacts {
    awg_events: SeqcResults,
    waveforms: ProcessedWaveforms,
    kernels: ProcessedKernels,
    measurement_info: MeasurementAnalysis,
    measurement_shapes: Option<AwgMeasurementShapes>,
    awg_info: AwgCompilationInfo,
    feedback_registers: FeedbackRegisters,
}

/// The feedback registers an AWG writes to and reads from.
struct FeedbackRegisters {
    /// Register this AWG publishes its discrimination results into.
    target: Option<FeedbackRegisterAllocation>,
    /// Register that is published into by the AWG from which this one receives feedback.
    source: Option<FeedbackRegisterAllocation>,
}

impl FeedbackRegisters {
    fn resolve(awg: &AwgCore, awg_info: &AwgCompilationInfo, ctx: &CodeGenContext) -> Self {
        let target = ctx.feedback_config.target_feedback_register(&awg.key());
        let source = awg_info.feedback_handles().first().and_then(|handle| {
            let source = ctx
                .feedback_config
                .feedback_source(handle)
                .expect("Internal Error: Missing feedback source for handle");
            ctx.feedback_config
                .target_feedback_register(&source.awg_key)
        });
        FeedbackRegisters {
            target: target.cloned(),
            source: source.cloned(),
        }
    }

    /// Index of the global register this AWG publishes into, if it uses one.
    fn global_target(&self) -> Option<FeedbackRegisterIndex> {
        match self.target {
            Some(FeedbackRegisterAllocation::Global { register }) => {
                Some(register as FeedbackRegisterIndex)
            }
            _ => None,
        }
    }

    /// Registers as reported to the controller, where -1 marks the local register.
    fn register_ids(&self) -> (Option<i64>, Option<i64>) {
        fn id(allocation: &FeedbackRegisterAllocation) -> i64 {
            match allocation {
                FeedbackRegisterAllocation::Local => -1,
                FeedbackRegisterAllocation::Global { register } => *register as i64,
            }
        }
        (self.target.as_ref().map(id), self.source.as_ref().map(id))
    }

    fn targets_local_register(&self) -> bool {
        matches!(self.target, Some(FeedbackRegisterAllocation::Local))
    }
}

fn generate_output(
    node: IrNode,
    awg: &AwgCore,
    wave_declarations: &[WaveDeclaration],
    has_readout_feedback: bool,
    feedback_register: Option<FeedbackRegisterIndex>,
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
    handle_sampled_events(
        sampled_events,
        &awg,
        &ctx.feedback_config,
        wave_declarations,
        feedback_register,
        &ctx.feedback_register_layout,
        ctx.settings.emit_timing_comments,
        has_readout_feedback,
        &ctx.acquisition_type,
    )
}

fn construct_awg_result(
    awg: &AwgCore,
    artifacts: AwgArtifacts,
    ctx: &CodeGenContext,
) -> Result<AwgCodeGenerationResult> {
    let AwgArtifacts {
        awg_events,
        waveforms: processed_waveforms,
        kernels,
        measurement_info,
        measurement_shapes,
        awg_info,
        feedback_registers,
    } = artifacts;

    let waveform_store =
        WaveformStore::merge([kernels.waveform_store, processed_waveforms.waveform_store])?;

    let pulse_map = processed_waveforms.pulse_map;
    let mut waveforms = kernels.waveforms;
    waveforms.extend(processed_waveforms.waveforms);

    let mut long_readout_signals = kernels.long_readout_signals;
    long_readout_signals.extend(processed_waveforms.long_readout_signals);

    let (mut channel_properties, mut input_channel_properties) = build_channel_properties(
        awg,
        &awg_info,
        &measurement_info,
        &long_readout_signals,
        ctx,
    );
    resolve_port_mode(awg, &mut channel_properties, &mut input_channel_properties)?;

    let total_execution_time =
        estimate_total_execution_time(awg_events.program_length, awg.sampling_rate);
    let (target_feedback_register, source_feedback_register) = feedback_registers.register_ids();
    let command_table = awg_events.command_table.map(|results| CommandTable {
        src: serde_json::to_string(&results.command_table).unwrap(),
        n_entries: results.n_entries,
        max_entries: awg
            .device_kind()
            .traits()
            .max_ct_entries
            .expect("Internal Error: Command table emitted for a device without command tables")
            as usize,
        parameter_phase_increment_map: results.parameter_phase_increment_map,
    });
    Ok(AwgCodeGenerationResult {
        awg: AwgProperties {
            key: awg.key(),
            kind: awg.kind,
        },
        total_execution_time,
        seqc: create_seqc_program(awg, awg_events.seqc),
        wave_indices: awg_events.wave_indices,
        command_table,
        shf_sweeper_config: awg_events.shf_sweeper_config,
        integration_lengths: measurement_info
            .integration_lengths
            .into_iter()
            .map(|x| SignalIntegrationInfo {
                signal: *x.signal(),
                is_play: x.is_play(),
                length: x.duration(),
            })
            .collect(),
        feedback_register_config: FeedbackRegisterConfig {
            local: feedback_registers.targets_local_register(),
            source_feedback_register,
            register_index_select: awg_events.feedback_register_config.register_index_select,
            codeword_bitshift: awg_events.feedback_register_config.codeword_bitshift,
            codeword_bitmask: awg_events.feedback_register_config.codeword_bitmask,
            command_table_offset: awg_events.feedback_register_config.command_table_offset,
            target_feedback_register,
        },
        output_channel_properties: channel_properties,
        input_channel_properties,
        integration_weights: kernels.weights,
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
    })
}

/// Construct the per-channel properties of an AWG's output and input signals.
fn build_channel_properties(
    awg: &AwgCore,
    awg_info: &AwgCompilationInfo,
    measurement_info: &MeasurementAnalysis,
    long_readout_signals: &HashSet<SignalUid>,
    ctx: &CodeGenContext,
) -> (Vec<ChannelProperties>, Vec<InputChannelProperties>) {
    let default_mixer_gains =
        awg.device_kind()
            .traits()
            .default_mixer_gains
            .map(|(diagonal, off_diagonal)| Gains {
                diagonal: FixedValueOrParameter::Value(diagonal),
                off_diagonal: FixedValueOrParameter::Value(off_diagonal),
            });
    let mut outputs = vec![];
    let mut inputs = vec![];
    for signal in awg.signals.iter() {
        let properties = ctx.signal_properties(signal.uid).unwrap_or_else(|| {
            panic!(
                "Internal Error: Missing initial signal properties for signal {:?}",
                signal.uid
            )
        });
        let channels = awg
            .awg_channels_for_signal(&signal.uid)
            .expect("Internal Error: Signal is not mapped to any AWG channel");
        let base_channel = channels
            .iter()
            .min()
            .expect("Internal error: Signal has no channels.");
        // The scheduler's delay alone. Output channels add the signal's alignment delay.
        let scheduler_delay = measurement_info
            .delays
            .get(signal)
            .map(|d| d.delay_port())
            .unwrap_or_default();
        let oscillator = channel_oscillator(awg, signal, properties);
        let requires_long_readout = long_readout_signals.contains(&signal.uid);

        for &channel in channels.iter() {
            if !signal.is_output() {
                inputs.push(InputChannelProperties {
                    signal: signal.uid,
                    channel,
                    scheduler_delay,
                    port_mode: properties.port_mode.clone(),
                    port_delay: properties.port_delay.clone(),
                    range: properties.range.clone(),
                    lo_frequency: properties.lo_frequency.clone(),
                    oscillator: oscillator.clone(),
                    requires_long_readout,
                });
                continue;
            }
            let mixer_props = calculate_mixer_properties(
                properties.mixer_calibration.as_ref(),
                IqChannel::from_relative_channel(channel - base_channel),
                &signal.kind,
                default_mixer_gains.as_ref(),
            );
            outputs.push(ChannelProperties {
                signal: signal.uid,
                channel,
                marker_mode: awg_info.marker_modes.get(&channel).cloned(),
                scheduler_delay: scheduler_delay + properties.signal_alignment_delay,
                precompensation: properties.precompensation.clone(),
                amplitude: properties.amplitude.clone(),
                voltage_offset: match signal.kind {
                    SignalKind::IQ => mixer_props.voltage_offset,
                    _ => properties
                        .voltage_offset
                        .clone()
                        .unwrap_or(FixedValueOrParameter::Value(0.0)),
                },
                gains: mixer_props.gains,
                port_mode: properties.port_mode.clone(),
                port_delay: properties.port_delay.clone(),
                range: properties.range.clone(),
                lo_frequency: properties.lo_frequency.clone(),
                routed_outputs: properties.routed_outputs.clone(),
                output_mute_enable: awg.output_mute_enable,
                oscillator: oscillator.clone(),
                requires_long_readout,
            });
        }
    }
    (outputs, inputs)
}

/// The hardware oscillator driving a signal, if one is allocated to it.
fn channel_oscillator(
    awg: &AwgCore,
    signal: &Signal,
    properties: &InitialSignalProperties,
) -> Option<ChannelOscillator> {
    let index = awg.oscillator_index(&signal.uid)?;
    let oscillator = signal
        .oscillator
        .as_ref()
        .expect("Internal Error: Oscillator index allocated to a signal without an oscillator");
    Some(ChannelOscillator {
        uid: oscillator.uid.clone(),
        index,
        frequency: properties
            .oscillator_frequency
            .clone()
            .expect("Expected a frequency for hardware oscillator"),
    })
}

/// The two channels carrying an IQ signal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IqChannel {
    I,
    Q,
}

impl IqChannel {
    fn from_relative_channel(relative_channel: ChannelIndex) -> Self {
        match relative_channel {
            0 => IqChannel::I,
            1 => IqChannel::Q,
            _ => panic!("Only relative channels 0 (I) and 1 (Q) are supported"),
        }
    }
}

/// Mixer calibration properties for a single channel.
#[derive(Debug, Clone)]
struct MixerChannelProperties {
    voltage_offset: FixedValueOrParameter<f64>,
    gains: Option<Gains>,
}

/// Calculate mixer calibration properties for a channel, falling back to the
/// given defaults where the calibration leaves a value unset.
fn calculate_mixer_properties(
    mixer_calibration: Option<&MixerCalibration>,
    channel: IqChannel,
    signal_kind: &SignalKind,
    default_gains: Option<&Gains>,
) -> MixerChannelProperties {
    // Currently we do not error on mixer calibration on non-IQ signals, but simply ignore it.
    let mixer_calibration = match signal_kind {
        SignalKind::IQ => mixer_calibration,
        _ => None,
    };
    let voltage_offset = mixer_calibration
        .and_then(|mc| match channel {
            IqChannel::I => mc.voltage_offset_i,
            IqChannel::Q => mc.voltage_offset_q,
        })
        .map(value_or_parameter_to_fixed)
        .unwrap_or(FixedValueOrParameter::Value(0.0));

    let gains = mixer_calibration
        .and_then(|mc| {
            let (diagonal, off_diagonal) = match channel {
                IqChannel::I => mc.gains_i(),
                IqChannel::Q => mc.gains_q(),
            }?;
            Some(Gains {
                diagonal: value_or_parameter_to_fixed(diagonal),
                off_diagonal: value_or_parameter_to_fixed(off_diagonal),
            })
        })
        .or_else(|| default_gains.cloned());

    MixerChannelProperties {
        voltage_offset,
        gains,
    }
}

/// Reconcile the port mode of input and output channels sharing a physical port.
fn resolve_port_mode(
    awg: &AwgCore,
    outputs: &mut [ChannelProperties],
    inputs: &mut [InputChannelProperties],
) -> Result<()> {
    if !awg.device_kind().traits().shares_input_output_port {
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

fn construct_integration_allocations(
    awg: &AwgCore,
    ctx: &CodeGenContext,
) -> Vec<IntegratorAllocation> {
    awg.signals
        .iter()
        .filter_map(|s| ctx.integration_unit_allocation_for_signal(s.uid))
        .map(|alloc| {
            let props = ctx.signal_properties(alloc.signal).unwrap_or_else(|| {
                panic!(
                    "Internal Error: Missing initial signal properties for signal {:?}",
                    alloc.signal
                )
            });
            let thresholds = if props.thresholds.is_empty() {
                // The number of thresholds needed for MSD is n*(n-1)/2, where
                // n is the number of states.
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
        })
        .collect::<Vec<_>>()
}

fn create_seqc_program(awg: &AwgCore, seqc_code: String) -> SeqCProgram {
    let traits = awg.device_kind().traits();
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
        sequencer: traits.sequencer_type,
        sampling_rate: traits
            .seqc_requires_sampling_rate
            .then_some(awg.sampling_rate),
    }
}

/// Estimate the real-time execution time of an emitted AWG program, in seconds.
fn estimate_total_execution_time(program_length: Samples, sampling_rate: f64) -> f64 {
    program_length as f64 / sampling_rate
}

fn evaluate_resource_usage(
    awg_results: &[AwgCodeGenerationResult],
) -> impl Iterator<Item = Result<(), LabOneQError>> + '_ {
    awg_results.iter().map(|result| {
        let Some(ct) = &result.command_table else {
            return Ok(());
        };
        let usage = ct.resource_usage_percentage();
        if usage <= 1.0 {
            return Ok(());
        }
        let msg = format!(
            "Exceeded max number of command table entries on device '{}', AWG({}): Needed: {}, max available: {}",
            result.awg.key.device_name(),
            result.awg.key.index(),
            ct.n_entries,
            ct.max_entries
        );
        Err(ResourceExhaustionError::new(msg, Some(usage)).into())
    })
}

/// Evaluate the measurements per AWG.
///
/// The measurements are grouped by AWG, and the longest acquire integration
/// length is taken for each.
fn evaluate_measurements_per_awg(awg_results: &[AwgCodeGenerationResult]) -> Vec<Measurement> {
    let mut length_by_awg: HashMap<&AwgKey, Samples> = HashMap::new();
    for result in awg_results.iter() {
        if result.integration_lengths.is_empty() {
            continue;
        }
        let max = result
            .integration_lengths
            .iter()
            .filter(|meas| !meas.is_play)
            .map(|meas| meas.length)
            .max()
            .unwrap_or_default();
        length_by_awg
            .entry(&result.awg.key)
            .and_modify(|length| *length = (*length).max(max))
            .or_insert(max);
    }
    let mut measurements = length_by_awg
        .into_iter()
        .map(|(key, length)| Measurement {
            device: key.device_name().clone(),
            channel: key.index(),
            length,
        })
        .collect::<Vec<_>>();
    // Sorted for output consistency
    measurements.sort_by_key(|a| (a.device.clone(), a.channel));
    measurements
}

/// Merge the PPC steps across signals and AWGs, ensuring consistency of
/// configurations for signals sharing a PPC channel.
///
/// Drains the PPC configurations from the AWG results and initial signal
/// properties, and merges them based on the assigned PPC channels. Returns an
/// error if signals sharing a PPC channel carry incompatible configurations.
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

    // Drain the PPC configurations from the AWG results (real-time) and merge
    // them by their assigned PPC channels, checking for consistency.
    let mut config_by_channel = HashMap::<ChannelKey, String>::new();
    for awg in awg_results.iter_mut() {
        let Some(ppc_config) = awg.shf_sweeper_config.take() else {
            continue;
        };
        let ppc_channel = awg_to_ppc_channel.get(&awg.awg.key).ok_or_else(|| {
            laboneq_error!(
                "Internal Error: Missing PPC channel assignment for AWG with SHFPPC config"
            )
        })?;
        // Check for compatibility with the existing config for this device, and
        // merge if compatible.
        if let Some(existing_config) = config_by_channel.get(ppc_channel) {
            if existing_config != &ppc_config {
                return Err(laboneq_error!(
                    "Incompatible configurations for PPC channel '{}' across different signals.",
                    ppc_channel
                ));
            }
        } else {
            config_by_channel.insert(*ppc_channel, ppc_config);
        }
    }

    // Drain the PPC configurations from the signal properties (near-time) and
    // merge them by their assigned PPC channels, checking for consistency.
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

/// Merge waveforms across AWGs.
///
/// Drains the waveforms from the AWG results and merges them based on their
/// key. Returns an error if AWGs carry inconsistent waveforms under the same
/// key, otherwise a unique set of waveforms for output.
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

fn merge_pulse_maps(awg_results: &mut [AwgCodeGenerationResult]) -> PulseMap {
    let pulse_maps = awg_results
        .iter_mut()
        .map(|awg_result| std::mem::take(&mut awg_result.pulse_map))
        .collect::<Vec<_>>();
    PulseMap::merge(pulse_maps)
}

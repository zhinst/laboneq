// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use anyhow::Context;
use codegenerator::CodeGeneratorSettings;
use codegenerator::WaveDeclaration;
use codegenerator::analyze_measurements;
use codegenerator::fanout_for_awg;
use codegenerator::handle_feedback_registers::FeedbackRegisterAllocation;
use codegenerator::handle_feedback_registers::{FeedbackConfig, collect_feedback_config};
use codegenerator::ir::compilation_job::AwgCore;
use codegenerator::ir::compilation_job::AwgKey;
use codegenerator::ir::compilation_job::SignalKind;
use codegenerator::ir::experiment::AcquisitionType;
use codegenerator::ir::experiment::Handle;
use codegenerator::ir::{IrNode, Samples, SectionId};
use codegenerator::tinysample::TINYSAMPLE;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::types::PyList;
use pyo3::wrap_pyfunction;
use rayon::prelude::*;
use sampled_event_handler::AwgEvent;
use sampled_event_handler::AwgEventList;
use sampled_event_handler::FeedbackRegisterLayout;
use sampled_event_handler::SeqcResults;
use sampled_event_handler::awg_events::ChangeHwOscPhase;
use sampled_event_handler::awg_events::EventType;
use sampled_event_handler::awg_events::Iterate;
use sampled_event_handler::awg_events::MatchEvent;
use sampled_event_handler::awg_events::PlayWaveEvent;
use sampled_event_handler::awg_events::PrngSetup;
use sampled_event_handler::awg_events::PushLoop;
use sampled_event_handler::awg_events::QaEvent;
use sampled_event_handler::awg_events::TriggerOutputBit;
use sampled_event_handler::awg_events::sort_events;
use seqc_tracker::FeedbackRegisterIndex;
use seqc_tracker::awg::Awg;
use seqc_tracker::awg::HwOscillator;
use signature::{PulseSignaturePy, WaveformSignaturePy};
use std::collections::HashSet;
use std::vec;
use waveform_sampler::PlayHoldPy;
use waveform_sampler::PlaySamplesPy;
mod py_conversions;
mod waveform_sampler;
use codegenerator::ir::{self, NodeKind};
use codegenerator::{
    AwgWaveforms, analyze_awg_ir, collect_and_finalize_waveforms, collect_integration_kernels,
    transform_ir_to_awg_events,
};
mod common_types;
mod pulse_parameters;
mod result;
mod settings;
mod signature;
mod triggers;
use crate::pulse_parameters::PulseParameters;
use crate::result::FeedbackRegisterConfigPy;
use crate::result::SignalIntegrationInfo;
use crate::settings::code_generator_settings_from_dict;
use codegenerator::ir::experiment::PulseParametersId;
use result::{AwgCodeGenerationResultPy, SampledWaveformPy, SeqCGenOutputPy};

mod error;
use crate::error::Result;
use crate::waveform_sampler::WaveformSamplerPy;
use crate::waveform_sampler::batch_calculate_integration_weights;
use std::collections::HashMap;
use triggers::generate_trigger_states;

struct GeneratorState {
    pub loop_step_starts_added: HashMap<Samples, HashSet<SectionId>>,
    pub loop_step_ends_added: HashMap<Samples, HashSet<SectionId>>,
    pub state: Option<u16>,
}

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
    let mut state = GeneratorState {
        loop_step_starts_added: HashMap::new(),
        loop_step_ends_added: HashMap::new(),
        state: None,
    };
    let mut awg_events = generate_output_recursive(node, awg, &mut state)?;
    sort_events(&mut awg_events);
    generate_trigger_states(&mut awg_events);
    let mut sampled_events = AwgEventList::new();
    // NOTE: Add playwave related events after the rest to mimic the original event insertion order
    // Can be removed once the event insertion order is not important anymore (all events generated in Rust)
    for mut event in awg_events.into_iter() {
        sampled_events
            .entry(event.start)
            .or_default()
            .push(std::mem::take(&mut event));
    }
    let awg = Awg {
        signal_kind: awg.kind.clone(),
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
    let seqc_results = sampled_event_handler::handle_sampled_events(
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

fn generate_output_recursive(
    mut node: IrNode,
    awg: &AwgCore,
    state: &mut GeneratorState,
) -> Result<Vec<AwgEvent>> {
    match node.swap_data(NodeKind::Nop { length: 0 }) {
        NodeKind::PhaseReset(_) => {
            panic!("Internal error: PhaseReset should have been replaced by ResetPhase");
        }
        NodeKind::PlayWave(mut ob) => {
            let hw_osc = if let Some(osc_name) = ob.oscillator.take() {
                let index = *awg
                    .osc_allocation
                    .get(&osc_name)
                    .expect("Internal error: Missing hardware oscillator allocation");
                let out = HwOscillator {
                    uid: osc_name,
                    index,
                };
                Some(out)
            } else {
                None
            };
            let end = node.offset() + ob.length();
            let event = AwgEvent {
                start: *node.offset(),
                end,
                kind: EventType::PlayWave(PlayWaveEvent::from_ir(ob, state.state, hw_osc)),
            };
            Ok(vec![event])
        }
        NodeKind::PlayHold(ob) => {
            let end = node.offset() + ob.length;
            Ok(vec![AwgEvent {
                start: *node.offset(),
                end,
                kind: EventType::PlayHold(),
            }])
        }
        NodeKind::Match(ob) => {
            let end = node.offset() + ob.length;
            let obj = MatchEvent::from_ir(ob);
            let event = AwgEvent {
                start: *node.offset(),
                end,
                kind: EventType::Match(obj),
            };
            let mut out = vec![event];
            for child in node.take_children() {
                out.extend(generate_output_recursive(child, awg, state)?);
            }
            Ok(out)
        }
        NodeKind::FrameChange(ob) => {
            let mut hw_osc = None;
            if awg.device_kind().traits().supports_oscillator_switching {
                if let Some(osc) = &ob.signal.oscillator {
                    let out = HwOscillator {
                        uid: osc.uid.clone(),
                        index: *awg.osc_allocation.get(&osc.uid).expect("Missing index"),
                    };
                    hw_osc = Some(out);
                }
            }
            Ok(vec![AwgEvent {
                start: *node.offset(),
                end: node.offset() + ob.length,
                kind: EventType::ChangeHwOscPhase(ChangeHwOscPhase {
                    phase: ob.phase,
                    hw_oscillator: hw_osc,
                    parameter: ob.parameter,
                }),
            }])
        }
        NodeKind::Case(ob) => {
            state.state = Some(ob.state);
            let mut out = vec![];
            for child in node.take_children() {
                out.extend(generate_output_recursive(child, awg, state)?);
            }
            state.state = None;
            Ok(out)
        }
        NodeKind::ResetPrecompensationFilters(ob) => {
            let event = AwgEvent {
                start: *node.offset(),
                end: *node.offset() + ob.length,
                kind: EventType::ResetPrecompensationFilters(ob.length),
            };
            Ok(vec![event])
        }
        NodeKind::PpcStep(ob) => {
            let start = *node.offset();
            let end = *node.offset() + ob.length;

            let start_event = AwgEvent {
                start,
                end: start,
                kind: EventType::PpcSweepStepStart(ob.sweep_command.clone()),
            };
            let end_event = AwgEvent {
                start: end,
                end,
                kind: EventType::PpcSweepStepEnd(),
            };
            Ok(vec![start_event, end_event])
        }
        NodeKind::InitAmplitudeRegister(ob) => {
            let event = AwgEvent {
                start: *node.offset(),
                end: *node.offset(),
                kind: EventType::InitAmplitudeRegister(ob),
            };
            Ok(vec![event])
        }
        NodeKind::Acquire(ob) => {
            let length: i64 = ob.length();
            let event = AwgEvent {
                start: *node.offset(),
                end: *node.offset() + length,
                kind: EventType::AcquireEvent(),
            };
            Ok(vec![event])
        }
        NodeKind::ResetPhase() => Ok(vec![AwgEvent {
            start: *node.offset(),
            end: *node.offset(),
            kind: EventType::ResetPhase(),
        }]),
        NodeKind::InitialResetPhase() => Ok(vec![AwgEvent {
            start: *node.offset(),
            end: *node.offset(),
            kind: EventType::InitialResetPhase(),
        }]),
        NodeKind::SetTrigger(ob) => Ok(vec![AwgEvent {
            start: *node.offset(),
            end: *node.offset(),
            kind: EventType::TriggerOutputBit(TriggerOutputBit {
                bit: ob.bit,
                set: ob.set,
            }),
        }]),
        NodeKind::SetupPrng(ob) => Ok(vec![AwgEvent {
            start: *node.offset(),
            end: *node.offset(),
            kind: EventType::PrngSetup(PrngSetup {
                range: ob.range.into(),
                seed: ob.seed,
            }),
        }]),
        NodeKind::Loop(ob) => {
            let mut events = vec![];
            let num_repeat = ob.count;
            for (idx, child) in node.take_children().into_iter().enumerate() {
                let start = *child.offset();
                let end = start + child.data().length();
                if !ob.compressed {
                    let already_added = state
                        .loop_step_starts_added
                        .get(&start)
                        .is_some_and(|set| set.contains(&ob.section_info.id));
                    if !already_added {
                        events.push(AwgEvent {
                            start,
                            end: start,
                            kind: EventType::LoopStepStart(),
                        });
                        state
                            .loop_step_starts_added
                            .entry(start)
                            .or_default()
                            .insert(ob.section_info.id);
                    }
                } else if idx == 0 {
                    events.push(AwgEvent {
                        start,
                        end: start,
                        kind: EventType::PushLoop(PushLoop {
                            num_repeats: num_repeat,
                            compressed: ob.compressed,
                        }),
                    });
                }
                // Add PRNG sample drawing around iteration
                let draw_prng_sample = ob.prng_sample.is_some();
                if draw_prng_sample {
                    events.push(AwgEvent {
                        start,
                        end: start,
                        kind: EventType::PrngSample(),
                    });
                }
                events.extend(generate_output_recursive(child, awg, state)?);
                if draw_prng_sample {
                    events.push(AwgEvent {
                        start: end,
                        end,
                        kind: EventType::PrngDropSample(),
                    });
                }
                if !ob.compressed {
                    let already_added = state
                        .loop_step_ends_added
                        .get(&end)
                        .is_some_and(|set| set.contains(&ob.section_info.id));
                    if !already_added {
                        events.push(AwgEvent {
                            start: end,
                            end,
                            kind: EventType::LoopStepEnd(),
                        });
                        state
                            .loop_step_ends_added
                            .entry(end)
                            .or_default()
                            .insert(ob.section_info.id);
                    }
                }
                if ob.compressed && idx == 0 {
                    events.push(AwgEvent {
                        start: end,
                        end,
                        kind: EventType::Iterate(Iterate {
                            num_repeats: num_repeat,
                        }),
                    });
                }
            }
            Ok(events)
        }
        NodeKind::QaEvent(ob) => {
            let length = ob.length();
            let event = AwgEvent {
                start: *node.offset(),
                end: *node.offset() + length,
                kind: EventType::QaEvent(QaEvent::from_ir(ob)),
            };
            Ok(vec![event])
        }
        NodeKind::SetOscillatorFrequencySweep(ob) => {
            let start = *node.offset();
            let end = *node.offset() + ob.length;
            let mut awg_events = vec![];
            for osc in ob.oscillators.into_iter() {
                let event = AwgEvent {
                    start,
                    end,
                    kind: EventType::SetOscillatorFrequency(osc),
                };
                awg_events.push(event);
            }
            Ok(awg_events)
        }
        _ => {
            let mut out = vec![];
            for child in node.take_children() {
                out.extend(generate_output_recursive(child, awg, state)?);
            }
            Ok(out)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn generate_code_for_awg(
    root: &IrNode,
    awg: &AwgCore,
    pulse_parameters: &HashMap<PulseParametersId, PulseParameters>,
    settings: &CodeGeneratorSettings,
    waveform_sampler: &Py<PyAny>,
    acquisition_type: &AcquisitionType,
    acquisition_config: &FeedbackConfig<'_>,
    feedback_register_layout: &FeedbackRegisterLayout,
    is_reference_clock_internal: bool,
) -> Result<AwgCodeGenerationResultPy> {
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
    let waveforms = if WaveformSamplerPy::supports_waveform_sampling(awg) {
        collect_and_finalize_waveforms(
            &mut awg_node,
            WaveformSamplerPy::new(waveform_sampler, awg, pulse_parameters),
        )
    } else {
        Ok(AwgWaveforms::default())
    }?;
    let integration_kernels = collect_integration_kernels(&awg_node, awg)?;
    let integration_weights = batch_calculate_integration_weights(
        awg,
        waveform_sampler,
        integration_kernels,
        pulse_parameters,
    )?;
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
    let output = AwgCodeGenerationResultPy::create(
        awg_events.seqc,
        awg_events.wave_indices,
        awg_events.command_table,
        awg_events.shf_sweeper_config,
        sampled_waveforms,
        integration_weights,
        &measurement_info
            .delays
            .iter()
            .map(|(k, v)| (k.as_str(), v.delay_port().into()))
            .collect(),
        measurement_info
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
        awg_info.ppc_device(),
        awg_events.parameter_phase_increment_map,
        awg_events.feedback_register_config,
        feedback_register,
        source_feedback_register.cloned(),
    )?;
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
fn generate_code_for_multiple_awgs(
    root: &IrNode,
    awgs: &[AwgCore],
    pulse_parameters: &HashMap<PulseParametersId, PulseParameters>,
    settings: &CodeGeneratorSettings,
    waveform_sampler: &Py<PyAny>,
    acquisition_type: &AcquisitionType,
    acquisition_config: &FeedbackConfig<'_>,
    feedback_register_layout: &FeedbackRegisterLayout,
) -> Result<Vec<AwgCodeGenerationResultPy>> {
    let awg_results: Vec<AwgCodeGenerationResultPy> = awgs
        .par_iter()
        .map(|awg| -> Result<AwgCodeGenerationResultPy> {
            let code = generate_code_for_awg(
                root,
                awg,
                pulse_parameters,
                settings,
                waveform_sampler,
                acquisition_type,
                acquisition_config,
                feedback_register_layout,
                awg.is_reference_clock_internal,
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

fn transform_ir_and_awg(
    ir_tree: &Bound<PyAny>,
    awgs: &Bound<PyList>,
) -> Result<(
    IrNode,
    Vec<AwgCore>,
    HashMap<PulseParametersId, PulseParameters>,
)> {
    let root_ir = ir_tree.getattr("root")?;
    let ir_signals = ir_tree.getattr("signals")?;
    let mut awg_cores = vec![];
    for awg in awgs.try_iter()? {
        let mut awg = py_conversions::extract_awg(&awg?, &ir_signals)?;
        // Sort the signals for deterministic ordering
        awg.signals.sort_by(|a, b| a.channels.cmp(&b.channels));
        awg_cores.push(awg);
    }
    let (root, pulse_parameters) = py_conversions::transform_py_ir(&root_ir, &awg_cores)?;
    Ok((root, awg_cores, pulse_parameters))
}

fn estimate_total_execution_time(root: &IrNode) -> f64 {
    root.data().length() as f64 * TINYSAMPLE
}

// NOTE: When changing the API, update the stub in 'laboneq/_rust/codegenerator'
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn generate_code(
    py: Python,
    // IRTree
    ir: &Bound<PyAny>,
    // list[AwgInfo]
    awgs: &Bound<PyList>,
    feedback_register_layout: &Bound<PyDict>,
    acquisition_type: &Bound<'_, PyAny>,
    // Dictionary with compiler settings
    settings: &Bound<PyDict>,
    waveform_sampler: Py<PyAny>,
) -> Result<SeqCGenOutputPy> {
    let mut settings = code_generator_settings_from_dict(settings)?;
    let acquisition_type = py_conversions::extract_acquisition_type(acquisition_type)?;
    let feedback_register_layout =
        py_conversions::extract_feedback_register_layout(feedback_register_layout)?;
    let (ir_root, awgs, pulse_parameters) = transform_ir_and_awg(ir, awgs)?;
    for msg in settings.sanitize()? {
        log::warn!(
            "Compiler setting `{}` is sanitized from {} to {}. Reason: {}",
            msg.field.to_uppercase(),
            msg.original,
            msg.sanitized,
            msg.reason
        );
    }
    let total_execution_time = estimate_total_execution_time(&ir_root);
    let awg_refs: Vec<&AwgCore> = awgs.iter().collect();
    let feedback_config: FeedbackConfig<'_> = collect_feedback_config(&ir_root, &awg_refs)
        .context("Error while processing feedback configuration")?;
    let feedback_config_ref = &feedback_config;
    let awg_results = py.allow_threads(|| {
        generate_code_for_multiple_awgs(
            &ir_root,
            awgs.as_slice(),
            &pulse_parameters,
            &settings,
            &waveform_sampler,
            &acquisition_type,
            feedback_config_ref,
            &feedback_register_layout,
        )
    })?;
    Python::with_gil(|py| {
        let result = SeqCGenOutputPy::new(py, awg_results, total_execution_time, feedback_config);
        Ok(result)
    })
}

pub fn create_py_module<'a>(
    parent: &Bound<'a, PyModule>,
    name: &str,
) -> PyResult<Bound<'a, PyModule>> {
    let m = PyModule::new(parent.py(), name)?;
    // Common types
    // Move up the compiler stack as we need the common types
    m.add_class::<common_types::SignalTypePy>()?;
    m.add_class::<common_types::DeviceTypePy>()?;
    m.add_class::<common_types::MixerTypePy>()?;
    m.add_class::<common_types::AwgKeyPy>()?;
    // AWG Code generation
    m.add_function(wrap_pyfunction!(generate_code, &m)?)?;
    m.add_class::<PulseSignaturePy>()?;
    m.add_class::<WaveformSignaturePy>()?;
    // Waveform sampling
    m.add_class::<PlaySamplesPy>()?;
    m.add_class::<PlayHoldPy>()?;
    m.add_class::<SampledWaveformPy>()?;
    // Result
    m.add_class::<AwgCodeGenerationResultPy>()?;
    m.add_class::<FeedbackRegisterConfigPy>()?;
    Ok(m)
}

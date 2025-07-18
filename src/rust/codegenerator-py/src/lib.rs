// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use anyhow::Context;
use awg_event::{AcquireEvent, AwgEvent, EventType, InitAmplitudeRegisterPy, PlayWaveEventPy};
use codegenerator::CodeGeneratorSettings;
use codegenerator::fanout_for_awg;
use codegenerator::ir::compilation_job::AwgCore;
use codegenerator::signature::WaveformSignature;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::types::PyList;
use pyo3::wrap_pyfunction;
use signature::{PulseSignaturePy, WaveformSignaturePy};
use std::collections::HashSet;
use std::vec;
use waveform_sampler::PlayHoldPy;
use waveform_sampler::PlaySamplesPy;
mod code_generator;
mod py_conversions;
mod waveform_sampler;
use codegenerator::ir::{self, NodeKind};
use codegenerator::{
    AwgWaveforms, analyze_awg_ir, calculate_awg_delays, collect_and_finalize_waveforms,
    collect_integration_kernels, transform_ir_to_awg_events,
};
mod awg_event;
mod common_types;
mod result;
mod signature;
mod triggers;
use crate::awg_event::SetOscillatorFrequencyPy;
use crate::awg_event::{ChangeHwOscPhase, MatchEvent, QaEventPy};
mod settings;
use crate::settings::code_generator_settings_from_dict;
use result::{AwgCodeGenerationResultPy, SampledWaveformPy, SeqCGenOutputPy};

mod error;
use crate::code_generator::{
    SeqCGeneratorPy, SeqCTrackerPy, WaveIndexTrackerPy, merge_generators_py,
    seqc_generator_from_device_and_signal_type_py, string_sanitize_py,
};
use crate::error::Result;
use crate::waveform_sampler::WaveformSamplerPy;
use crate::waveform_sampler::batch_calculate_integration_weights;
use codegenerator::ir::SectionId;
use std::collections::HashMap;
use triggers::generate_trigger_states;

struct GeneratorState {
    pub loop_step_starts_added: HashMap<ir::Samples, HashSet<SectionId>>,
    pub loop_step_ends_added: HashMap<ir::Samples, HashSet<SectionId>>,
    pub state: Option<u16>,
}

fn generate_output(node: ir::IrNode, awg: &AwgCore) -> Result<Vec<AwgEvent>> {
    let mut state = GeneratorState {
        loop_step_starts_added: HashMap::new(),
        loop_step_ends_added: HashMap::new(),
        state: None,
    };
    let mut awg_events = generate_output_recursive(node, awg, &mut state)?;
    awg_event::sort_events(&mut awg_events);
    generate_trigger_states(&mut awg_events);
    add_positions(&mut awg_events); // todo: Check if needed - probably not
    Ok(awg_events)
}

/// Generate Python compatible AWG sampled events from the IR tree
fn generate_output_recursive(
    mut node: ir::IrNode,
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
                let out = signature::HwOscillator {
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
                kind: EventType::PlayWave(PlayWaveEventPy::from_ir(ob, state.state, hw_osc)),
                position: None,
            };
            Ok(vec![event])
        }
        NodeKind::PlayHold(ob) => {
            let end = node.offset() + ob.length;
            Ok(vec![AwgEvent {
                start: *node.offset(),
                end,
                kind: EventType::PlayHold(awg_event::PlayHoldEvent { length: ob.length }),
                position: None,
            }])
        }
        NodeKind::Match(ob) => {
            let end = node.offset() + ob.length;
            let obj = MatchEvent::from_ir(ob);
            let event = AwgEvent {
                start: *node.offset(),
                end,
                kind: EventType::Match(obj),
                position: None,
            };
            let mut out = vec![event];
            for child in node.take_children() {
                out.extend(generate_output_recursive(child, awg, state)?);
            }
            Ok(out)
        }
        NodeKind::FrameChange(ob) => {
            let mut hw_osc = None;
            if awg.device_kind.traits().supports_oscillator_switching {
                if let Some(osc) = &ob.signal.oscillator {
                    let out = signature::HwOscillator {
                        uid: osc.uid.clone(),
                        index: *awg.osc_allocation.get(&osc.uid).expect("Missing index"),
                    };
                    hw_osc = Some(out);
                }
            }
            Ok(vec![AwgEvent {
                start: *node.offset(),
                end: node.offset() + ob.length,
                position: Some(0),
                kind: EventType::ChangeHwOscPhase(ChangeHwOscPhase {
                    signal: ob.signal.uid.clone(),
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
                position: Some(0),
                kind: EventType::ResetPrecompensationFilters {
                    signature: WaveformSignaturePy::new(WaveformSignature::Pulses {
                        length: ob.length,
                        pulses: vec![],
                    }),
                },
            };
            Ok(vec![event])
        }
        NodeKind::PpcStep(ob) => {
            let start = *node.offset();
            let end = *node.offset() + ob.length;

            let start_event = AwgEvent {
                start,
                end: start,
                position: Some(0),
                kind: EventType::PpcSweepStepStart(awg_event::PpcSweepStepStart {
                    pump_power: ob.sweep_command.pump_power,
                    pump_frequency: ob.sweep_command.pump_frequency,
                    probe_power: ob.sweep_command.probe_power,
                    probe_frequency: ob.sweep_command.probe_frequency,
                    cancellation_phase: ob.sweep_command.cancellation_phase,
                    cancellation_attenuation: ob.sweep_command.cancellation_attenuation,
                }),
            };
            let end_event = AwgEvent {
                start: end,
                end,
                position: Some(0),
                kind: EventType::PpcSweepStepEnd(),
            };
            Ok(vec![start_event, end_event])
        }
        NodeKind::InitAmplitudeRegister(ob) => {
            let event = AwgEvent {
                start: *node.offset(),
                end: *node.offset(),
                position: None,
                kind: EventType::InitAmplitudeRegister(InitAmplitudeRegisterPy::new(ob)),
            };
            Ok(vec![event])
        }
        NodeKind::Acquire(ob) => {
            let length: i64 = ob.length();
            let event = AcquireEvent::from_ir(ob);
            let event = AwgEvent {
                start: *node.offset(),
                end: *node.offset() + length,
                position: Some(0),
                kind: EventType::AcquireEvent(event),
            };
            Ok(vec![event])
        }
        NodeKind::ResetPhase() => Ok(vec![AwgEvent {
            start: *node.offset(),
            end: *node.offset(),
            position: Some(0),
            kind: EventType::ResetPhase(),
        }]),
        NodeKind::InitialResetPhase() => Ok(vec![AwgEvent {
            start: *node.offset(),
            end: *node.offset(),
            position: Some(0),
            kind: EventType::InitialResetPhase(),
        }]),
        NodeKind::SetTrigger(ob) => Ok(vec![AwgEvent {
            start: *node.offset(),
            end: *node.offset(),
            position: Some(0),
            kind: EventType::TriggerOutputBit(awg_event::TriggerOutputBit {
                bit: ob.bit,
                set: ob.set,
            }),
        }]),
        NodeKind::SetupPrng(ob) => Ok(vec![AwgEvent {
            start: *node.offset(),
            end: *node.offset(),
            position: Some(0),
            kind: EventType::PrngSetup(awg_event::PrngSetup {
                range: ob.range,
                seed: ob.seed,
            }),
        }]),
        NodeKind::LoopIteration(ob) => {
            let start = *node.offset();
            let end = start + ob.length;
            let mut events = vec![];
            if !ob.compressed {
                let already_added = state
                    .loop_step_starts_added
                    .get(&start)
                    .is_some_and(|set| set.contains(&ob.section_info.id));
                if !already_added {
                    events.push(AwgEvent {
                        start,
                        end: start,
                        position: Some(0),
                        kind: EventType::LoopStepStart(),
                    });
                    state
                        .loop_step_starts_added
                        .entry(start)
                        .or_default()
                        .insert(ob.section_info.id);
                }
            } else if ob.iteration == 0 {
                events.push(AwgEvent {
                    start,
                    end: start,
                    position: Some(0),
                    kind: EventType::PushLoop(awg_event::PushLoop {
                        num_repeats: ob.num_repeats,
                        compressed: ob.compressed,
                    }),
                });
            }

            if let Some(sample_name) = &ob.prng_sample {
                events.push(AwgEvent {
                    start,
                    end: start,
                    position: Some(0),
                    kind: EventType::PrngSample(awg_event::PrngSample {
                        section_name: ob.section_info.name.clone(),
                        sample_name: sample_name.clone(),
                    }),
                });
            }
            for child in node.take_children() {
                events.extend(generate_output_recursive(child, awg, state)?);
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
                        position: Some(0),
                        kind: EventType::LoopStepEnd(),
                    });
                    state
                        .loop_step_ends_added
                        .entry(end)
                        .or_default()
                        .insert(ob.section_info.id);
                }
            }

            if let Some(sample_name) = &ob.prng_sample {
                events.push(AwgEvent {
                    start: end,
                    end,
                    position: Some(0),
                    kind: EventType::PrngDropSample(awg_event::PrngDropSample {
                        sample_name: sample_name.clone(),
                    }),
                });
            }

            if ob.compressed && ob.iteration == 0 {
                events.push(AwgEvent {
                    start: end,
                    end,
                    position: Some(0),
                    kind: EventType::Iterate(awg_event::Iterate {
                        num_repeats: ob.num_repeats,
                    }),
                });
            }
            Ok(events)
        }
        NodeKind::QaEvent(ob) => {
            let length = ob.length();
            let event = AwgEvent {
                start: *node.offset(),
                end: *node.offset() + length,
                position: Some(0),
                kind: EventType::QaEvent(QaEventPy::from_ir(ob)),
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
                    position: None,
                    kind: EventType::SetOscillatorFrequency(SetOscillatorFrequencyPy::new(osc)),
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

fn add_positions(events: &mut [AwgEvent]) {
    for (position, event) in events.iter_mut().enumerate() {
        if event.position.is_some() {
            event.position = Some(position as u64);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn generate_code_for_awg(
    root: &ir::IrNode,
    awg: &AwgCore,
    settings: &CodeGeneratorSettings,
    delays: &HashMap<String, f64>,
    waveform_sampler: &Py<PyAny>,
) -> Result<AwgCodeGenerationResultPy> {
    let root = fanout_for_awg(root, awg);
    let awg_info = analyze_awg_ir(&root);
    let awg_timing = calculate_awg_delays(awg, delays)?;
    let mut awg_node = transform_ir_to_awg_events(root, awg, settings, &awg_timing)?;
    let waveforms = if WaveformSamplerPy::supports_waveform_sampling(awg) {
        collect_and_finalize_waveforms(&mut awg_node, WaveformSamplerPy::new(waveform_sampler, awg))
    } else {
        Ok(AwgWaveforms::default())
    }?;
    let integration_kernels = collect_integration_kernels(&awg_node, awg)?;
    let integration_weights =
        batch_calculate_integration_weights(awg, waveform_sampler, integration_kernels)?;
    let awg_events = generate_output(awg_node, awg)?;
    let (sampled_waveforms, wave_declarations) = waveforms.into_inner();
    let output = AwgCodeGenerationResultPy::create(
        awg_events,
        sampled_waveforms,
        wave_declarations,
        integration_weights,
        awg_info,
        awg_timing.delay(),
    )?;
    Ok(output)
}

fn transform_delays(value: &Bound<PyDict>) -> Result<HashMap<String, f64>> {
    Ok(value.extract::<HashMap<String, f64>>()?)
}

fn transform_ir_and_awg(
    ir_tree: &Bound<PyAny>,
    awgs: &Bound<PyList>,
) -> Result<(ir::IrNode, Vec<AwgCore>)> {
    let root_ir = ir_tree.getattr("root")?;
    let ir_signals = ir_tree.getattr("signals")?;
    let mut awg_cores = vec![];
    for awg in awgs.try_iter()? {
        let mut awg = py_conversions::extract_awg(&awg?, &ir_signals)?;
        // Sort the signals for deterministic ordering
        awg.signals.sort_by(|a, b| a.channels.cmp(&b.channels));
        awg_cores.push(awg);
    }
    let root = py_conversions::transform_py_ir(&root_ir, &awg_cores)?;
    Ok((root, awg_cores))
}

// NOTE: When changing the API, update the stub in 'laboneq/_rust/codegenerator'
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn generate_code(
    // IRTree
    ir: &Bound<PyAny>,
    // list[AwgInfo]
    awgs: &Bound<PyList>,
    // Dictionary with compiler settings
    settings: &Bound<PyDict>,
    waveform_sampler: Py<PyAny>,
    delays: &Bound<PyDict>,
) -> Result<SeqCGenOutputPy> {
    let mut settings = code_generator_settings_from_dict(settings)?;
    let (ir_root, awgs) = transform_ir_and_awg(ir, awgs)?;
    let delays: HashMap<String, f64> = transform_delays(delays)?;
    for msg in settings.sanitize()? {
        log::warn!(
            "Compiler setting `{}` is sanitized from {} to {}. Reason: {}",
            msg.field.to_uppercase(),
            msg.original,
            msg.sanitized,
            msg.reason
        );
    }
    let mut awg_results = vec![];
    for awg in awgs.iter() {
        let awg_result =
            generate_code_for_awg(&ir_root, awg, &settings, &delays, &waveform_sampler).context(
                format!(
                    "Error while generating code for signals: {}",
                    &awg.signals
                        .iter()
                        .map(|s| s.uid.to_string())
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
            )?;
        awg_results.push(awg_result);
    }
    Python::with_gil(|py| {
        let result = SeqCGenOutputPy::new(py, awg_results);
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
    // AWG Code generation
    m.add_function(wrap_pyfunction!(generate_code, &m)?)?;
    m.add_class::<PulseSignaturePy>()?;
    m.add_class::<WaveformSignaturePy>()?;
    // Waveform sampling
    m.add_class::<PlaySamplesPy>()?;
    m.add_class::<PlayHoldPy>()?;
    m.add_class::<SampledWaveformPy>()?;
    // Sampled event handler
    m.add_class::<WaveIndexTrackerPy>()?;
    m.add_class::<SeqCGeneratorPy>()?;
    m.add_class::<SeqCTrackerPy>()?;
    m.add_function(wrap_pyfunction!(
        seqc_generator_from_device_and_signal_type_py,
        &m
    )?)?;
    m.add_function(wrap_pyfunction!(merge_generators_py, &m)?)?;
    m.add_function(wrap_pyfunction!(string_sanitize_py, &m)?)?;
    // Result
    m.add_class::<AwgCodeGenerationResultPy>()?;
    Ok(m)
}

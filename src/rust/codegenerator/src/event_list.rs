// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use crate::context::CodeGenContext;
use crate::ir::PlayWave;
use crate::ir::compilation_job::AwgCore;
use crate::sampled_event_handler::AwgEventList;
use crate::sampled_event_handler::awg_events::{
    AcquireEvent, AwgEvent, ChangeHwOscPhase, EventType, Iterate, MatchEvent, PlayWaveEvent,
    PrngSetup, PushLoop, QaEvent, StaticWaveformSignature, TriggerOutputBit,
};
use crate::sampled_event_handler::seqc_tracker::awg::HwOscillator;
use crate::signature::WaveformSignature;
use crate::triggers::generate_trigger_states;
use crate::utils::normalize_phase;
use crate::{Result, ir::IrNode, ir::NodeKind, ir::Samples, ir::SectionId};
use std::collections::BTreeMap;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

pub(crate) fn generate_event_list(
    node: IrNode,
    awg: &AwgCore,
    ctx: &CodeGenContext,
) -> Result<AwgEventList> {
    let mut state = GeneratorState {
        loop_step_starts_added: HashMap::new(),
        loop_step_ends_added: HashMap::new(),
        state: None,
        waveforms: HashMap::new(),
    };
    let mut awg_events = generate_output_recursive(node, awg, &mut state, ctx)?;
    awg_events.sort_by_key(|event| event.start);
    generate_trigger_states(&mut awg_events);
    Ok(awg_events
        .into_iter()
        .fold(BTreeMap::new(), |mut map: AwgEventList, event| {
            map.entry(event.start).or_default().push(event);
            map
        }))
}

struct GeneratorState {
    pub loop_step_starts_added: HashMap<Samples, HashSet<SectionId>>,
    pub loop_step_ends_added: HashMap<Samples, HashSet<SectionId>>,
    pub state: Option<u16>,
    pub waveforms: HashMap<u64, Rc<StaticWaveformSignature>>,
}

fn playwave_to_event(mut ob: PlayWave, awg: &AwgCore, state: &mut GeneratorState) -> PlayWaveEvent {
    let wave = state.waveforms.entry(ob.waveform.uid()).or_insert_with(|| {
        let sig_string = ob.waveform.signature_string();
        Rc::new(StaticWaveformSignature::new(
            ob.waveform.uid(),
            ob.waveform,
            sig_string,
        ))
    });
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
    PlayWaveEvent {
        waveform: Rc::clone(wave),
        state: state.state,
        hw_oscillator: hw_osc,
        amplitude_register: ob.amplitude_register,
        amplitude: ob.amplitude,
        increment_phase: ob.increment_phase,
        increment_phase_params: ob.increment_phase_params,
        channels: ob
            .signals
            .first()
            .map_or_else(Vec::new, |sig| sig.channels.clone()),
    }
}

fn generate_output_recursive(
    mut node: IrNode,
    awg: &AwgCore,
    state: &mut GeneratorState,
    ctx: &CodeGenContext,
) -> Result<Vec<AwgEvent>> {
    match node.swap_data(NodeKind::Nop { length: 0 }) {
        NodeKind::PhaseReset(_) => {
            panic!("Internal error: PhaseReset should have been replaced by ResetPhase");
        }
        NodeKind::PlayWave(ob) => {
            let end = node.offset() + ob.length();
            let e = playwave_to_event(ob, awg, state);
            let event = AwgEvent {
                start: *node.offset(),
                end,
                kind: EventType::PlayWave(e),
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
                out.extend(generate_output_recursive(child, awg, state, ctx)?);
            }
            Ok(out)
        }
        NodeKind::FrameChange(ob) => {
            let mut hw_osc = None;
            if awg.device_kind().traits().supports_oscillator_switching
                && let Some(osc) = &ob.signal.oscillator
            {
                let out = HwOscillator {
                    uid: osc.uid.clone(),
                    index: *awg.osc_allocation.get(&osc.uid).expect("Missing index"),
                };
                hw_osc = Some(out);
            }
            // The `phase_resolution_range` is irrelevant here; for the CT phase a fixed
            // precision is used.
            const PHASE_RESOLUTION_CT: f64 = (1 << 24) as f64 / (2.0 * std::f64::consts::PI);
            let quantized_phase =
                normalize_phase((ob.phase * PHASE_RESOLUTION_CT).round() / PHASE_RESOLUTION_CT);
            let waveform = WaveformSignature::Pulses {
                length: 0,
                pulses: vec![],
            };
            let wf2 = StaticWaveformSignature::new(waveform.uid(), waveform, "".to_string());
            let signature = PlayWaveEvent {
                waveform: Rc::new(wf2),
                state: None,
                hw_oscillator: hw_osc,
                amplitude_register: 0,
                amplitude: None,
                increment_phase: Some(quantized_phase),
                increment_phase_params: vec![ob.parameter],
                channels: vec![],
            };
            Ok(vec![AwgEvent {
                start: *node.offset(),
                end: node.offset() + ob.length,
                kind: EventType::ChangeHwOscPhase(ChangeHwOscPhase { signature }),
            }])
        }
        NodeKind::Case(ob) => {
            state.state = Some(ob.state);
            let mut out = vec![];
            for child in node.take_children() {
                out.extend(generate_output_recursive(child, awg, state, ctx)?);
            }
            state.state = None;
            Ok(out)
        }
        NodeKind::ResetPrecompensationFilters(ob) => {
            let waveform = WaveformSignature::Pulses {
                length: ob.length,
                pulses: vec![],
            };
            let wf2 =
                StaticWaveformSignature::new(waveform.uid(), waveform, "precomp_reset".to_string());
            let signature = PlayWaveEvent {
                hw_oscillator: None,
                amplitude: None,
                amplitude_register: 0,
                increment_phase: None,
                increment_phase_params: vec![],
                waveform: Rc::new(wf2),
                state: None,
                channels: vec![],
            };
            let event = AwgEvent {
                start: *node.offset(),
                end: *node.offset() + ob.length,
                kind: EventType::ResetPrecompensationFilters { signature },
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
            let waveform = WaveformSignature::Pulses {
                length: 0,
                pulses: vec![],
            };
            let wave = state.waveforms.entry(waveform.uid()).or_insert_with(|| {
                let sig_string = waveform.signature_string();
                Rc::new(StaticWaveformSignature::new(
                    waveform.uid(),
                    waveform,
                    sig_string,
                ))
            });
            let signature = PlayWaveEvent {
                waveform: Rc::clone(wave),
                state: None,
                hw_oscillator: None,
                amplitude_register: ob.register,
                amplitude: Some(ob.value),
                increment_phase: None,
                increment_phase_params: vec![],
                channels: vec![],
            };
            let event = AwgEvent {
                start: *node.offset(),
                end: *node.offset(),
                kind: EventType::InitAmplitudeRegister { signature },
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
                bits: ob.bits,
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
                events.extend(generate_output_recursive(child, awg, state, ctx)?);
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
            let (acquires, waveforms) = ob.into_parts();
            let e =
                QaEvent {
                    acquire_events: acquires
                        .into_iter()
                        .map(|acq| {
                            let integration_channels = ctx
                        .integration_units_for_signal(acq.signal().uid)
                        .cloned()
                        .expect("Internal error: Missing integration unit allocation for signal");
                            AcquireEvent {
                                channels: integration_channels,
                            }
                        })
                        .collect(),
                    play_wave_events: waveforms
                        .into_iter()
                        .map(|wf| playwave_to_event(wf, awg, state))
                        .collect(),
                };
            let event = AwgEvent {
                start: *node.offset(),
                end: *node.offset() + length,
                kind: EventType::QaEvent(e),
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
                out.extend(generate_output_recursive(child, awg, state, ctx)?);
            }
            Ok(out)
        }
    }
}

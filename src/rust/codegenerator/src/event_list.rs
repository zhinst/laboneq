// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use crate::ir::compilation_job::AwgCore;
use crate::sampled_event_handler::AwgEventList;
use crate::sampled_event_handler::awg_events::{
    AwgEvent, ChangeHwOscPhase, EventType, Iterate, MatchEvent, PlayWaveEvent, PrngSetup, PushLoop,
    QaEvent, TriggerOutputBit,
};
use crate::sampled_event_handler::seqc_tracker::awg::HwOscillator;
use crate::triggers::generate_trigger_states;
use crate::{Result, ir::IrNode, ir::NodeKind, ir::Samples, ir::SectionId};
use std::collections::BTreeMap;
use std::collections::{HashMap, HashSet};

pub fn generate_event_list(node: IrNode, awg: &AwgCore) -> Result<AwgEventList> {
    let mut state = GeneratorState {
        loop_step_starts_added: HashMap::new(),
        loop_step_ends_added: HashMap::new(),
        state: None,
    };
    let mut awg_events = generate_output_recursive(node, awg, &mut state)?;
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

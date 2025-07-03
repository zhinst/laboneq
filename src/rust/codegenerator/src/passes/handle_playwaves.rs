// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use crate::Result;
use crate::ir;
use crate::ir::compilation_job::{self as cjob};
use crate::passes::handle_frame_changes::insert_frame_changes;
use crate::passes::handle_oscillators::SoftwareOscillatorParameters;
use crate::signature::{
    PulseSignature, WaveformSignature, sort_pulses, split_complex_pulse_amplitude,
};
use crate::tinysample::{self, ceil_to_grid};
use crate::utils;
use crate::virtual_signal::{VirtualSignal, VirtualSignals};
use anyhow::anyhow;
use interval_calculator::calculate_intervals;
use interval_calculator::interval::{Interval, OrderedRange};
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use super::amplitude_registers::AmplitudeRegisterAllocation;

fn adjust_sequence_end_point(
    end: ir::Samples,
    sample_multiple: u16,
    delay: ir::Samples,
    play_wave_size_hint: u16,
    play_zero_size_hint: u16,
) -> ir::Samples {
    let mut end = end + delay;
    end += play_wave_size_hint as ir::Samples + play_zero_size_hint as ir::Samples;
    end += (-end) % sample_multiple as ir::Samples;
    end
}

struct CutPoints {
    general: HashSet<ir::Samples>,
    command_table: HashSet<(ir::Samples, ir::Samples)>,
}

impl CutPoints {
    fn new() -> Self {
        CutPoints {
            general: HashSet::new(),
            command_table: HashSet::new(),
        }
    }
}

fn calculate_cut_points(node: &ir::IrNode, cut_pts: &mut CutPoints) {
    match node.data() {
        ir::NodeKind::Match(_) => {
            // TODO: HW oscillator switch cannot happen inside a match case
            // TODO: A single AWG must not have overlapping match cases
            let offset = *node.offset();
            let end = offset + node.data().length();
            cut_pts.general.extend([offset, end]);
            cut_pts.command_table.insert((offset, end));
            for child in node.iter_children() {
                calculate_cut_points(child, cut_pts);
            }
        }
        ir::NodeKind::Acquire(_) => {
            cut_pts.general.insert(*node.offset());
        }
        _ => {
            for child in node.iter_children() {
                calculate_cut_points(child, cut_pts);
            }
        }
    }
}

fn evaluate_cut_points(
    node: &ir::IrNode,
    sample_multiple: u16,
    delay: ir::Samples,
    play_wave_size_hint: u16,
    play_zero_size_hint: u16,
) -> CutPoints {
    let end_time = adjust_sequence_end_point(
        node.data().length(),
        sample_multiple,
        delay,
        play_wave_size_hint,
        play_zero_size_hint,
    );
    let mut cut_pts = CutPoints::new();
    cut_pts.general.insert(end_time);
    calculate_cut_points(node, &mut cut_pts);
    cut_pts
}

#[derive(Debug)]
struct PlayPulseSlot<'a> {
    node: &'a mut ir::IrNode,
    state: Option<u16>,
    signal: Rc<cjob::Signal>,
}

impl PlayPulseSlot<'_> {
    fn kind(&self) -> &ir::NodeKind {
        self.node.data()
    }
}

/// Assign and collect play wave nodes
/// Adds an empty play node for each signal contained in an empty
/// match case.
/// Returns nodes pointing to the play nodes in the tree.
fn assign_pulse_slots(node: &mut ir::IrNode, state: Option<u16>) -> Vec<PlayPulseSlot<'_>> {
    match node.data() {
        ir::NodeKind::FrameChange(ob) => {
            let signal = Rc::clone(&ob.signal);
            let slot = PlayPulseSlot {
                node,
                state,
                signal,
            };
            vec![slot]
        }
        ir::NodeKind::PlayPulse(ob) => {
            // Ignore frame change and zero length pulse
            if ob.pulse_def.is_none() || ob.length == 0 {
                return vec![];
            }
            let signal = Rc::clone(&ob.signal);
            let slot = PlayPulseSlot {
                node,
                state,
                signal,
            };
            vec![slot]
        }
        ir::NodeKind::Case(ob) => {
            let state = Some(ob.state);
            if (ob.length == 0) || node.has_children() {
                let mut out = vec![];
                for child in node.iter_children_mut() {
                    out.extend(assign_pulse_slots(child, state));
                }
                return out;
            }
            // Populate the empty case with placeholders which are
            // meant to be replaced by waveforms.
            let len = ob.length;
            let signals = ob.signals.clone();
            for _ in 0..signals.len() {
                node.add_child(*node.offset(), ir::NodeKind::Nop { length: len });
            }
            let mut out: Vec<PlayPulseSlot<'_>> = vec![];
            for (idx, ch) in node.iter_children_mut().enumerate() {
                out.push(PlayPulseSlot {
                    node: ch,
                    state,
                    signal: Rc::clone(&signals[idx]),
                })
            }
            out
        }
        _ => {
            let mut out = vec![];
            for child in node.iter_children_mut() {
                out.extend(assign_pulse_slots(child, state));
            }
            out
        }
    }
}

#[derive(Debug)]
struct WaveformSlot {
    node: usize,
    start: ir::Samples,
    end: ir::Samples,
    signal: Rc<cjob::Signal>,
}

fn signals_share_hw_oscillator(s0: &cjob::Signal, s1: &cjob::Signal) -> bool {
    if s0.is_hw_modulated() && s1.is_hw_modulated() {
        if let (Some(topd), Some(nextd)) = (&s0.oscillator, &s1.oscillator) {
            return topd == nextd;
        }
    }
    true
}

/// Analyze oscillator switches across pulses.
///
/// # Returns
///
/// Each point where oscillator switch happens or an error if pulses
/// using different hardware oscillators overlap.
fn analyze_oscillator_switches<'a, I, D>(
    intervals: I,
    disallow_switch_ranges: D,
) -> Result<Option<Vec<ir::Samples>>>
where
    I: IntoIterator<Item = (ir::Samples, ir::Samples, &'a cjob::Signal)>,
    D: IntoIterator<Item = (ir::Samples, ir::Samples)>,
{
    let mut intervals: Vec<(OrderedRange<ir::Samples>, Option<&cjob::Signal>)> = intervals
        .into_iter()
        .map(|(start, stop, signal)| (OrderedRange(start..stop), Some(signal)))
        .collect();
    intervals.extend(
        disallow_switch_ranges
            .into_iter()
            .map(|(start, stop)| (OrderedRange(start..stop), None)),
    );
    intervals.sort_by(|a, b| a.0.cmp(&b.0));
    if intervals.is_empty() {
        return Ok(None);
    }
    let mut osc_switch_cut_pts = Vec::new();
    let mut top = intervals[0].clone();
    for next in intervals.iter_mut().skip(1) {
        // No overlap, store oscillator switch points if one
        if top.0.0.end <= next.0.0.start {
            if let (Some(s0), Some(s1)) = (top.1, next.1) {
                if !signals_share_hw_oscillator(s0, s1) {
                    osc_switch_cut_pts.push(next.0.0.start);
                }
            }
            top.0.0.end = next.0.0.end;
            top.1 = next.1;
        } else {
            // Overlapping ranges
            if let (Some(s0), Some(s1)) = (top.1, next.1) {
                if !signals_share_hw_oscillator(s0, s1) {
                    let msg = format!(
                        "Overlapping HW oscillators: '{:}' on signal '{:}' and '{:}' on signal '{:}'",
                        s0.oscillator.as_ref().unwrap().uid,
                        s0.uid,
                        s1.oscillator.as_ref().unwrap().uid,
                        s1.uid
                    );
                    return Err(anyhow!(msg).into());
                }
            }
            if top.0.0.end < next.0.0.end {
                top.0.0.end = next.0.0.end;
            }
            top.1 = top.1.or(next.1);
        }
    }
    Ok(Some(osc_switch_cut_pts))
}

fn create_pulse_signature(
    node: &mut PlayPulseSlot,
    virtual_signal: &VirtualSignal,
    amp_reg_alloc: &AmplitudeRegisterAllocation,
    osc_parameters: &SoftwareOscillatorParameters,
) -> PulseSignature {
    let length = node.node.data().length();
    let channel = if !virtual_signal.is_multiplexed() {
        None
    } else {
        Some(
            virtual_signal
                .get_channel_by_signal(&node.signal.uid)
                .expect("Invalid signal"),
        )
    };
    match node.node.swap_data(ir::NodeKind::Nop { length: 0 }) {
        ir::NodeKind::PlayPulse(ob) => {
            let osc_phase = osc_parameters
                .phase_at(&ob.signal, *node.node.offset())
                .map(utils::normalize_phase);
            let (increment_oscillator_phase, incr_phase_params) = if ob.signal.is_hw_modulated() {
                (
                    ob.increment_oscillator_phase,
                    ob.incr_phase_param_name.map_or(vec![], |x| vec![x]),
                )
            } else {
                (None, vec![])
            };
            let (amplitude, phase) = if let Some(amp) = ob.amplitude {
                let (amplitude, phase) =
                    split_complex_pulse_amplitude(amp, utils::normalize_phase(ob.phase));
                (Some(amplitude), phase)
            } else {
                (None, utils::normalize_phase(ob.phase))
            };
            PulseSignature {
                start: node.node.offset() + virtual_signal.delay(),
                length: ob.length,
                pulse: ob.pulse_def,
                amplitude,
                phase,
                oscillator_frequency: osc_parameters.freq_at(&ob.signal, *node.node.offset()),
                oscillator_phase: osc_phase,
                increment_oscillator_phase,
                channel,
                sub_channel: virtual_signal.subchannel(),
                id_pulse_params: ob.id_pulse_params,
                markers: ob.markers,
                preferred_amplitude_register: Some(
                    amp_reg_alloc.get_allocation(ob.amp_param_name.as_deref()),
                ),
                incr_phase_params,
            }
        }
        _ => PulseSignature {
            start: node.node.offset() + virtual_signal.delay(),
            length,
            pulse: None,
            amplitude: None,
            phase: 0.0,
            oscillator_frequency: None,
            oscillator_phase: None,
            increment_oscillator_phase: None,
            channel,
            sub_channel: virtual_signal.subchannel(),
            id_pulse_params: None,
            markers: vec![],
            preferred_amplitude_register: Some(0),
            incr_phase_params: vec![],
        },
    }
}

fn create_waveform_slots(
    pulses: &Vec<PlayPulseSlot<'_>>,
    signal: &VirtualSignal,
) -> (Vec<WaveformSlot>, Option<ir::Samples>) {
    // First frame change that happens after the last pulse is a hard cut point.
    // Every frame change after that are handled as oscillator phase increment.
    let mut frame_change_cut_point = None;
    let mut waveform_slots: Vec<WaveformSlot> = vec![];
    for (node_id, pulse_slot) in pulses.iter().enumerate() {
        if !signal.contains_signal(&pulse_slot.signal.uid) {
            continue;
        }
        match pulse_slot.kind() {
            &ir::NodeKind::FrameChange(_) => {
                if let Some(current_waveform) = waveform_slots.last() {
                    if pulse_slot.node.offset() >= &current_waveform.end {
                        frame_change_cut_point = Some(pulse_slot.node.offset());
                    }
                }
            }
            _ => {
                let start_adjusted = pulse_slot.node.offset() + signal.delay();
                let end = start_adjusted + pulse_slot.node.data().length();
                if frame_change_cut_point.is_some_and(|x| x < &end) {
                    frame_change_cut_point = None;
                }
                let wf = WaveformSlot {
                    node: node_id,
                    start: start_adjusted,
                    end,
                    signal: Rc::clone(&pulse_slot.signal),
                };
                waveform_slots.push(wf);
            }
        }
    }
    (waveform_slots, frame_change_cut_point.cloned())
}

/// Finds a single hardware oscillator for a set of signals.
fn find_hw_oscillator<'a, S>(signals: S) -> Option<&'a cjob::Oscillator>
where
    S: IntoIterator<Item = &'a cjob::Signal>,
{
    for sig in signals {
        if sig.is_hw_modulated() {
            return sig.oscillator.as_ref();
        }
    }
    None
}

/// Transform play wave nodes into waveforms.
#[allow(clippy::too_many_arguments)]
pub fn handle_plays(
    program: &mut ir::IrNode,
    awg: &cjob::AwgCore,
    virtual_signals: &VirtualSignals,
    cut_points: HashSet<ir::Samples>,
    play_wave_size_hint: u16,
    play_zero_size_hint: u16,
    osc_parameters: &SoftwareOscillatorParameters,
    amp_reg_alloc: &AmplitudeRegisterAllocation,
) -> Result<()> {
    let traits = &awg.device_kind.traits();
    let mut local_cut_pts = evaluate_cut_points(
        program,
        traits.sample_multiple,
        *virtual_signals.delay(),
        play_wave_size_hint,
        play_zero_size_hint,
    );
    local_cut_pts.general.extend(cut_points);
    let ct_intervals: HashSet<OrderedRange<i64>> = local_cut_pts
        .command_table
        .into_iter()
        .map(|(start, end)| OrderedRange(start..end))
        .collect();
    // Group nodes by signal
    let mut plays_by_signal: HashMap<String, Vec<_>> = HashMap::new();
    for node in assign_pulse_slots(program, None) {
        if let Some(entry) = plays_by_signal.get_mut(node.signal.uid.as_str()) {
            entry.push(node);
        } else {
            plays_by_signal.insert(node.signal.uid.clone(), vec![node]);
        }
    }
    for signal in virtual_signals.iter() {
        let mut signal_events: Vec<PlayPulseSlot> = signal
            .signals()
            .filter_map(|ch| plays_by_signal.remove(ch.uid.as_str()))
            .flatten()
            .collect();
        if signal_events.is_empty() {
            continue;
        }
        let (mut signatures, cut_point_frame_change) =
            create_waveform_slots(&signal_events, signal);
        if signatures.is_empty() {
            continue;
        }
        if let Some(frame_change) = cut_point_frame_change {
            local_cut_pts
                .general
                .insert(ceil_to_grid(frame_change, traits.sample_multiple.into()));
        }
        // Round waveform slots to sequencer grid
        signatures.iter_mut().for_each(|x| {
            x.start = tinysample::floor_to_grid(x.start, traits.sample_multiple.into());
            x.end = ceil_to_grid(x.end, traits.sample_multiple.into());
        });
        if awg.kind != cjob::AwgKind::DOUBLE && signal.is_multiplexed() {
            // Skip for now. In double mode, 2 oscillators may (?) be active.
            // Still, oscillator switching is not supported on these instruments.
            let osc_switch_cut_points = analyze_oscillator_switches(
                signatures
                    .iter()
                    .map(|x| (x.start, x.end, x.signal.as_ref())),
                ct_intervals.iter().map(|x| (x.0.start, x.0.end)),
            )?;
            if let Some(osc_switch) = osc_switch_cut_points {
                local_cut_pts.general.extend(osc_switch);
            }
        }
        let mut cut_points_inp = local_cut_pts
            .general
            .iter()
            .copied()
            .collect::<Vec<ir::Samples>>();
        cut_points_inp.sort();
        let waveform_intervals = signatures
            .iter()
            .enumerate()
            .map(|(idx, ob)| Interval::from_range(ob.start..ob.end, vec![idx]))
            .collect();
        let compacted_intervals = calculate_intervals(
            waveform_intervals,
            &cut_points_inp,
            traits.sample_multiple.into(),
            traits.min_play_wave.into(),
            play_wave_size_hint.into(),
            play_zero_size_hint.into(),
            Some(traits.playwave_max_hint.unwrap_or(i64::MAX as u64) as i64),
            Some(&ct_intervals),
        );
        let compacted_intervals = match compacted_intervals {
            Err(e) => match e {
                interval_calculator::Error::MinimumWaveformLengthViolation(_) => {
                    let msg = format!(
                        "Failed to map the scheduled pulses to SeqC without violating the minimum waveform size \
                    {} samples on device '{}'.\n\
                    Suggested workaround: manually add delays to overly short loops, etc.",
                        traits.min_play_wave,
                        awg.device_kind.as_str()
                    );
                    return Err(anyhow!(msg).into());
                }
                _ => return Err(anyhow!(e.to_string()).into()),
            },
            Ok(out) => out,
        };
        let use_ct_hw_oscillator =
            awg.use_command_table_phase_amp() && awg.device_kind == cjob::DeviceKind::SHFSG;
        for wave_range in compacted_intervals.into_iter() {
            let waveform_start = wave_range.start();
            let waveform_length = wave_range.length();
            let mut iv_per_state: HashMap<Option<u16>, Vec<&WaveformSlot>> = HashMap::new();
            // Group by state
            for idx in wave_range.data.iter() {
                let waveform_slot = &signatures[*idx];
                iv_per_state
                    .entry(signal_events[waveform_slot.node].state)
                    .or_default()
                    .push(waveform_slot);
            }
            for merged_pulses in iv_per_state.into_values() {
                // Double can have multiple HW oscillators active, but we don't support oscillator switching.
                // In that case the shared HW oscillator must be set to None I guess?
                let hw_osc = if use_ct_hw_oscillator {
                    find_hw_oscillator(merged_pulses.iter().map(|x| x.signal.as_ref()))
                } else {
                    None
                };
                let mut pulse_signatures: Vec<PulseSignature> = merged_pulses
                    .iter()
                    .map(|slot| {
                        let mut signature = create_pulse_signature(
                            &mut signal_events[slot.node],
                            signal,
                            amp_reg_alloc,
                            osc_parameters,
                        );
                        // Adjust signature start time to be relative to waveform
                        signature.start -= waveform_start;
                        signature
                    })
                    .collect();
                sort_pulses(&mut pulse_signatures);
                let waveform = WaveformSignature::Pulses {
                    length: waveform_length,
                    pulses: pulse_signatures,
                };
                let playwave = ir::PlayWave {
                    waveform,
                    signals: signal.signals().cloned().collect(),
                    oscillator: hw_osc.map(|x| x.uid.clone()),
                    amplitude_register: 0,
                    amplitude: None,
                    increment_phase: None,
                    increment_phase_params: vec![],
                };
                // As the pulses are merged, we use the slot of the first one to put them into the tree
                *signal_events[merged_pulses[0].node].node.offset_mut() = waveform_start;
                signal_events[merged_pulses[0].node]
                    .node
                    .replace_data(ir::NodeKind::PlayWave(playwave));
                // Set merged nodes as Nop to be removed or ignored
                for node in merged_pulses.get(1..).unwrap_or_default() {
                    signal_events[node.node]
                        .node
                        .replace_data(ir::NodeKind::Nop { length: 0 });
                }
            }
        }
        insert_frame_changes(
            signal_events
                .into_iter()
                .map(|x| (x.state, x.node))
                .collect(),
        )?;
    }
    Ok(())
}

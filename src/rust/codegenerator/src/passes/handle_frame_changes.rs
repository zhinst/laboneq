// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use crate::Result;
use crate::ir;
use crate::ir::compilation_job::{self as cjob};
use crate::signature::PulseSignature;
use anyhow::anyhow;
use std::collections::HashMap;
use std::ops::Range;
use std::rc::Rc;

fn try_convert_pulse_to_frame_change(pulse: &ir::PlayPulse) -> Option<ir::FrameChange> {
    // TODO: Prune `Delay` nodes from  the IR
    if pulse.pulse_def.is_some() || !pulse.signal.is_hw_modulated() {
        return None;
    }
    if let Some(incr_phase) = pulse.increment_oscillator_phase {
        let frame = ir::FrameChange {
            length: pulse.length,
            phase: incr_phase,
            parameter: pulse.incr_phase_param_name.clone(),
            signal: Rc::clone(&pulse.signal),
        };
        return Some(frame);
    }
    None
}

/// Transform hardware oscillator phase increment pulses into frame changes
pub fn handle_frame_changes(node: &mut ir::IrNode) {
    match node.data_mut() {
        ir::NodeKind::PlayPulse(x) => {
            if let Some(fc) = try_convert_pulse_to_frame_change(x) {
                node.replace_data(ir::NodeKind::FrameChange(fc));
            }
        }
        _ => {
            for child in node.iter_children_mut() {
                handle_frame_changes(child);
            }
        }
    }
}

struct ActivateWaveform<'a> {
    start: ir::Samples,
    length: ir::Samples,
    osc: Option<&'a str>,
    signatures: &'a mut Vec<PulseSignature>,
}

struct FrameChangeTracker<'a> {
    current_waveform: HashMap<Option<u16>, ActivateWaveform<'a>>,
}

impl<'a> FrameChangeTracker<'a> {
    fn new() -> Self {
        FrameChangeTracker {
            current_waveform: HashMap::new(),
        }
    }

    fn set_active_waveform(
        &mut self,
        start: ir::Samples,
        length: ir::Samples,
        state: Option<u16>,
        osc: Option<&'a str>,
        signatures: &'a mut Vec<PulseSignature>,
    ) {
        self.current_waveform.insert(
            state,
            ActivateWaveform {
                start,
                length,
                osc,
                signatures,
            },
        );
    }

    fn point_inside_range<T: Ord>(range: &Range<T>, point: T) -> bool {
        range.start < point && point < range.end
    }

    /// Try to insert a frame change into a currently playing waveform
    ///
    /// # Returns
    ///
    /// True if the frame change information was inserted into the waveform.
    fn try_insert_frame_change(
        &mut self,
        state: &Option<u16>,
        signal: &cjob::Signal,
        offset: ir::Samples,
        phase: f64,
        parameter: &mut Option<String>,
    ) -> Result<bool> {
        if let Some(wf) = self.current_waveform.get_mut(state) {
            let wf_start = wf.start;
            let wf_range = wf_start..wf_start + wf.length;
            // Frame change does not overlap with waveform within the case block
            if !wf_range.contains(&offset) {
                if state.is_some() {
                    let msg = "Cannot handle free-standing oscillator phase change in conditional branch.\n\
                    A zero-length 'increment_oscillator_phase' must not occur at the \
                    very end of a 'case' block. Consider stretching the branch by \
                    adding a small delay after the phase increment.";
                    return Err(anyhow!(msg).into());
                }
                return Ok(false);
            }
            // Check for overlapping oscillators
            // If the frame change happens at the start or end of an waveform, that is ok:
            // we can emit a 0-length command table entry.
            if let Some(cjob::Oscillator { uid, .. }) = &signal.oscillator {
                if let Some(wf_osc) = wf.osc.as_ref().filter(|&wf_osc| wf_osc != uid) {
                    if FrameChangeTracker::point_inside_range(&wf_range, offset) {
                        let msg = format!(
                            "Cannot increment oscillator '{}' of signal '{}': the line is occupied by '{}'",
                            uid, signal.uid, wf_osc
                        );
                        return Err(anyhow!(msg).into());
                    } else {
                        return Ok(false);
                    }
                }
            }
            // Change timing to relative for comparison
            let offset_fc = offset - wf_start;
            let mut apply_to_last = false;
            // Insert the frame change into the first overlapping pulse
            for pulse in wf.signatures.iter_mut() {
                if pulse.start < offset_fc {
                    apply_to_last = true;
                    continue;
                }
                *pulse.increment_oscillator_phase.get_or_insert(0.0) += phase;
                if let Some(incr_phase_param) = parameter.take() {
                    pulse.incr_phase_params.push(incr_phase_param.clone());
                }
                apply_to_last = false;
                break;
            }
            // Insert the frame change into the last pulse anyways, and rewind its
            // own phase by the same amount.
            if apply_to_last {
                if let Some(pulse) = wf.signatures.last_mut() {
                    *pulse.increment_oscillator_phase.get_or_insert(0.0) += phase;
                    *pulse.oscillator_phase.get_or_insert(0.0) -= phase;
                    if let Some(incr_phase_param) = parameter.take() {
                        pulse.incr_phase_params.push(incr_phase_param);
                    }
                }
            }
            return Ok(true);
        }
        Ok(false)
    }
}

fn sort_nodes(nodes: &mut Vec<(Option<u16>, &mut ir::IrNode)>) {
    fn order_node_kind(kind: &ir::NodeKind) -> usize {
        match &kind {
            ir::NodeKind::PlayWave(_) => 1,
            ir::NodeKind::FrameChange(_) => 2,
            _ => 2,
        }
    }
    nodes.sort_by(|a, b| {
        (a.1.offset(), order_node_kind(a.1.data()))
            .cmp(&(b.1.offset(), order_node_kind(b.1.data())))
    });
}

/// Insert frame changes into waveform pulses.
///
/// Iterates the nodes and whenever a waveform with identical oscillator overlaps with a frame change,
/// the frame change node is removed and its phase and sweep parameters (if present) are added to the overlapping pulse.
///
/// # Returns
///
/// Error if one of the following happens:
///     * Frame change overlaps a waveform with different hardware oscillator
///     * Frame change does not overlap with any of the waveforms within a case block
pub fn insert_frame_changes(mut nodes: Vec<(Option<u16>, &mut ir::IrNode)>) -> Result<()> {
    // Sort so that the waveforms are always before frame change if they happen at the same time.
    // This is due to the fact that a frame change which start at the same time a waveform with length 0, can potentially be
    // after the waveform in the source tree.
    // Then whenever a frame change is encountered, a waveform starting at the same time or before
    // is already seen.
    sort_nodes(&mut nodes);
    let mut tracker = FrameChangeTracker::new();
    // Track nodes that are to be removed
    let mut nodes_to_be_removed = vec![];
    for (idx, (state, node)) in nodes.iter_mut().enumerate() {
        let offset = *node.offset();
        match node.data_mut() {
            ir::NodeKind::FrameChange(ob) => {
                if tracker.try_insert_frame_change(
                    state,
                    &ob.signal,
                    offset,
                    ob.phase,
                    &mut ob.parameter,
                )? {
                    nodes_to_be_removed.push(idx);
                }
            }
            ir::NodeKind::PlayWave(ob) => {
                tracker.set_active_waveform(
                    offset,
                    ob.length(),
                    *state,
                    ob.oscillator.as_deref(),
                    ob.waveform
                        .pulses_mut()
                        .expect("Internal error: Waveform pulses must be present"),
                );
            }
            _ => {}
        }
    }
    nodes_to_be_removed.into_iter().for_each(|idx| {
        nodes[idx].1.replace_data(ir::NodeKind::Nop { length: 0 });
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::signature::PulseSignature;
    use std::rc::Rc;

    use super::*;

    fn make_signature(
        start: ir::Samples,
        length: ir::Samples,
        oscillator_phase: Option<f64>,
        increment_oscillator_phase: Option<f64>,
        incr_phase_params: Vec<String>,
    ) -> PulseSignature {
        PulseSignature {
            start,
            pulse: None,
            length,
            amplitude: None,
            phase: 0.0,
            oscillator_phase,
            oscillator_frequency: None,
            increment_oscillator_phase,
            channel: None,
            sub_channel: None,
            id_pulse_params: None,
            markers: vec![],
            preferred_amplitude_register: None,
            incr_phase_params,
        }
    }

    #[test]
    fn test_frame_change_tracker_increment_phase() {
        let signal = Rc::new(cjob::Signal {
            uid: "test".to_string(),
            kind: cjob::SignalKind::IQ,
            channels: vec![0],
            signal_delay: 0,
            start_delay: 0,
            oscillator: None,
            mixer_type: None,
        });
        let mut tracker = FrameChangeTracker::new();
        let pulse_signature = make_signature(0, 50, None, Some(1.0), vec![]);
        let pulse_signature_last = make_signature(60, 30, Some(2.0), None, vec![]);
        let mut signatures = vec![pulse_signature, pulse_signature_last];
        tracker.set_active_waveform(0, 100, None, None, &mut signatures);

        // Frame change into the first pulse
        let inserted0 = tracker
            .try_insert_frame_change(&None, &signal, 0, 1.0, &mut None)
            .unwrap();
        assert!(inserted0);
        // Frame change that overlaps waveform, but not any of the pulses. Must be applied to last.
        tracker
            .try_insert_frame_change(&None, &signal, 99, 0.5, &mut None)
            .unwrap();
        // Test that both phase increments applied to the first pulse
        assert_eq!(signatures[0].increment_oscillator_phase, Some(1.0 + 1.0));
        // The that last pulse gets the last phase increment
        assert_eq!(signatures[1].increment_oscillator_phase, Some(0.5));
        // Test that last pulse oscillator phase is rewound by the incremented amount
        assert_eq!(signatures[1].oscillator_phase, Some(2.0 - 0.5));
    }

    #[test]
    fn test_frame_change_tracker_increment_phase_parameters() {
        let signal = Rc::new(cjob::Signal {
            uid: "test".to_string(),
            kind: cjob::SignalKind::IQ,
            channels: vec![0],
            signal_delay: 0,
            start_delay: 0,
            oscillator: None,
            mixer_type: None,
        });
        let mut tracker = FrameChangeTracker::new();
        let pulse_signature = make_signature(0, 50, None, None, vec!["param0".to_string()]);
        let mut signatures = vec![pulse_signature];
        tracker.set_active_waveform(0, 100, None, None, &mut signatures);
        tracker
            .try_insert_frame_change(&None, &signal, 0, 1.0, &mut Some("param1".to_string()))
            .unwrap();
        assert_eq!(
            signatures[0].incr_phase_params,
            vec!["param0".to_string(), "param1".to_string()]
        );
    }

    #[test]
    fn test_frame_change_not_inserted() {
        let signal = Rc::new(cjob::Signal {
            uid: "test".to_string(),
            kind: cjob::SignalKind::IQ,
            channels: vec![0],
            signal_delay: 0,
            start_delay: 0,
            oscillator: None,
            mixer_type: None,
        });
        let mut tracker = FrameChangeTracker::new();
        let pulse_signature = make_signature(0, 50, None, None, vec!["param0".to_string()]);
        let mut signatures = vec![pulse_signature];

        // Test that if overlapping frame change is recorded before waveform, it is discarded.
        let inserted_before_overlap = tracker
            .try_insert_frame_change(&None, &signal, 15, 1.0, &mut Some("param1".to_string()))
            .unwrap();
        assert!(!inserted_before_overlap);

        tracker.set_active_waveform(10, 100, None, None, &mut signatures);

        // Test frame change does not overlap with the waveform
        let inserted = tracker
            .try_insert_frame_change(&None, &signal, 0, 1.0, &mut Some("param1".to_string()))
            .unwrap();

        assert!(!inserted);
        assert_eq!(signatures[0].increment_oscillator_phase, None);
        assert_eq!(signatures[0].incr_phase_params, vec!["param0".to_string()]);
    }

    #[test]
    fn test_frame_change_osc_switch() {
        let mut tracker = FrameChangeTracker::new();
        let pulse_signature = make_signature(0, 50, None, None, vec![]);
        let mut signatures = vec![pulse_signature];
        tracker.set_active_waveform(0, 10, None, Some("osc0"), &mut signatures);

        let signal_fc = Rc::new(cjob::Signal {
            uid: "test".to_string(),
            kind: cjob::SignalKind::IQ,
            channels: vec![0],
            signal_delay: 0,
            start_delay: 0,
            oscillator: Some(cjob::Oscillator {
                uid: "osc1".to_string(),
                kind: cjob::OscillatorKind::HARDWARE,
            }),
            mixer_type: None,
        });

        // Test error when frame change inside waveform, start/end exclusive
        let result = tracker.try_insert_frame_change(&None, &signal_fc, 5, 1.0, &mut None);
        assert!(result.is_err());

        // Test frame change at waveform start point: No error, not consumed
        let result = tracker.try_insert_frame_change(&None, &signal_fc, 0, 1.0, &mut None);
        assert!(!result.unwrap());

        // Test frame change at waveform end point: No error, not consumed
        let result = tracker.try_insert_frame_change(&None, &signal_fc, 10, 1.0, &mut None);
        assert!(!result.unwrap());
    }
}

// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use crate::Result;
use crate::ir;
use crate::ir::compilation_job::{self as cjob};
use crate::signature::PulseSignature;
use std::collections::HashMap;
use std::ops::Range;
use std::sync::Arc;

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
            signal: Arc::clone(&pulse.signal),
        };
        return Some(frame);
    }
    None
}

/// Transform hardware oscillator phase increment pulses into frame changes
pub(crate) fn handle_frame_changes(node: &mut ir::IrNode) {
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

struct ActiveWaveform<'a> {
    start: ir::Samples,
    length: ir::Samples,
    osc: Option<&'a str>,
    signatures: &'a mut Vec<PulseSignature>,
}

struct FrameChangeTracker<'a> {
    current_waveform: HashMap<Option<u16>, ActiveWaveform<'a>>,
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
            ActiveWaveform {
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

    fn insert_into_last_pulse_signature(
        wf: &mut ActiveWaveform,
        phase: f64,
        parameter: &mut Option<String>,
    ) {
        let Some(pulse) = wf.signatures.last_mut() else {
            panic!("Internal error: found waveform without pulse signature(s).");
        };
        *pulse.increment_oscillator_phase.get_or_insert(0.0) += phase;
        *pulse.oscillator_phase.get_or_insert(0.0) -= phase;
        if let Some(incr_phase_param) = parameter.take() {
            pulse.incr_phase_params.push(incr_phase_param);
        }
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
        node_offset: ir::Samples,
        phase: f64,
        parameter: &mut Option<String>,
    ) -> Result<bool, anyhow::Error> {
        let Some(wf) = self.current_waveform.get_mut(state) else {
            match state {
                Some(s) => panic!("Internal error: no waveform found for case-branch {s}."),
                None => return Ok(false),
            }
        };
        assert!(!wf.signatures.is_empty(), "No pulse signature(s) found.");

        let wf_start = wf.start;
        let wf_range = wf_start..wf_start + wf.length;
        let overlapping_osc = match (wf.osc.as_ref(), signal.oscillator.as_ref()) {
            (Some(&wf_uid), Some(o)) if wf_uid != o.uid => Some((wf_uid, &o.uid)),
            _ => None,
        };

        // Frame-change does not overlap with current waveform
        if !wf_range.contains(&node_offset) {
            // Leave frame-changes outside a match-block as standalone,
            // i.e. to be handled as 0-length command-table entries
            if state.is_none() {
                return Ok(false);
            }
            // Here we are inside a case-branch. Assume it has a duration interval ['a', 'b').
            // Waveforms are merged by state, so there should be one waveform per case-branch starting at 'a'.
            // Nodes have been sorted such that for equal start times, waveforms appear before frame-changes.
            // The earliest start-time for a conditional frame-change is that of its case-branch, i.e. 'a'.
            // Thus, a frame-change reaching this point must be occurring after the current waveform,
            // and we try to insert the frame-change into its last pulse.
            assert!(
                wf_range.end <= node_offset,
                "Internal error: found conditional frame-change starting before its parent case-branch."
            );
            if overlapping_osc.is_none() {
                FrameChangeTracker::insert_into_last_pulse_signature(wf, phase, parameter);
                return Ok(true);
            }
            // Frame-change could not be handled. This stage should be unreachable
            // since every conditional branch should have an available (placeholder) waveform
            let msg = "Cannot handle free-standing oscillator phase change in conditional branch.\n\
            A zero-length 'increment_oscillator_phase' must not occur at the \
            very end of a 'case' block. Consider stretching the branch by \
            adding a small delay after the phase increment.";
            panic!("{}", msg);
        }
        // Check for overlapping oscillators
        // If the frame change happens at the start or end of a waveform, that is ok:
        // we can emit a 0-length command table entry.
        if let Some((wf_osc, uid)) = overlapping_osc {
            if !FrameChangeTracker::point_inside_range(&wf_range, node_offset) {
                return Ok(false);
            }
            let msg = format!(
                "Cannot increment oscillator '{}' of signal '{}': the line is occupied by '{}'",
                uid, signal.uid.0, wf_osc
            );
            anyhow::bail!(msg);
        }
        // Change timing to relative for comparison
        let offset_fc = node_offset - wf_start;
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
        // Insert the phase increment into the last pulse signature, rewinding its own phase by the same amount.
        if apply_to_last {
            FrameChangeTracker::insert_into_last_pulse_signature(wf, phase, parameter);
        }
        Ok(true)
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
pub(crate) fn insert_frame_changes(mut nodes: Vec<(Option<u16>, &mut ir::IrNode)>) -> Result<()> {
    // Sort so that the waveforms are always before frame change if they happen at the same time.
    // This is due to the fact that a frame change with the same starting time as a length-0 waveform, can potentially be
    // after the waveform in the source tree.
    // Then whenever a frame change is encountered, a waveform starting at the same time or before
    // is already seen.
    sort_nodes(&mut nodes);
    let mut tracker = FrameChangeTracker::new();
    // Track nodes that are to be removed
    let mut nodes_to_be_removed = vec![];
    for (idx, (state, node)) in nodes.iter_mut().enumerate() {
        let (node_data, node_offset) = node.data_and_offset_mut();
        match node_data {
            ir::NodeKind::FrameChange(ob) => {
                if tracker.try_insert_frame_change(
                    state,
                    &ob.signal,
                    *node_offset,
                    ob.phase,
                    &mut ob.parameter,
                )? {
                    nodes_to_be_removed.push(idx);
                }
            }
            ir::NodeKind::PlayWave(ob) => {
                tracker.set_active_waveform(
                    *node_offset,
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
        let signal = Arc::new(cjob::Signal {
            uid: 0.into(),
            kind: cjob::SignalKind::IQ,
            channels: vec![0],
            signal_delay: 0,
            start_delay: 0,
            oscillator: None,
            automute: false,
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
        // Test that last pulse gets the last phase increment
        assert_eq!(signatures[1].increment_oscillator_phase, Some(0.5));
        // Test that last pulse oscillator phase is rewound by the incremented amount
        assert_eq!(signatures[1].oscillator_phase, Some(2.0 - 0.5));
    }

    #[test]
    fn test_frame_change_tracker_increment_phase_parameters() {
        let signal = Arc::new(cjob::Signal {
            uid: 0.into(),
            kind: cjob::SignalKind::IQ,
            channels: vec![0],
            signal_delay: 0,
            start_delay: 0,
            oscillator: None,
            automute: false,
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
        let signal = Arc::new(cjob::Signal {
            uid: 0.into(),
            kind: cjob::SignalKind::IQ,
            channels: vec![0],
            signal_delay: 0,
            start_delay: 0,
            oscillator: None,
            automute: false,
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

        let signal_fc = Arc::new(cjob::Signal {
            uid: 0.into(),
            kind: cjob::SignalKind::IQ,
            channels: vec![0],
            signal_delay: 0,
            start_delay: 0,
            oscillator: Some(cjob::Oscillator {
                uid: "osc1".to_string(),
                kind: cjob::OscillatorKind::HARDWARE,
            }),
            automute: false,
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

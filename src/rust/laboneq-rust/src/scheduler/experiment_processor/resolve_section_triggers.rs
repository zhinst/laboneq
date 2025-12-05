// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use crate::error::Result;
use laboneq_scheduler::experiment::ExperimentNode;
use laboneq_scheduler::experiment::types::{Operation, SignalUid};

/// Computes the effective triggers for nested sections in the experiment.
///
/// The effective triggers are derived from the section triggers by considering
/// only the transitions in trigger states. For each signal, only the bits that
/// change from the previous state to the current state are retained in the effective triggers.
///
/// The trigger states are only tracked for nested sections. Parallel sections are not considered.
pub(super) fn resolve_effective_triggers(node: &mut ExperimentNode) -> Result<()> {
    // TODO: It might be more efficient to calculate the effective triggers after scheduling to also
    // take into account parallel sections.
    let mut trackers: HashMap<SignalUid, TriggerStateTracker> = HashMap::new();
    visit_node(node, &mut trackers)
}

fn visit_node(
    node: &mut ExperimentNode,
    trackers: &mut HashMap<SignalUid, TriggerStateTracker>,
) -> Result<()> {
    if let Operation::Section(obj) = &mut node.kind
        && !obj.triggers.is_empty()
    {
        let mut local_trackers = trackers.clone();
        obj.triggers.retain_mut(|trig| {
            let tracker = local_trackers
                .entry(trig.signal)
                .or_insert_with(TriggerStateTracker::new);
            if let Some(effective_bits) = tracker.detect_transitions(trig.state) {
                trig.state = effective_bits;
                true
            } else {
                false
            }
        });

        for child in node.children.iter_mut() {
            visit_node(child.make_mut(), &mut local_trackers)?;
        }
    } else {
        for child in node.children.iter_mut() {
            visit_node(child.make_mut(), trackers)?;
        }
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct TriggerStateTracker {
    state: u8,
}

impl TriggerStateTracker {
    fn new() -> Self {
        Self { state: 0 }
    }

    /// Detects transitions in the trigger state.
    ///
    /// Returns the bits that have transitioned from 0 to 1 from the
    /// previous state to the current state.
    /// The instance keeps track of the last known state.
    /// If no bits have transitioned, returns None.
    fn detect_transitions(&mut self, new_state: u8) -> Option<u8> {
        let transitions = (!self.state) & new_state;
        if transitions != 0 {
            self.state = new_state;
            Some(transitions)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_state_tracker() {
        let mut tracker = TriggerStateTracker::new();
        let changes = [
            (0b00, None),
            (0b01, Some(0b01)),
            (0b01, None),
            (0b11, Some(0b10)),
            (0b10, None),
            (0b00, None),
        ];
        for (i, (state, expected)) in changes.iter().enumerate() {
            let result = tracker.detect_transitions(*state);
            assert_eq!(result, *expected, "{:?}", (i, state));
        }
    }
}

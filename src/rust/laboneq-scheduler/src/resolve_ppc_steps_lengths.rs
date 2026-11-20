// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use laboneq_dsl::types::SignalUid;
use laboneq_ir::IrKind;
use laboneq_units::duration::{Duration, Second, seconds};
use laboneq_units::tinysample::{seconds_to_tinysamples, tiny_samples};

use crate::utils::round_to_grid;
use crate::{ScheduledNode, SignalInfo};

/// We currently assert the PPC trigger for 25 ms. We then reserve a 0.1 ms buffer to guarantee
/// that, if the next sweep step were to occur shortly after, there is sufficient time for PPC
/// to register the falling edge of deassertion.
const PPC_TRIGGER_ASSERT_DURATION: Duration<Second> = seconds(25e-3);
const PPC_TRIGGER_DEASSERT_DURATION: Duration<Second> = seconds(0.1e-3);

/// Sets the lengths of all PPC step nodes in the tree.
pub(crate) fn resolve_ppc_steps_lengths(
    node: &mut ScheduledNode,
    signals: &HashMap<SignalUid, impl SignalInfo>,
) {
    if signals
        .values()
        .all(|signal| signal.amplifier_pump().is_none())
    {
        return;
    }
    round_ppc_steps_lengths(node);
}

/// Rounds the trigger duration of every PPC step to the grid of the preamble it belongs to.
fn round_ppc_steps_lengths(node: &mut ScheduledNode) {
    if let IrKind::LoopIterationPreamble = node.kind {
        let grid = node.schedule.grid;
        for child in node.children.iter_mut() {
            let ScheduledNode { kind, schedule, .. } = &mut child.node;
            if let IrKind::PpcStep(step) = kind {
                let assert_duration = tiny_samples(round_to_grid(
                    seconds_to_tinysamples(PPC_TRIGGER_ASSERT_DURATION).value(),
                    grid.value(),
                ));
                let deassert_duration = tiny_samples(round_to_grid(
                    seconds_to_tinysamples(PPC_TRIGGER_DEASSERT_DURATION).value(),
                    grid.value(),
                ));
                step.trigger_duration = assert_duration;
                schedule.resolve_length(assert_duration + deassert_duration);
            }
        }
        return;
    }
    for child in node.children.iter_mut() {
        if child.node.holds_children() {
            round_ppc_steps_lengths(&mut child.node);
        }
    }
}

#[cfg(test)]
mod tests {
    use laboneq_ir::PpcStep;

    use super::*;
    use crate::schedule_info::ScheduleInfoBuilder;

    #[test]
    fn ppc_steps_take_the_grid_of_their_preamble() {
        let step = PpcStep {
            signal: 1.into(),
            trigger_duration: 0.into(),
            pump_power: None,
            pump_frequency: None,
            probe_power: None,
            probe_frequency: None,
            cancellation_phase: None,
            cancellation_attenuation: None,
        };
        let grid = 51000000;
        let mut preamble = ScheduledNode::new(
            IrKind::LoopIterationPreamble,
            ScheduleInfoBuilder::new().grid(grid).build(),
        );
        preamble.add_child(
            tiny_samples(0),
            ScheduledNode::new(IrKind::PpcStep(step), ScheduleInfoBuilder::new().build()),
        );
        let mut root = ScheduledNode::new(IrKind::Root, ScheduleInfoBuilder::new().build());
        root.add_child(tiny_samples(0), preamble);

        round_ppc_steps_lengths(&mut root);

        let step = &root.children[0].node.children[0].node;
        let IrKind::PpcStep(kind) = &step.kind else {
            unreachable!()
        };

        let assert_tinysamples = seconds_to_tinysamples(PPC_TRIGGER_ASSERT_DURATION).value();
        let deassert_tinysamples = seconds_to_tinysamples(PPC_TRIGGER_DEASSERT_DURATION).value();
        let rounded_assert_tinysamples = round_to_grid(assert_tinysamples, grid);
        let rounded_deassert_tinysamples = round_to_grid(deassert_tinysamples, grid);

        // Double check that the chosen numerics do indeed cause rounding
        assert_ne!(rounded_assert_tinysamples, assert_tinysamples);
        assert_ne!(rounded_deassert_tinysamples, deassert_tinysamples);

        assert_eq!(kind.trigger_duration.value(), rounded_assert_tinysamples);
        assert_eq!(
            step.length().value(),
            rounded_assert_tinysamples + rounded_deassert_tinysamples
        );
    }
}

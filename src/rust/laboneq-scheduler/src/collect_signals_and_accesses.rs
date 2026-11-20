// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use crate::ScheduledNode;

/// Collects the signals and the accesses of every node: its own plus those of its children.
pub(crate) fn collect_signals_and_accesses(node: &mut ScheduledNode) {
    let ScheduledNode {
        schedule, children, ..
    } = node;
    for child in children.iter_mut() {
        if child.node.holds_children() {
            collect_signals_and_accesses(&mut child.node);
            schedule.signals.extend(&child.node.schedule.signals);
        } else {
            schedule.signals.extend(child.node.kind.signals());
        }
        schedule.merge_accesses(&child.node.schedule.accesses);
    }
}

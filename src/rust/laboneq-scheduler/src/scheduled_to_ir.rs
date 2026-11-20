// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use laboneq_ir::node::IrNode;

use crate::ScheduledNode;

pub(crate) fn scheduled_node_to_ir_node(scheduled_node: ScheduledNode) -> IrNode {
    let length = scheduled_node.length();
    let mut ir_node =
        IrNode::new_with_capacity(scheduled_node.kind, length, scheduled_node.children.len());
    for child in scheduled_node.children {
        ir_node.add_child(child.offset, scheduled_node_to_ir_node(child.node));
    }
    ir_node
}

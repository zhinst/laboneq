// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use laboneq_ir::node::IrNode;

use crate::ScheduledNode;
use std::rc::Rc;

pub(crate) fn scheduled_node_to_ir_node(scheduled_node: ScheduledNode) -> IrNode {
    let length = scheduled_node.length();
    let mut ir_node = IrNode::new(scheduled_node.kind, length);
    for child in scheduled_node.children {
        let node = Rc::unwrap_or_clone(child.node);
        ir_node.add_child(child.offset, scheduled_node_to_ir_node(node));
    }
    ir_node
}

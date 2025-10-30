// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::rc::Rc;

use crate::{TinySample, ir::IrKind, schedule_info::ScheduleInfo};

pub type NodeRef = Rc<Node>;

/// A node that hold the scheduling information of an IR node and its children.
#[derive(Debug, Clone, PartialEq)]
pub struct Node {
    pub kind: IrKind,
    pub schedule: ScheduleInfo,
    pub children: Vec<NodeChild>,
}

impl Node {
    pub fn new(kind: IrKind, schedule: ScheduleInfo) -> Self {
        Self {
            kind,
            schedule,
            children: Vec::new(),
        }
    }

    pub fn add_child(&mut self, offset: TinySample, child: Node) {
        self.children.push(NodeChild {
            offset,
            node: Rc::new(child),
        });
    }

    pub fn make_mut(self: &mut Rc<Node>) -> &mut Self {
        Rc::make_mut(self)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct NodeChild {
    pub offset: TinySample,
    pub node: NodeRef,
}

#[cfg(test)]
/// Helper macro to build scheduled IR trees
///
/// Note: `Node` must be in the scope where the macro is used.
/// TODO: Add scheduled information support. Currently only offset is supported.
///
/// ```
/// use crate::scheduled_node::ir_node_structure;
/// use crate::ir::IrKind;
///
/// let tree = ir_node_structure!(
///     IrKind::NotYetImplemented,
///     [
///         (0, IrKind::NotYetImplemented, []),
///         (16, IrKind::NotYetImplemented, [
///             (0, IrKind::NotYetImplemented, []),
///         ])
///     ]
/// );
/// ```
macro_rules! ir_node_structure {
        ($kind:expr, []) => {
            $crate::ScheduledNode::new($kind.clone(), $crate::schedule_info::ScheduleInfoBuilder::default().build())
        };
        ($kind:expr, [ $( ($offset:expr, $child:expr, $subtree:tt) ),* $(,)? ]) => {{
            let mut node = $crate::ScheduledNode::new($kind.clone(), $crate::schedule_info::ScheduleInfoBuilder::default().build());
            $(
                let child = self::ir_node_structure!($child, $subtree);
                node.add_child($offset, child);
            )*
            node
        }};
    }
#[cfg(test)]
pub(crate) use ir_node_structure;

#[cfg(test)]
mod tests {
    use crate::scheduled_node::ir_node_structure;

    use super::*;

    #[test]
    fn test_node_structure_macro() {
        let root = ir_node_structure!(
            IrKind::NotYetImplemented,
            [(
                10,
                IrKind::NotYetImplemented,
                [(0, IrKind::NotYetImplemented, []),]
            )]
        );

        // Test root
        assert_eq!(root.children.len(), 1);
        assert_eq!(root.kind, IrKind::NotYetImplemented);

        // Test children of root
        let node0_children = root.children;
        assert_eq!(node0_children.len(), 1);
        let node1 = &node0_children[0];
        assert_eq!(node1.node.kind, IrKind::NotYetImplemented);
        assert_eq!(node1.offset, 10);

        // Test nested child
        let node2 = &node1.node.children[0];
        assert_eq!(node2.offset, 0);
        assert_eq!(node2.node.kind, IrKind::NotYetImplemented);
    }
}

// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::rc::Rc;

use crate::{ir::IrKind, schedule_info::ScheduleInfo};

pub(crate) type NodeRef = Rc<Node>;

/// A node that hold the scheduling information of an IR node and its children.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct Node {
    pub kind: IrKind,
    pub(crate) schedule: ScheduleInfo,
    pub children: Vec<NodeChild>,
}

impl Node {
    pub(crate) fn new(kind: IrKind, mut schedule: ScheduleInfo) -> Self {
        schedule.signals.extend(kind.signals().into_iter().cloned());
        Self {
            kind,
            schedule,
            children: Vec::new(),
        }
    }

    pub(crate) fn add_child(&mut self, offset: TinySamples, child: Node) {
        self.schedule.signals.extend(&child.schedule.signals);
        self.children.push(NodeChild {
            offset,
            node: Rc::new(child),
        });
    }

    pub(crate) fn make_mut(self: &mut Rc<Node>) -> &mut Self {
        Rc::make_mut(self)
    }

    /// Returns the length of the node.
    pub fn length(&self) -> TinySamples {
        self.schedule
            .try_length()
            .expect("Expected node to have length")
    }

    /// Returns an iterator over the signals used in this node and its children.
    pub fn signals(&self) -> impl Iterator<Item = &SignalUid> {
        self.schedule.signals.iter()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct NodeChild {
    pub(crate) offset: TinySamples,
    pub node: NodeRef,
}

impl NodeChild {
    /// Returns the relative offset from the parent node.
    pub fn offset(&self) -> TinySamples {
        self.offset
    }
}

#[cfg(test)]
/// Helper macro to build scheduled IR trees
///
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
                node.add_child($offset.into(), child);
            )*
            node
        }};
    }
#[cfg(test)]
pub(crate) use ir_node_structure;
use laboneq_dsl::types::SignalUid;
use laboneq_units::tinysample::TinySamples;

#[cfg(test)]
mod tests {
    use crate::scheduled_node::ir_node_structure;
    use laboneq_units::tinysample::tiny_samples;

    use super::*;

    #[test]
    fn test_node_structure_macro() {
        let root = ir_node_structure!(IrKind::Root, [(10, IrKind::Root, [(0, IrKind::Root, []),])]);

        // Test root
        assert_eq!(root.children.len(), 1);
        assert_eq!(root.kind, IrKind::Root);

        // Test children of root
        let node0_children = root.children;
        assert_eq!(node0_children.len(), 1);
        let node1 = &node0_children[0];
        assert_eq!(node1.node.kind, IrKind::Root);
        assert_eq!(node1.offset, tiny_samples(10));

        // Test nested child
        let node2 = &node1.node.children[0];
        assert_eq!(node2.offset, tiny_samples(0));
        assert_eq!(node2.node.kind, IrKind::Root);
    }
}

// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use laboneq_ir::IrKind;

use crate::schedule_info::ScheduleInfo;

/// A node that hold the scheduling information of an IR node and its children.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub(crate) struct Node {
    pub kind: IrKind,
    pub schedule: ScheduleInfo,
    pub children: Vec<NodeChild>,
}

impl Node {
    pub(crate) fn new(kind: IrKind, mut schedule: ScheduleInfo) -> Self {
        schedule.signals.extend(kind.signals().cloned());
        Self {
            kind,
            schedule,
            children: Vec::new(),
        }
    }

    pub(crate) fn new_with_capacity(
        kind: IrKind,
        mut schedule: ScheduleInfo,
        capacity: usize,
    ) -> Self {
        schedule.signals.extend(kind.signals().cloned());
        Self {
            kind,
            schedule,
            children: Vec::with_capacity(capacity),
        }
    }

    pub(crate) fn add_child(&mut self, offset: TinySamples, child: Node) {
        self.children.push(NodeChild {
            offset,
            node: child,
        });
    }

    /// Returns whether the node is of a kind that holds children, even if it holds none.
    pub(crate) fn holds_children(&self) -> bool {
        match self.kind {
            IrKind::Root
            | IrKind::Section(_)
            | IrKind::Loop(_)
            | IrKind::LoopIteration
            | IrKind::LoopIterationPreamble
            | IrKind::Match(_)
            | IrKind::Case(_) => true,
            IrKind::InitialOscillatorFrequency(_)
            | IrKind::PlayPulse(_)
            | IrKind::ChangeOscillatorPhase(_)
            | IrKind::Acquire(_)
            | IrKind::SetOscillatorFrequency(_)
            | IrKind::ResetOscillatorPhase { .. }
            | IrKind::PpcStep(_)
            | IrKind::Delay { .. }
            | IrKind::ClearPrecompensation { .. }
            | IrKind::Expect(_)
            | IrKind::MarkStale(_) => false,
        }
    }

    /// Returns the length of the node.
    pub(crate) fn length(&self) -> TinySamples {
        self.schedule
            .try_length()
            .expect("Expected node to have length")
    }

    /// Takes the children of the node, leaving it with an empty children vector.
    pub(crate) fn take_children(&mut self) -> Vec<NodeChild> {
        std::mem::take(&mut self.children)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct NodeChild {
    pub offset: TinySamples,
    pub node: Node,
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
use laboneq_units::tinysample::TinySamples;

#[cfg(test)]
mod tests {
    use laboneq_units::tinysample::tiny_samples;

    use super::*;
    use crate::scheduled_node::ir_node_structure;

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

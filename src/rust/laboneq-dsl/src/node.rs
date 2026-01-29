// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

pub type NodeChild<T> = Arc<Node<T>>;

/// A generic node in a tree.
///
/// `T` is the type of the node's kind.
#[derive(Debug, Clone, PartialEq)]
pub struct Node<T>
where
    T: Clone,
{
    pub kind: T,
    pub children: Vec<NodeChild<T>>,
}

impl<T> Node<T>
where
    T: Clone,
{
    pub fn new(kind: T) -> Self {
        Self {
            kind,
            children: Vec::new(),
        }
    }

    pub fn make_mut(self: &mut Arc<Node<T>>) -> &mut Self {
        Arc::make_mut(self)
    }

    pub fn as_ptr(self: &Arc<Node<T>>) -> *const Node<T> {
        Arc::as_ptr(self)
    }

    pub fn take_children(&mut self) -> Vec<NodeChild<T>> {
        std::mem::take(&mut self.children)
    }
}

/// Helper macro to build trees
///
/// Note: `Node` must be in the scope where the macro is used.
///
/// Example usage:
///
/// ```
/// use laboneq_dsl::node_structure;
///
/// let tree = node_structure!(
///     "root",
///     [
///         ("child1", []),
///         ("child2", [
///             ("grandchild1", []),
///         ])
///     ]
/// );
/// ```
#[macro_export]
macro_rules! node_structure {
    ($value:expr, []) => {
        $crate::node::Node::new($value.clone())
    };
    ($value:expr, [ $( ($child:expr, $subtree:tt) ),* $(,)? ]) => {{
        let mut node = $crate::node::Node::new($value.clone());
        $(
            let child = node_structure!($child, $subtree);
            node.children.push(child.into());
        )*
        node
    }};
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_node_structure_macro() {
        let root = node_structure!(0, [(1, []), (2, [(3, []),])]);

        // Test root
        assert_eq!(root.children.len(), 2);
        let node0 = &root.kind;
        assert_eq!(*node0, 0);

        // Test children of root
        let node0_children = root.children;
        assert_eq!(node0_children.len(), 2);
        let node1 = &node0_children[0];

        assert!(matches!(node1.kind, 1));
        assert_eq!(node1.children.len(), 0);
        let node2 = &node0_children[1];
        assert!(matches!(node2.kind, 2));
        assert_eq!(node2.children.len(), 1);

        // Test child of second child of root
        assert!(matches!(node2.children[0].kind, 3));
    }
}

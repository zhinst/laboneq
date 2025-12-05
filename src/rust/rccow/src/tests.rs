// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use super::*;
use std::cell::RefCell;
use std::rc::Rc;

thread_local!(static CLONE_COUNT: RefCell<usize> = const { RefCell::new(0) });

#[derive(Debug)]
struct TestNode {
    value: i32,
    children: Vec<Rc<TestNode>>,
}

impl Clone for TestNode {
    fn clone(&self) -> Self {
        CLONE_COUNT.with(|c| *c.borrow_mut() += 1);
        Self {
            value: self.value,
            children: self.children.clone(),
        }
    }
}

impl GetChildren for TestNode {
    fn get_child(&self, index: usize) -> &Rc<Self> {
        &self.children[index]
    }

    fn get_child_mut(&mut self, index: usize) -> &mut Rc<Self> {
        &mut self.children[index]
    }

    fn iter_children(&self) -> impl Iterator<Item = &Rc<Self>> + '_ {
        self.children.iter()
    }
}

fn reset_clone_count() {
    CLONE_COUNT.with(|c| *c.borrow_mut() = 0);
}

fn get_clone_count() -> usize {
    CLONE_COUNT.with(|c| *c.borrow())
}

fn leaf(value: i32) -> Rc<TestNode> {
    Rc::new(TestNode {
        value,
        children: vec![],
    })
}

fn node_with_children(value: i32, children: Vec<Rc<TestNode>>) -> Rc<TestNode> {
    Rc::new(TestNode { value, children })
}

fn simple_tree() -> Rc<TestNode> {
    // Tree structure:
    //     10
    //    /  \
    //   20  21
    node_with_children(10, vec![leaf(20), leaf(21)])
}

fn deep_tree() -> Rc<TestNode> {
    // Tree structure:
    //       10
    //      /  \
    //     20  21
    //    /  \
    //   30  31
    let child1 = node_with_children(20, vec![leaf(30), leaf(31)]);
    let child2 = leaf(21);
    node_with_children(10, vec![child1, child2])
}

// === Basic Operations Tests ===

#[test]
fn test_deref_root() {
    let mut root = simple_tree();
    let cow = RcCow::root(&mut root);

    assert_eq!(cow.value, 10);
    assert_eq!(cow.children.len(), 2);
}

#[test]
fn test_deref_ref() {
    let mut root = simple_tree();
    let mut cow = RcCow::root(&mut root);
    let child = cow.descend(0);

    assert!(matches!(child, RcCow::Ref { .. }));
    assert_eq!(child.value, 20);
}

// === Tree Traversal Tests ===

#[test]
fn test_descend_single_level() {
    let mut root = simple_tree();
    let mut cow = RcCow::root(&mut root);

    let child0 = cow.descend(0);
    assert_eq!(child0.value, 20);

    let child1 = cow.descend(1);
    assert_eq!(child1.value, 21);
}

#[test]
fn test_descend_multiple_levels() {
    let mut root = deep_tree();
    let mut cow = RcCow::root(&mut root);

    let mut child = cow.descend(0);
    assert_eq!(child.value, 20);

    let grandchild = RcCow::descend(&mut child, 0);
    assert_eq!(grandchild.value, 30);
}

#[test]
#[should_panic]
fn test_descend_invalid_index() {
    let mut root = simple_tree();
    let mut cow = RcCow::root(&mut root);

    let _ = cow.descend(10);
}

#[test]
fn test_iter_children_count() {
    let mut root = simple_tree();
    let mut cow = RcCow::root(&mut root);

    let children: Vec<_> = RcCow::iter(&mut cow).collect();
    assert_eq!(children.len(), 2);
}

#[test]
fn test_iter_children_values() {
    let mut root = simple_tree();
    let mut cow = RcCow::root(&mut root);

    let values: Vec<_> = RcCow::iter(&mut cow).map(|c| c.value).collect();
    assert_eq!(values, vec![20, 21]);
}

// === Copy-on-Write Tests ===

#[test]
fn test_promote_unshared_node() {
    reset_clone_count();
    let mut root = simple_tree();
    assert_eq!(Rc::strong_count(&root), 1);

    let mut cow = RcCow::root(&mut root);
    let node_mut = cow.promote();
    node_mut.value = 999;

    // Should have modified in-place (no clone)
    assert_eq!(get_clone_count(), 0);
    assert_eq!(root.value, 999);
}

#[test]
fn test_promote_shared_node() {
    reset_clone_count();
    let mut root = simple_tree();
    let original = Rc::clone(&root);

    let mut cow = RcCow::root(&mut root);
    let node_mut = cow.promote();
    node_mut.value = 999;

    // Should have cloned once
    assert_eq!(get_clone_count(), 1);

    // Original should be unchanged
    assert_eq!(original.value, 10);

    // New root should be changed
    assert_eq!(root.value, 999);
}

#[test]
fn test_promote_root() {
    reset_clone_count();
    let mut root = simple_tree();
    let original = Rc::clone(&root);

    let mut cow = RcCow::root(&mut root);
    cow.promote().value = 999;

    assert_eq!(original.value, 10);
    assert_eq!(root.value, 999);
}

#[test]
fn test_promote_deep_child() {
    reset_clone_count();
    let mut root = deep_tree();
    let original = Rc::clone(&root);

    let mut cow = RcCow::root(&mut root);
    let mut child = cow.descend(0);
    let mut grandchild = RcCow::descend(&mut child, 0);

    grandchild.promote().value = 999;

    // Should have cloned: root -> child -> grandchild (3 clones)
    assert_eq!(get_clone_count(), 3);

    // Original tree unchanged
    assert_eq!(original.value, 10);
    assert_eq!(original.children[0].value, 20);
    assert_eq!(original.children[0].children[0].value, 30);

    // New tree changed
    assert_eq!(root.children[0].children[0].value, 999);
}

#[test]
fn test_promote_clones_minimal_path() {
    reset_clone_count();
    let mut root = deep_tree();
    let sibling_backup = Rc::clone(&root.children[0].children[1]);

    let mut cow = RcCow::root(&mut root);
    let mut child = cow.descend(0);
    let mut grandchild = RcCow::descend(&mut child, 0);

    grandchild.promote().value = 999;

    // Sibling node should NOT have been cloned (same Rc)
    assert!(Rc::ptr_eq(&sibling_backup, &root.children[0].children[1]));
}

// === Reference Counting Tests ===

#[test]
fn test_refcount_unchanged_on_descend() {
    let mut root = simple_tree();
    let child_rc = Rc::clone(&root.children[0]);
    let initial_count = Rc::strong_count(&child_rc);

    let mut cow = RcCow::root(&mut root);
    let _child = cow.descend(0);

    // Refcount should not increase from descending
    assert_eq!(Rc::strong_count(&child_rc), initial_count);
}

#[test]
fn test_refcount_after_promote() {
    let mut root = simple_tree();
    let _backup = Rc::clone(&root);

    let mut cow = RcCow::root(&mut root);
    cow.promote();

    // After promotion, root should have refcount 1 (backup still has old root)
    assert_eq!(Rc::strong_count(&root), 1);
}

#[test]
fn test_original_tree_unchanged() {
    let mut root = deep_tree();
    let original = Rc::clone(&root);

    let mut cow = RcCow::root(&mut root);
    let mut child = cow.descend(0);
    let mut grandchild = RcCow::descend(&mut child, 1);

    grandchild.promote().value = 777;

    // Original tree structure completely unchanged
    assert_eq!(original.value, 10);
    assert_eq!(original.children[0].value, 20);
    assert_eq!(original.children[0].children[0].value, 30);
    assert_eq!(original.children[0].children[1].value, 31);
    assert_eq!(original.children[1].value, 21);

    // New tree has the change
    assert_eq!(root.children[0].children[1].value, 777);
}

// === Multiple Mutations Tests ===

#[test]
fn test_multiple_promotes_same_node() {
    reset_clone_count();
    let mut root = simple_tree();
    let _backup = Rc::clone(&root);

    let mut cow = RcCow::root(&mut root);
    cow.promote().value = 100;
    cow.promote().value = 200;
    cow.promote().value = 300;

    // Should only clone once (first promote)
    assert_eq!(get_clone_count(), 1);
    assert_eq!(root.value, 300);
}

#[test]
fn test_promote_different_branches() {
    reset_clone_count();
    let mut root = deep_tree();
    let _backup = Rc::clone(&root);

    let mut cow = RcCow::root(&mut root);

    // Promote left branch
    let mut left = cow.descend(0);
    let mut left_child = RcCow::descend(&mut left, 0);
    left_child.promote().value = 300;

    // Promote right branch
    let mut right = cow.descend(1);
    right.promote().value = 210;

    assert_eq!(root.children[0].children[0].value, 300);
    assert_eq!(root.children[1].value, 210);
}

#[test]
fn test_sequential_mutations() {
    let mut root = deep_tree();

    // First mutation
    {
        let mut cow = RcCow::root(&mut root);
        cow.promote().value = 100;
    }

    // Second mutation on same tree
    {
        let mut cow = RcCow::root(&mut root);
        let mut child = cow.descend(0);
        child.promote().value = 200;
    }

    assert_eq!(root.value, 100);
    assert_eq!(root.children[0].value, 200);
}

// === Edge Cases ===

#[test]
fn test_empty_children() {
    let mut root = leaf(42);
    let mut cow = RcCow::root(&mut root);

    let children: Vec<_> = RcCow::iter(&mut cow).collect();
    assert_eq!(children.len(), 0);

    cow.promote().value = 999;
    assert_eq!(root.value, 999);
}

#[test]
fn test_single_child() {
    let mut root = node_with_children(10, vec![leaf(20)]);
    let mut cow = RcCow::root(&mut root);

    let mut child = cow.descend(0);
    child.promote().value = 999;

    assert_eq!(root.children[0].value, 999);
}

#[test]
fn test_deeply_nested_tree() {
    // Create a 10-level deep tree
    let mut current = leaf(10);
    for i in 1..10 {
        current = node_with_children(i, vec![current]);
    }

    let mut root = current;
    let _backup = Rc::clone(&root);

    // Navigate to the deepest node and mutate it

    let mut cow = RcCow::root(&mut root);
    let mut child1 = cow.descend(0);
    let mut child2 = RcCow::descend(&mut child1, 0);
    let mut child3 = RcCow::descend(&mut child2, 0);
    let mut child4 = RcCow::descend(&mut child3, 0);
    let mut child5 = RcCow::descend(&mut child4, 0);
    let mut child6 = RcCow::descend(&mut child5, 0);
    let mut child7 = RcCow::descend(&mut child6, 0);
    let mut child8 = RcCow::descend(&mut child7, 0);
    let mut child9 = RcCow::descend(&mut child8, 0);
    child9.promote().value = 999;

    // Verify the deep change propagated
    let mut node = &root;
    for _ in 0..9 {
        node = &node.children[0];
    }
    assert_eq!(node.value, 999);
}

#[test]
fn test_wide_tree() {
    // Create a node with many children
    let children: Vec<_> = (0..100).map(leaf).collect();
    let mut root = node_with_children(999, children);

    let mut cow = RcCow::root(&mut root);
    let count = RcCow::iter(&mut cow).count();
    assert_eq!(count, 100);

    // Modify a child in the middle
    let mut child = cow.descend(50);
    child.promote().value = 5000;

    assert_eq!(root.children[50].value, 5000);
}

// === Iterator-specific Tests ===

#[test]
fn test_iter_vs_manual_descend() {
    let mut root = simple_tree();

    let iter_values: Vec<_> = {
        let mut cow = RcCow::root(&mut root);
        RcCow::iter(&mut cow).map(|c| c.value).collect()
    };

    let manual_values: Vec<_> = {
        let num_children = root.children.len();
        let mut cow = RcCow::root(&mut root);
        (0..num_children).map(|i| cow.descend(i).value).collect()
    };

    assert_eq!(iter_values, manual_values);
}

#[test]
fn test_iter_empty_node() {
    let mut root = leaf(42);
    let mut cow = RcCow::root(&mut root);

    let count = RcCow::iter(&mut cow).count();
    assert_eq!(count, 0);
}

#[test]
fn test_iter_then_promote() {
    reset_clone_count();
    let mut root = simple_tree();
    let _backup = Rc::clone(&root);

    let mut cow = RcCow::root(&mut root);

    for mut child in RcCow::iter(&mut cow) {
        if child.value == 20 {
            child.promote().value = 2000;
        }
    }

    assert_eq!(root.children[0].value, 2000);
    assert_eq!(root.children[1].value, 21);
}

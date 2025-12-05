// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use rccow::{GetChildren, RcCow};
use std::cell::RefCell;
use std::fmt;
use std::rc::Rc;

/// Our example node data type.
#[derive(Debug)]
struct Node {
    data: i32,
    children: Vec<(i64, Rc<Node>)>,
}

/// `RcCow` needs the node to implement this trait so it can walk the tree.
impl GetChildren for Node {
    fn get_child(&self, index: usize) -> &Rc<Self> {
        &self.children[index].1
    }

    fn get_child_mut(&mut self, index: usize) -> &mut Rc<Self> {
        &mut self.children[index].1
    }

    fn iter_children(&self) -> impl Iterator<Item = &Rc<Self>> + '_ {
        self.children.iter().map(|(_, c)| c)
    }
}

thread_local!(static LOG_RECORDS: RefCell<Vec<String>> = RefCell::new(Vec::with_capacity(3)));

/// Implementation that logs each clone operation (for test only)
impl Clone for Node {
    fn clone(&self) -> Self {
        LOG_RECORDS.with_borrow_mut(|l| l.push(format!("cloning Node(data: {})", self.data)));
        Self {
            data: self.data,
            children: self.children.clone(),
        }
    }
}

impl fmt::Display for Node {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fn pretty_print(node: &Node) -> String {
            let mut result = String::new();

            result.push_str(&format!("Node(data: {})\n", node.data));

            for (key, child) in node.children.iter() {
                result.push_str(&format!("├─ [{}] ", key));
                let child_lines = pretty_print(child);
                let lines: Vec<&str> = child_lines.lines().collect();
                if let Some(first_line) = lines.first() {
                    result.push_str(first_line);
                    result.push('\n');

                    for line in &lines[1..] {
                        result.push_str(&format!("│  {}\n", line.trim_start()));
                    }
                }
            }

            result
        }

        write!(f, "{}", pretty_print(self))
    }
}

fn example_tree() -> Rc<Node> {
    // Level 3 (leaf nodes)
    let leaf1 = Rc::new(Node {
        data: 30,
        children: vec![],
    });

    let leaf2 = Rc::new(Node {
        data: 31,
        children: vec![],
    });

    let leaf3 = Rc::new(Node {
        data: 32,
        children: vec![],
    });

    let leaf4 = Rc::new(Node {
        data: 33,
        children: vec![],
    });

    // Level 2 (intermediate nodes)
    let child1 = Rc::new(Node {
        data: 20,
        children: vec![(0, leaf1), (1, leaf2)],
    });

    let child2 = Rc::new(Node {
        data: 21,
        children: vec![(0, leaf3), (1, leaf4)],
    });

    // Level 1 (root node)
    Rc::new(Node {
        data: 10,
        children: vec![(0, child1), (1, child2)],
    })
}

#[test]
fn test() {
    main()
}

fn main() {
    // create example tree
    let mut root = example_tree();

    println!("Tree structure:");
    println!("{}", root);
    let root_backup = Rc::clone(&root);

    fn visit(node: &mut RcCow<Node>) {
        if node.data > 30 && node.data < 32 {
            let node = node.promote();
            node.data *= 2000000;
        }
        // // Alternative: use `RcCow::iter()`
        // for mut c in RcCow::iter(node) {
        //     visit(&mut c)
        // }
        for i in 0..node.children.len() {
            let mut child = RcCow::descend(node, i);
            visit(&mut child);
        }
    }
    // Create RcCow from root
    let mut cow_root = RcCow::root(&mut root);
    visit(&mut cow_root);

    println!("Tree structure after visitation:");
    println!("{}", root);
    LOG_RECORDS.with_borrow(|log_records| {
        assert_eq!(
            log_records,
            &[
                "cloning Node(data: 10)",
                "cloning Node(data: 20)",
                "cloning Node(data: 31)"
            ]
        )
    });
    println!("Previous tree is still available:");
    println!("{}", root_backup);
}

// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::rc::Rc;
use std::{cell::RefCell, ops::Deref};

use crate::shfppc_sweeper_config::SHFPPCSweeperConfig;
use codegenerator::ir::experiment::SweepCommand;

enum SweepNode {
    SweepCommand(SweepCommand),
    SweepStackFrame(Rc<RefCell<SweepStackFrame>>),
}

struct SweepStackFrame {
    count: u64,
    items: Vec<Rc<RefCell<SweepNode>>>,
}

enum SweepConfigNode {
    SweepCommand(SweepCommand),
    SHFPPCSweeperConfig(SHFPPCSweeperConfig),
}

pub(crate) struct SHFPPCSweeperConfigTracker {
    command_tree: Rc<RefCell<SweepStackFrame>>,
    stack: Vec<Rc<RefCell<SweepStackFrame>>>,
    finished: bool,
}

impl SHFPPCSweeperConfigTracker {
    pub fn new() -> Self {
        let command_tree = Rc::new(RefCell::new(SweepStackFrame {
            count: 1,
            items: Vec::new(),
        }));
        Self {
            command_tree: Rc::clone(&command_tree),
            stack: vec![command_tree],
            finished: false,
        }
    }

    fn current_frame(&self) -> &Rc<RefCell<SweepStackFrame>> {
        self.stack
            .last()
            .expect("Internal error: No stack frame available")
    }

    pub fn add_step(&mut self, sweep_command: SweepCommand) {
        self.current_frame()
            .borrow_mut()
            .items
            .push(Rc::new(RefCell::new(SweepNode::SweepCommand(
                sweep_command,
            ))));
    }

    pub fn enter_loop(&mut self, count: u64) {
        let new_frame = Rc::new(RefCell::new(SweepStackFrame {
            count,
            items: Vec::new(),
        }));
        let new_node = Rc::new(RefCell::new(SweepNode::SweepStackFrame(Rc::clone(
            &new_frame,
        ))));
        self.current_frame().borrow_mut().items.push(new_node);
        self.stack.push(new_frame);
    }

    pub fn exit_loop(&mut self) {
        let closed_frame = self.stack.pop().expect("Internal error: No frame to close");
        // If the frame we are closing is empty, we don't need to keep it
        if closed_frame.borrow().items.is_empty() {
            self.current_frame().borrow_mut().items.pop();
        }
    }

    /// Get the final SHFPPCSweeperConfig from the command tree.
    ///
    /// After calling this function, the command tree will be empty.
    /// This function may be called only once at the end of the configuration.
    ///
    pub fn finish(&mut self) -> Option<SHFPPCSweeperConfig> {
        assert!(
            !self.finished,
            "Internal error: finish() called multiple times"
        );
        self.finished = true;
        let (commands, count) = flatten(&self.command_tree);
        if commands.is_empty() {
            return None;
        }
        Some(SHFPPCSweeperConfig { count, commands })
    }
}

/// Remove any nesting from the command tree
///
/// After flattening, `items` will only hold `SweepCommand`s. The repetition
/// count may get adjusted to efficiently represent any internal loops.
///
fn flatten(command_tree: &Rc<RefCell<SweepStackFrame>>) -> (Vec<SweepCommand>, u64) {
    let command_tree = command_tree.borrow_mut();
    let mut count = command_tree.count;
    let mut commands = command_tree
        .items
        .iter()
        .map(|c| match c.borrow().deref() {
            SweepNode::SweepStackFrame(frame) => {
                assert!(
                    !frame.borrow().items.is_empty(),
                    "Internal error: Empty frames should have been dropped earlier"
                );
                let (commands, count) = flatten(frame);
                SweepConfigNode::SHFPPCSweeperConfig(SHFPPCSweeperConfig { count, commands })
            }
            SweepNode::SweepCommand(command) => SweepConfigNode::SweepCommand(command.clone()),
        })
        .collect::<Vec<_>>();

    let commands = match commands.as_mut_slice() {
            [SweepConfigNode::SHFPPCSweeperConfig(_)] => {
                if let SweepConfigNode::SHFPPCSweeperConfig(child_config) = commands.pop().unwrap() {
                    count *= child_config.count;
                    child_config.commands
                } else {
                    unreachable!("Internal error: Expected SHFPPCSweeperConfig")
                }
            }
            _ => commands
                .into_iter()
                .map(|node| match node {
                    SweepConfigNode::SweepCommand(command) => command,
                    _ => panic!("Internal error: Items must be SweepCommand. Nesting rolled loops is not supported."),
                })
                .collect(),
        };
    (commands, count)
}

// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use crate::ir::compilation_job::DeviceKind;
use crate::ir::{IrNode, NodeKind, QaEvent, Samples};
use crate::{Error, Result};
use std::collections::HashMap;

struct AcquireContext<'a> {
    nodes: HashMap<Samples, Vec<&'a mut IrNode>>,
    has_acquire: bool,
}

impl AcquireContext<'_> {
    fn new() -> Self {
        AcquireContext {
            nodes: HashMap::new(),
            has_acquire: false,
        }
    }
}

fn collect_nodes<'a>(node: &'a mut IrNode, ctx: &mut AcquireContext<'a>) {
    match node.data() {
        NodeKind::Acquire(_) => {
            ctx.has_acquire = true;
            ctx.nodes.entry(*node.offset()).or_default().push(node);
        }
        NodeKind::PlayWave(_) => {
            ctx.nodes.entry(*node.offset()).or_default().push(node);
        }
        NodeKind::PlayHold(_) => {
            // Current logic does not support PlayHold for SHFQA devices.
            // If play wave compression happens on QA, it should not produce `PlayHold` nodes.
            panic!("Internal error: PlayHold is not supported for SHFQA device");
        }
        _ => {
            for child in node.iter_children_mut() {
                collect_nodes(child, ctx);
            }
        }
    }
}

fn validate_qa_event(event: &QaEvent, disallow_standalone_play: bool) -> Result<()> {
    if !event.play_waves().is_empty() && event.acquires().is_empty() && disallow_standalone_play {
        let msg = format!(
            "Play and acquire must happen at the same time on device SHFQA. Invalid play timing on signal(s): {}",
            event
                .play_waves()
                .iter()
                .map(|w| w
                    .signals
                    .iter()
                    .map(|s| s.uid.0.to_string())
                    .collect::<String>())
                .collect::<Vec<_>>()
                .join(", ")
        );
        return Err(Error::new(&msg));
    }
    Ok(())
}

/// Create a [`QaEvent`] from the given nodes.
///
/// Overwrites the first node in the `nodes` vector with the new [`QaEvent`], which
/// consumes the nodes in the vector.
fn create_qa_event(nodes: &mut Vec<&mut IrNode>, disallow_standalone_play: bool) -> Result<()> {
    let mut acquires = Vec::with_capacity(nodes.len());
    let mut play_waves = Vec::with_capacity(nodes.len());
    for node in nodes.iter_mut() {
        let node_kind = node.swap_data(NodeKind::Nop { length: 0 });
        match node_kind {
            NodeKind::Acquire(ob) => {
                acquires.push(ob);
            }
            NodeKind::PlayWave(ob) => {
                play_waves.push(ob);
            }
            _ => {
                unreachable!();
            }
        }
    }
    let event = QaEvent::new(acquires, play_waves);
    validate_qa_event(&event, disallow_standalone_play)?;
    nodes[0].swap_data(NodeKind::QaEvent(event));
    Ok(())
}

/// Handle QA events in the IR node for SHFQA devices.
///
/// This function processes the IR node to create [`NodeKind::QaEvent`] nodes
/// replacing the [`NodeKind::Acquire`] and [`NodeKind::PlayWave`] nodes.
///
/// # Returns
///
/// `Ok(())`: Processing is successful.
/// `Error`: There are acquire events, but one or more play events have no corresponding acquire events
///         happening at the same time.
pub(crate) fn handle_qa_events(node: &mut IrNode, device: &DeviceKind) -> Result<()> {
    if device != &DeviceKind::SHFQA {
        return Ok(());
    }
    let mut acquire_info = AcquireContext::new();
    collect_nodes(node, &mut acquire_info);
    for mut nodes in acquire_info.nodes.into_values() {
        if nodes.is_empty() {
            continue;
        }
        create_qa_event(&mut nodes, acquire_info.has_acquire)?;
    }
    Ok(())
}

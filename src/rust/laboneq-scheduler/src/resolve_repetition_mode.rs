// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use laboneq_dsl::types::SectionUid;

use crate::error::{Error, Result};
use crate::ir::IrKind;
use crate::schedule_info::RepetitionMode;
use crate::{ScheduleInfo, ScheduledNode};

/// Assign the repetition mode to the loop that corresponds to the "shot boundary".
///
/// If a node has a repetition mode other than `Fastest`, it ensures that the loop chain is linear
/// and assigns the repetition mode to the innermost sweep loop found in its children.
pub(crate) fn resolve_repetition_mode(node: &mut ScheduledNode) -> Result<()> {
    if let Some(repetition_mode) = node.schedule.repetition_mode {
        if repetition_mode != RepetitionMode::Fastest {
            let mut innermost_loop = None;
            for child in node.children.iter_mut() {
                innermost_loop = find_innermost_sweep(child.node.make_mut())?;
            }
            if let Some(innermost) = innermost_loop {
                // Move repetition mode to innermost sweep loop
                innermost.repetition_mode = Some(repetition_mode);
                node.schedule.repetition_mode = None;
                ensure_linear_chain_until_shot_loop(node, None)?;
            }
        }
        // Nothing to do here
        return Ok(());
    }
    for child in node.children.iter_mut() {
        resolve_repetition_mode(child.node.make_mut())?;
    }
    Ok(())
}

/// Find the innermost sweep loops in the scheduled node tree.
fn find_innermost_sweep(node: &mut ScheduledNode) -> Result<Option<&mut ScheduleInfo>> {
    let mut innermost_sweep = None;
    if let IrKind::Loop(_) = &node.kind {
        innermost_sweep = Some(&mut node.schedule);
    }
    for child in node.children.iter_mut() {
        if let Some(innermost) = find_innermost_sweep(child.node.make_mut())? {
            innermost_sweep = Some(innermost);
        }
    }
    Ok(innermost_sweep)
}

/// Ensure the section chain is linear until the shot loop.
///
/// A linear chain means that there is at most one section or sweep loop at each level
/// until the shot loop is reached. If multiple sections or sweeps are found, an error is returned.
fn ensure_linear_chain_until_shot_loop(
    node: &ScheduledNode,
    mut parent_uid: Option<SectionUid>,
) -> Result<()> {
    if let Some(parent) = node.kind.section_info() {
        parent_uid = Some(*parent.uid);
    }
    let mut node_count = 0;
    for child in node.children.iter() {
        let mut shot_loop_found = false;
        match &child.node.kind {
            IrKind::Loop(_) => {
                node_count += 1;
                // We have reached the shot loop, stop skip children
                shot_loop_found = child.node.schedule.repetition_mode.is_some();
            }
            // Skip preamble nodes
            IrKind::LoopIterationPreamble => {}
            _ => {
                node_count += 1;
            }
        }
        // Recurse into child nodes if shot loop not found
        if !shot_loop_found {
            ensure_linear_chain_until_shot_loop(&child.node, parent_uid)?;
        }
    }
    if node_count > 1 {
        let msg = format!(
            "Mixing both sections and sweeps or multiple sweeps in '{}' is not permitted unless the repetition mode is set to 'fastest'. \
            Use only either sections or a single sweep.",
            parent_uid.unwrap().0,
        );
        return Err(Error::new(msg));
    }
    Ok(())
}

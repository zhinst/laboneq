// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use crate::error::{Error, Result};
use crate::experiment::ExperimentNode;
use crate::experiment::types::{Operation, RepetitionMode, SectionUid};

#[derive(Debug, Clone, PartialEq)]
pub struct RepetitionInfo {
    pub mode: RepetitionMode,
    /// Loop which corresponds to the shot boundary.
    /// This section shall be padded to the repetition length.
    pub loop_uid: SectionUid,
}

/// Locate the loop section which corresponds to the shot boundary.
pub(crate) fn resolve_repetition_time(node: &ExperimentNode) -> Result<Option<RepetitionInfo>> {
    if let Operation::AveragingLoop(obj) = &node.kind
        && obj.repetition_mode != RepetitionMode::Fastest
    {
        let shot_loop = find_innermost_sweep(node)?.unwrap_or(&obj.uid);
        return Ok(RepetitionInfo {
            mode: obj.repetition_mode,
            loop_uid: *shot_loop,
        }
        .into());
    }
    for child in node.children.iter() {
        if let Some(s) = resolve_repetition_time(child)? {
            return Ok(Some(s));
        }
    }
    Ok(None)
}

/// Find the innermost sweep loop in the node's children.
///
/// Also validate that no sections are mixed with sweeps.
fn find_innermost_sweep(node: &ExperimentNode) -> Result<Option<&SectionUid>> {
    let mut innermost_loop = None;
    for child in node.children.iter() {
        if let Some(loop_info) = &child.kind.loop_info() {
            if node.children.len() > 1 {
                let msg = format!(
                    "Mixing both sections and sweeps or multiple sweeps in '{}' is not permitted unless the repetition mode is set to 'fastest'. \
                    Use only either sections or a single sweep.",
                    {
                        node.kind
                            .section_info()
                            .map(|s| s.uid.0)
                            .expect("Internal error: Expected a section")
                    },
                );
                return Err(Error::new(&msg));
            }
            innermost_loop = Some(loop_info.uid);
        };
        if let Some(s) = find_innermost_sweep(child)? {
            innermost_loop = Some(s);
        }
    }
    Ok(innermost_loop)
}

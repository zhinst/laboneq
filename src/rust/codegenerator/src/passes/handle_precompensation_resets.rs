// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use crate::Result;
use crate::ir;
use std::collections::HashSet;

const PRECOMP_RESET_LENGTH: i64 = 32;

/// Transform precompensation reset nodes from IR to AWG commands
pub(crate) fn handle_precompensation_resets(
    node: &mut ir::IrNode,
    cut_points: &mut HashSet<ir::Samples>,
) -> Result<()> {
    match node.data_mut() {
        ir::NodeKind::PrecompensationFilterReset { .. } => {
            // We clear the precompensation filters in a dedicated command table entry.
            // Currently, a bug (HULK-1246) prevents us from doing so in a zero-length
            // command, so instead we allocate 32 samples (minimum waveform length) for this, and
            // register the end of this interval as a cut point.
            cut_points.insert(*node.offset());
            cut_points.insert(node.offset() + PRECOMP_RESET_LENGTH);
            node.replace_data(ir::NodeKind::ResetPrecompensationFilters(
                ir::ResetPrecompensationFilters {
                    length: PRECOMP_RESET_LENGTH,
                },
            ));
        }
        _ => {
            for child in node.iter_children_mut() {
                handle_precompensation_resets(child, cut_points)?;
            }
        }
    }
    Ok(())
}

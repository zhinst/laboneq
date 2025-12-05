// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use crate::Error;
use crate::Result;
use crate::ir;
use crate::ir::TriggerBitData;
use crate::ir::compilation_job as cjob;
use std::collections::HashSet;

/// Converts [ir::NodeKind::TriggerSet] nodes to [ir::NodeKind::SetTrigger] nodes with adjusted trigger bits,
/// taking into account the HDawg RF mode and validating against the device's
/// number of supported trigger bits.
fn handle_triggers_recursive(
    node: &mut ir::IrNode,
    hdawg_rf_mode: bool,
    number_of_trigger_bits: u8,
    cut_points: &mut HashSet<ir::Samples>,
) -> Result<()> {
    for child in node.iter_children_mut() {
        if let ir::NodeKind::TriggerSet(data) = child.data() {
            // TODO: Move validation to later stage as there are some manipulations of trigger bits after this?
            for bit in active_bits(data.bits) {
                if bit >= number_of_trigger_bits {
                    return Err(Error::new(format!(
                        "Trigger bit {} is out of range for device with {} trigger bits in section '{}'",
                        bit, number_of_trigger_bits, data.section_info.name,
                    )));
                }
            }
            let mut trigger = data.clone();
            trigger.bits = data.bits << mask_shift(hdawg_rf_mode, data)?;
            child.replace_data(ir::NodeKind::SetTrigger(trigger));
            cut_points.insert(*child.offset());
        }
        handle_triggers_recursive(child, hdawg_rf_mode, number_of_trigger_bits, cut_points)?;
    }
    Ok(())
}

// Returns the positions of active bits in the mask.
fn active_bits(mask: u8) -> impl Iterator<Item = u8> {
    (0..u8::BITS as u8).filter(move |bit| (mask & (1 << bit)) != 0)
}

fn mask_shift(hdawg_rf_mode: bool, data: &TriggerBitData) -> Result<u8> {
    let mask_shift = if hdawg_rf_mode {
        data.signal.channels.first().unwrap_or(&0) % 2
    } else {
        0
    };
    Ok(mask_shift)
}

pub(crate) fn handle_triggers(
    node: &mut ir::IrNode,
    cut_points: &mut HashSet<ir::Samples>,
    awg: &cjob::AwgCore,
) -> Result<()> {
    let signals = &awg.signals;
    let signal_kind = &signals
        .first()
        .expect("Internal error: No signals found")
        .kind;
    let hdawg_rf_mode = signal_kind == &cjob::SignalKind::SINGLE;
    let number_of_trigger_bits = if hdawg_rf_mode {
        1
    } else {
        awg.device_kind().traits().number_of_trigger_bits
    };
    handle_triggers_recursive(node, hdawg_rf_mode, number_of_trigger_bits, cut_points)
}

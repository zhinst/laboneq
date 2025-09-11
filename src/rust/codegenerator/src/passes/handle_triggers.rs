// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use crate::Result;
use crate::ir;
use crate::ir::TriggerBitData;
use crate::ir::compilation_job as cjob;
use std::collections::HashSet;
use std::sync::Arc;

// Create trigger events from sections.
//
// The timestamp of the trigger events depends on the signal delay. Therefore,
//
// The trigger value is stateful, so we need to keep track of the current state
// and must process each event in the correct order. The timestamp of the trigger
// events depend on the signal delay though, so we need to process the events
// _after_ flattening the tree.

fn handle_triggers_recursive(
    node: &mut ir::IrNode,
    hdawg_rf_mode: bool,
    number_of_trigger_bits: u8,
    cut_points: &mut HashSet<ir::Samples>,
) -> Result<()> {
    for child in node.iter_children_mut() {
        if let ir::NodeKind::TriggerSet(data) = child.data() {
            child.replace_data(ir::NodeKind::SetTrigger(TriggerBitData {
                section_info: Arc::clone(&data.section_info),
                bit: data.bit + mask_shift(hdawg_rf_mode, number_of_trigger_bits, data)?,
                set: data.set,
                signal: Arc::clone(&data.signal),
            }));
            cut_points.insert(*child.offset());
        }
        handle_triggers_recursive(child, hdawg_rf_mode, number_of_trigger_bits, cut_points)?;
    }
    Ok(())
}

fn mask_shift(
    hdawg_rf_mode: bool,
    number_of_trigger_bits: u8,
    data: &TriggerBitData,
) -> Result<u8> {
    if data.bit >= number_of_trigger_bits {
        return Err(anyhow::anyhow!(
            "Trigger bit {} is out of range for device with {} trigger bits in section '{}'",
            data.bit,
            number_of_trigger_bits,
            data.section_info.name
        )
        .into());
    }
    let mask_shift = if hdawg_rf_mode {
        data.signal.channels.first().unwrap_or(&0) % 2
    } else {
        0
    };
    Ok(mask_shift)
}

pub fn handle_triggers(
    node: &mut ir::IrNode,
    cut_points: &mut HashSet<ir::Samples>,
    awg: &cjob::AwgCore,
) -> Result<()> {
    let signals = &awg.signals;
    let signal_kind = match signals.first() {
        Some(signal) => &signal.kind,
        None => panic!("Internal error: No signals found"),
    };
    let hdawg_rf_mode = *signal_kind == cjob::SignalKind::SINGLE;
    let number_of_trigger_bits = if hdawg_rf_mode {
        1
    } else {
        awg.device_kind().traits().number_of_trigger_bits
    };
    handle_triggers_recursive(node, hdawg_rf_mode, number_of_trigger_bits, cut_points)
}

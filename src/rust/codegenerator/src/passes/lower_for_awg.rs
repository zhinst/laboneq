// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use crate::Result;
use crate::awg_delays::AwgTiming;
use crate::ir::compilation_job as cjob;
use crate::ir::{self, NodeKind};
use crate::tinysample;

/// Transformation pass to convert tiny samples to unit samples on target AWG.
pub fn convert_to_samples(node: &mut ir::IrNode, awg: &cjob::AwgCore) {
    *node.offset_mut() = tinysample::tinysample_to_samples(*node.offset(), awg.sampling_rate);
    let len_ts = node.data().length();
    node.data_mut()
        .set_length(tinysample::tinysample_to_samples(len_ts, awg.sampling_rate));
    for child in node.iter_children_mut() {
        convert_to_samples(child, awg);
    }
}

/// Transformation pass to convert relative node offsets to absolute values starting from the root.
pub fn offset_to_absolute(node: &mut ir::IrNode, offset: ir::Samples) {
    *node.offset_mut() += offset;
    let parent_offset = *node.offset();
    for child in node.iter_children_mut() {
        offset_to_absolute(child, parent_offset);
    }
}

fn adjust_sequence_end_point(
    end: ir::Samples,
    sample_multiple: u16,
    delay: ir::Samples,
    play_wave_size_hint: u16,
    play_zero_size_hint: u16,
) -> ir::Samples {
    let mut end = end + delay;
    end += play_wave_size_hint as ir::Samples + play_zero_size_hint as ir::Samples;
    end += (-end) % sample_multiple as ir::Samples;
    end
}

fn apply_delays(node: &mut ir::IrNode, delays: &AwgTiming) {
    match node.data() {
        NodeKind::PlayPulse(ob) => {
            *node.offset_mut() += delays.signal_delay(ob.signal.uid.as_str());
        }
        NodeKind::AcquirePulse(ob) => {
            *node.offset_mut() += delays.signal_delay(ob.signal.uid.as_str());
        }
        NodeKind::TriggerSet(ob) => {
            *node.offset_mut() += delays.signal_delay(ob.signal.uid.as_str());
        }
        _ => {
            *node.offset_mut() += delays.delay();
        }
    }
    for child in node.iter_children_mut() {
        apply_delays(child, delays);
    }
}

/// Transformation pass to apply delay information to the root node and its children.
///
/// This pass adjusts the offsets of the root node and its children based on the given delays.
/// It also extends the root node length to account for the AWG delay and waveform size hints.
pub fn apply_delay_information(
    node: &mut ir::IrNode,
    awg: &cjob::AwgCore,
    delays: &AwgTiming,
    play_wave_size_hint: u16,
    play_zero_size_hint: u16,
) -> Result<()> {
    for child in node.iter_children_mut() {
        apply_delays(child, delays);
    }
    // Extend the root node length to account for the global delay and waveform size hints.
    // This is necessary to ensure that the waveform can be played correctly.
    let adjusted_end = adjust_sequence_end_point(
        node.data().length(),
        awg.device_kind().traits().sample_multiple,
        delays.delay(),
        play_wave_size_hint,
        play_zero_size_hint,
    );
    let adjusted_length = adjusted_end + node.offset();
    *node.offset_mut() = node.offset() + delays.delay();
    node.data_mut().set_length(adjusted_length);
    Ok(())
}

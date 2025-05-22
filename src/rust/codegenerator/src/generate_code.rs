// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use crate::Result;
use crate::ir;
use crate::ir::compilation_job::{self as cjob};
use crate::passes::{
    amplitude_registers, handle_frame_changes, handle_match, handle_playwaves, handle_signatures,
    lower_for_awg, osc_parameters,
};
use crate::virtual_signal::create_virtual_signals;
use std::collections::HashSet;

#[allow(clippy::too_many_arguments)]
pub fn generate_code_for_awg(
    program: &ir::IrNode,
    awg: &mut cjob::AwgCore,
    cut_points: HashSet<ir::Samples>,
    play_wave_size_hint: u16,
    play_zero_size_hint: u16,
    amplitude_resolution_range: u64,
    use_amplitude_increment: bool,
    phase_resolution_range: u64,
) -> Result<ir::IrNode> {
    // NOTE: Sorting should probably happen outside of this function
    // Sort the signals for deterministic ordering
    awg.signals.sort_by(|a, b| a.channels.cmp(&b.channels));
    // Source IR children offsets are relative to parent, change them to absolute from the start (0)
    // so that they are easier to work with.
    let mut program = lower_for_awg::offset_to_absolute(program, 0);
    // Calculate oscillator parameters before applying sample conversion to avoid timing rounding errors
    let osc_params = osc_parameters::handle_oscillator_parameters(
        &mut program,
        &awg.signals,
        &awg.device_kind,
        &awg.sampling_rate,
    )?;
    lower_for_awg::convert_to_samples(&mut program, awg);
    let Some(virtual_signals) = create_virtual_signals(awg)? else {
        return Ok(program); // No signals that play anything, return
    };
    handle_frame_changes::handle_frame_changes(&mut program, virtual_signals.delay());
    handle_match::handle_match_nodes(&mut program, virtual_signals.delay())?;
    let amp_reg_alloc = amplitude_registers::assign_amplitude_registers(&program, awg);
    amplitude_registers::handle_amplitude_register_events(
        &mut program,
        &amp_reg_alloc,
        &awg.device_kind,
        *virtual_signals.delay(),
    );
    handle_playwaves::handle_plays(
        &mut program,
        awg,
        &virtual_signals,
        cut_points,
        play_wave_size_hint,
        play_zero_size_hint,
        &osc_params,
        &amp_reg_alloc,
    )?;
    handle_signatures::optimize_signatures(
        &mut program,
        awg,
        use_amplitude_increment,
        amp_reg_alloc.available_register_count(),
        amplitude_resolution_range,
        phase_resolution_range,
    );
    Ok(program)
}

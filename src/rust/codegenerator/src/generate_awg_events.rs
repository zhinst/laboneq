// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use crate::Result;
use crate::ir;
use crate::ir::compilation_job::{self as cjob};
use crate::passes::{
    amplitude_registers, handle_acquire, handle_frame_changes, handle_match, handle_playwaves,
    handle_ppc_sweeps, handle_precompensation_resets, handle_signatures, lower_for_awg,
    osc_parameters,
};
use crate::virtual_signal::create_virtual_signals;
use std::collections::HashSet;

/// Transform the IR program into a AWG events for the target AWG.
///
/// This function processes the IR program, applying various transformations and optimizations
/// to generate a set of AWG events that can be executed on the device.
#[allow(clippy::too_many_arguments)]
pub fn transform_ir_to_awg_events(
    program: ir::IrNode,
    awg: &cjob::AwgCore,
    cut_points: HashSet<ir::Samples>,
    play_wave_size_hint: u16,
    play_zero_size_hint: u16,
    amplitude_resolution_range: u64,
    use_amplitude_increment: bool,
    phase_resolution_range: u64,
    global_delay_samples: ir::Samples,
) -> Result<ir::IrNode> {
    let mut program = program;
    let mut cut_points = cut_points;
    // Source IR children offsets are relative to parent, change them to absolute from the start (0)
    // so that they are easier to work with.
    lower_for_awg::offset_to_absolute(&mut program, 0);
    // Calculate oscillator parameters before applying sample conversion to avoid timing rounding errors
    let osc_params = osc_parameters::handle_oscillator_parameters(
        &mut program,
        &awg.signals,
        &awg.device_kind,
        &awg.sampling_rate,
    )?;
    lower_for_awg::convert_to_samples(&mut program, awg);
    handle_acquire::handle_acquisitions(
        &mut program,
        awg.device_kind.traits().sample_multiple,
        &osc_params,
    )?;
    let Some(virtual_signals) = create_virtual_signals(awg)? else {
        return Ok(program); // No signals that play anything, return
    };
    handle_frame_changes::handle_frame_changes(&mut program, virtual_signals.delay());
    handle_precompensation_resets::handle_precompensation_resets(
        &mut program,
        global_delay_samples,
        &mut cut_points,
    )?;
    if let cjob::DeviceKind::SHFQA = &awg.device_kind {
        handle_ppc_sweeps::handle_ppc_sweep_steps(&mut program, global_delay_samples)?;
    }
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

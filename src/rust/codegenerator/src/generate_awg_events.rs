// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use crate::Result;
use crate::awg_delays::calculate_awg_delays;
use crate::ir;
use crate::ir::compilation_job::{AwgCore, DeviceKind};
use crate::passes::{
    handle_acquire, handle_amplitude_registers, handle_frame_changes, handle_hw_phase_resets,
    handle_loops, handle_match, handle_oscillators, handle_playwaves, handle_ppc_sweeps,
    handle_precompensation_resets, handle_prng, handle_qa_events, handle_signatures,
    handle_triggers, lower_for_awg,
};
use crate::virtual_signal::create_virtual_signals;
use laboneq_units::tinysample::tinysamples_to_samples;
use std::collections::{HashMap, HashSet};

/// Transform the IR program into a AWG events for the target AWG.
///
/// This function processes the IR program, applying various transformations and optimizations
/// to generate a set of AWG events that can be executed on the device.
#[allow(clippy::too_many_arguments)]
pub fn transform_ir_to_awg_events(
    program: ir::IrNode,
    awg: &AwgCore,
    settings: &crate::CodeGeneratorSettings,
    signal_delays: &HashMap<&str, ir::Samples>,
) -> Result<ir::IrNode> {
    let awg_timing: crate::awg_delays::AwgTiming = calculate_awg_delays(awg, signal_delays)?;
    let mut program = program;
    let mut cut_points = HashSet::from_iter([awg_timing.delay()]); // Sequencer start
    // Source IR children offsets are relative to parent, change them to absolute from the start (0)
    // so that they are easier to work with.
    lower_for_awg::offset_to_absolute(&mut program, 0);
    // Calculate oscillator parameters before applying sample conversion to avoid timing rounding errors
    let osc_params = handle_oscillators::handle_oscillator_parameters(
        &mut program,
        &awg.signals,
        awg.device_kind(),
        |signal_uid, ts| {
            tinysamples_to_samples(ts.into(), awg.sampling_rate)
                + awg_timing.signal_delay(signal_uid)
        },
    )?;
    lower_for_awg::convert_to_samples(&mut program, awg);
    let (play_wave_size_hint, play_zero_size_hint) =
        settings.waveform_size_hints(awg.device_kind());
    lower_for_awg::apply_delay_information(
        &mut program,
        awg,
        &awg_timing,
        play_wave_size_hint,
        play_zero_size_hint,
    )?;
    handle_oscillators::handle_oscillator_sweeps(&mut program, awg, &mut cut_points)?;
    handle_acquire::handle_acquisitions(
        &mut program,
        awg.device_kind().traits().sample_multiple,
        &osc_params,
    )?;
    handle_hw_phase_resets::handle_hw_phase_resets(&mut program, &mut cut_points)?;
    handle_precompensation_resets::handle_precompensation_resets(&mut program, &mut cut_points)?;
    if let DeviceKind::SHFQA = awg.device_kind() {
        handle_ppc_sweeps::handle_ppc_sweep_steps(&mut program)?;
    }
    handle_loops::handle_loops(&program, &mut cut_points)?;
    handle_triggers::handle_triggers(&mut program, &mut cut_points, awg)?;
    handle_prng::handle_prng(&program, &mut cut_points)?;
    if let Some(virtual_signals) = create_virtual_signals(awg)? {
        handle_frame_changes::handle_frame_changes(&mut program);
        handle_match::handle_match_nodes(&program)?;
        let amp_reg_alloc = handle_amplitude_registers::assign_amplitude_registers(&program, awg);
        handle_amplitude_registers::handle_amplitude_register_events(
            &mut program,
            &amp_reg_alloc,
            awg.device_kind(),
        );
        let (play_wave_size_hint, play_zero_size_hint) =
            settings.waveform_size_hints(awg.device_kind());
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
            settings.use_amplitude_increment(),
            amp_reg_alloc.available_register_count(),
            settings.amplitude_resolution_range(),
            settings.phase_resolution_range(),
        );
    }
    handle_qa_events(&mut program, awg.device_kind())?;
    Ok(program)
}

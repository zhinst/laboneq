// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::iter;

use crate::ir::compilation_job::{AwgCore, OscillatorKind, PulseDefKind};
use crate::ir::{InitAmplitudeRegister, IrNode, NodeKind, ParameterOperation};
use crate::signature::{PulseSignature, quantize_amplitude_ct, quantize_amplitude_pulse};
use crate::signature::{quantize_phase_ct, quantize_phase_pulse};

fn reduce_signature_phase(
    pulses: &mut [PulseSignature],
    use_ct_phase: bool,
) -> (Option<f64>, Vec<Option<String>>) {
    let total_phase_increment: f64 = pulses
        .iter()
        .filter_map(|pulse| pulse.increment_oscillator_phase)
        .sum();
    let has_phase_increment_param = pulses
        .iter()
        .any(|pulse| !pulse.incr_phase_params.is_empty());
    if total_phase_increment != 0.0 || has_phase_increment_param {
        assert!(
            use_ct_phase,
            "cannot increment oscillator phase w/o command table"
        );
    }

    let mut signature_incr_phase_params: Vec<Option<String>> = vec![];
    let mut running_increment = 0.0;
    for pulse in pulses.iter_mut() {
        // Absorb pulse oscillator phase increment into signature
        running_increment += pulse.increment_oscillator_phase.take().unwrap_or(0.0);
        if running_increment - total_phase_increment != 0.0 {
            pulse.oscillator_phase = Some(
                pulse.oscillator_phase.unwrap_or(0.0) + running_increment - total_phase_increment,
            );
            // Tell the compiler that there was a phase increment, but not from
            // a sweep parameter.
            // TODO: Could this logic be simplified to avoid magic `None`s?
            signature_incr_phase_params.push(None);
        }
        // Absorb oscillator phase into pulse phase
        pulse.phase += pulse.oscillator_phase.take().unwrap_or(0.0);
        // Absorb pulse oscillator phase increment parameters into signature
        signature_incr_phase_params.extend(pulse.incr_phase_params.drain(..).map(Some));
    }
    if total_phase_increment != 0.0 || has_phase_increment_param {
        return (Some(total_phase_increment), signature_incr_phase_params);
    }
    (None, signature_incr_phase_params)
}

fn handle_signature_phases(
    pulses: &mut [PulseSignature],
    use_ct_phase: bool,
    phase_resolution_range: u64,
) -> (Option<f64>, Vec<Option<String>>) {
    let (mut increment_phase, increment_phase_parameters) =
        reduce_signature_phase(pulses, use_ct_phase);
    // Quantize phases
    if phase_resolution_range >= 1 {
        for pulse in pulses.iter_mut() {
            pulse.phase = quantize_phase_pulse(pulse.phase, phase_resolution_range);
        }
        if let Some(phase) = increment_phase.as_mut() {
            *phase = quantize_phase_ct(*phase);
        }
    }
    (increment_phase, increment_phase_parameters)
}

/// Evaluate whether the phase increments should be used via command table.
fn evaluate_use_ct_phase(awg: &AwgCore) -> bool {
    if awg.use_command_table_phase_amp() {
        let mut hw_oscs = awg.signals.iter().map(|signal| {
            signal
                .oscillator
                .as_ref()
                .is_some_and(|osc| osc.kind == OscillatorKind::HARDWARE)
        });
        hw_oscs.all(|is_hw_osc| is_hw_osc)
    } else {
        false
    }
}

/// Determines the amplitude register of the pulses.
///
/// Aggregate the preferred amplitude register of the individual pulses.
/// For two or more pulses that prefer different registers, the
/// result will fall back to register 0.
///
/// Consumes the value of [`PulseSignature::preferred_amplitude_register`], setting it to `None`.
///
/// # Returns
///
/// Playback amplitude register of the pulses
fn determine_amplitude_register(pulses: &mut Vec<PulseSignature>) -> u16 {
    let mut registers: Vec<Option<u16>> = Vec::new();
    for pulse in pulses {
        // Clear per pulse register
        let amp_reg = pulse.preferred_amplitude_register.take();
        if pulse
            .pulse
            .as_ref()
            .is_none_or(|p_def| p_def.kind != PulseDefKind::Marker)
            && !registers.contains(&amp_reg)
        {
            registers.push(amp_reg);
        }
    }
    if registers.len() == 1 {
        return registers[0].unwrap_or(0);
    }
    0
}

/// Aggregates pulse amplitudes for command table.
///
/// Whenever possible, the waveforms will be sampled at unit
/// amplitude, making waveform reuse more likely.
///
/// # Returns
///
/// Calculated command table amplitude if:
///
/// * Each pulse has an amplitude
/// * Any of the pulses has a pulse definition that is not of type marker
fn aggregate_ct_amplitude(pulses: &mut [PulseSignature]) -> Option<f64> {
    if pulses.iter().any(|pulse| pulse.amplitude.is_none()) || pulses.is_empty() {
        return None;
    }
    // Filter relevant pulses
    let pulses_filtered: Vec<_> = pulses
        .iter_mut()
        .filter(|pulse| {
            pulse
                .pulse
                .as_ref()
                .is_some_and(|p_def| p_def.kind != PulseDefKind::Marker)
        })
        .collect();
    // Find the maximum amplitude among pulses
    let ct_amplitude = pulses_filtered
        .iter()
        .map(|pulse| {
            pulse
                .amplitude
                .expect("Expected pulse to have an amplitude")
                .abs()
        })
        .collect::<Vec<f64>>()
        .into_iter()
        .reduce(f64::max)
        .unwrap_or(1.0);
    let ct_amplitude = ct_amplitude.min(1.0);
    if ct_amplitude != 0.0 {
        for pulse in pulses_filtered {
            if let Some(amp) = &mut pulse.amplitude {
                *amp /= ct_amplitude;
            }
        }
    }
    Some(ct_amplitude)
}

struct AmplitudeRegisterValues {
    available_registers: Vec<Option<f64>>,
}

impl AmplitudeRegisterValues {
    fn new(register_count: u16) -> Self {
        // Mark each available register as uninitialized at start
        let available_registers: Vec<Option<f64>> =
            iter::repeat_n(None, register_count.into()).collect();
        AmplitudeRegisterValues {
            available_registers,
        }
    }

    fn reset(&mut self) {
        self.available_registers.iter_mut().for_each(|x| {
            *x = None;
        });
    }

    fn set(&mut self, register: u16, value: f64) {
        self.available_registers[register as usize] = Some(value);
    }

    fn get(&self, register: u16) -> Option<f64> {
        self.available_registers[register as usize]
    }

    fn increment(&mut self, register: u16, value: f64) {
        if let Some(amp) = &mut self.available_registers[register as usize] {
            *amp += value;
        }
    }
}

/// Try to make command table amplitude incremental
///
/// If the current value of the amplitude register is known, the new amplitude is
/// expressed as an _increment_ of the old value. This way, amplitude sweeps can be
/// modelled with very few command table entries.
///
/// We attempt to make the amplitude relative to the previous value.
/// The idea is that if there is a linear sweep, the increment is constant
/// especially if the register is reserved for a sweep parameter). The same command
/// table entry can then be reused for every step of the sweep.
fn try_make_ct_amplitude_incremental(
    amp_op: &mut ParameterOperation<f64>,
    amp_register: u16,
    amplitude_register_values: &AmplitudeRegisterValues,
) {
    if amp_register < 1 {
        return;
    }
    if let Some(previous_amplitude) = amplitude_register_values.get(amp_register) {
        *amp_op = ParameterOperation::INCREMENT(amp_op.value() - previous_amplitude);
    }
}

/// Evaluate signature amplitude based on its pulses.
///
/// # Returns
///
/// Determined amplitude register and the potential command table amplitude.
fn evaluate_signature_amplitude(
    pulses: &mut Vec<PulseSignature>,
    use_command_table: bool,
    use_amplitude_increment: bool,
    amplitude_resolution_range: u64,
    amplitude_register_values: &mut AmplitudeRegisterValues,
) -> (u16, Option<ParameterOperation<f64>>) {
    let amplitude_register = determine_amplitude_register(pulses);
    let mut ct_amplitude = use_command_table.then(|| {
        let mut ct_amp = ParameterOperation::SET(aggregate_ct_amplitude(pulses).unwrap_or(1.0));
        if use_amplitude_increment {
            try_make_ct_amplitude_incremental(
                &mut ct_amp,
                amplitude_register,
                amplitude_register_values,
            );
        }
        ct_amp
    });
    // Quantize amplitudes
    if amplitude_resolution_range >= 1 {
        if let Some(amp) = ct_amplitude.as_mut() {
            *amp.value_mut() = quantize_amplitude_ct(amp.value());
        }
        for pulse in pulses {
            if let Some(amp) = &mut pulse.amplitude {
                *amp = quantize_amplitude_pulse(*amp, amplitude_resolution_range);
            }
        }
    }
    // Update amplitude register with the current amplitude
    if let Some(amp_op) = &ct_amplitude {
        match amp_op {
            ParameterOperation::SET(_) => {
                amplitude_register_values.set(amplitude_register, amp_op.value());
            }
            ParameterOperation::INCREMENT(_) => {
                assert_ne!(
                    amplitude_register, 0,
                    "Cannot use amplitude increment on register 0"
                );
                amplitude_register_values.increment(amplitude_register, amp_op.value());
            }
        }
    }
    (amplitude_register, ct_amplitude)
}

fn handle_init_amplitude_register(
    amp_reg: &mut InitAmplitudeRegister,
    use_command_table: bool,
    amplitude_resolution_range: u64,
    amplitude_register_values: &mut AmplitudeRegisterValues,
) {
    if use_command_table {
        try_make_ct_amplitude_incremental(
            &mut amp_reg.value,
            amp_reg.register,
            amplitude_register_values,
        );
    }
    if amplitude_resolution_range >= 1 {
        *amp_reg.value.value_mut() = quantize_amplitude_ct(amp_reg.value.value());
    }
    match amp_reg.value {
        ParameterOperation::SET(value) => {
            amplitude_register_values.set(amp_reg.register, value);
        }
        ParameterOperation::INCREMENT(value) => {
            amplitude_register_values.increment(amp_reg.register, value);
        }
    }
}

struct PassContext<'a> {
    use_command_table: bool,
    use_amplitude_increment: bool,
    amplitude_resolution_range: u64,
    amplitude_register_values: &'a mut AmplitudeRegisterValues,
    phase_resolution_range: u64,
    use_ct_phase: bool,
    in_branch: bool,
}

fn transform_waveforms(node: &mut IrNode, ctx: &mut PassContext) {
    // For branches, we _must_ emit the same signature every time, and cannot depend
    // what came before. So using the increment is not valid.
    let use_amplitude_increment = ctx.use_amplitude_increment && !ctx.in_branch;
    match node.data_mut() {
        NodeKind::PlayWave(ob) => {
            let pulses = ob
                .waveform
                .pulses_mut()
                .expect("Internal error: Waveform pulses must be present");
            let (amplitude_register, ct_amp) = evaluate_signature_amplitude(
                pulses,
                ctx.use_command_table,
                use_amplitude_increment,
                ctx.amplitude_resolution_range,
                ctx.amplitude_register_values,
            );
            ob.amplitude_register = amplitude_register;
            ob.amplitude = ct_amp;

            let (increment_phase, params) =
                handle_signature_phases(pulses, ctx.use_ct_phase, ctx.phase_resolution_range);
            ob.increment_phase = increment_phase;
            ob.increment_phase_params = params;
            // After a conditional playback, we do not know for sure what value
            // was written to the amplitude register. Mark it as such, to avoid
            // emitting an increment for the next signature.
            // Todo: There are ways to improve on this:
            //   - Do not write the amplitude register in branches
            //   - Only clear those registers that were indeed written in a branch
            //   - ...
            if ctx.in_branch {
                ctx.amplitude_register_values.reset();
            }
        }
        NodeKind::InitAmplitudeRegister(ob) => {
            handle_init_amplitude_register(
                ob,
                ctx.use_command_table,
                ctx.amplitude_resolution_range,
                ctx.amplitude_register_values,
            );
        }
        NodeKind::Case(_) => {
            ctx.in_branch = true;
            for child in node.iter_children_mut() {
                transform_waveforms(child, ctx);
            }
            ctx.in_branch = false;
        }
        _ => {
            for child in node.iter_children_mut() {
                transform_waveforms(child, ctx);
            }
        }
    }
}

/// Transformation pass to optimize signatures.
///
/// * Determines amplitude register for each waveform
/// * Splits complex amplitudes into a real amplitude and a phase
/// * Absorbs the pulse amplitude into the command table. Whenever possible, the
///   waveforms will be sampled at unit amplitude, making waveform reuse more likely
pub fn optimize_signatures(
    node: &mut IrNode,
    awg: &AwgCore,
    use_amplitude_increment: bool,
    amplitude_register_count: u16,
    amplitude_resolution_range: u64,
    phase_resolution_range: u64,
) {
    let mut amplitude_register_values = AmplitudeRegisterValues::new(amplitude_register_count);
    let mut ctx = PassContext {
        use_command_table: awg.use_command_table_phase_amp(),
        use_amplitude_increment: awg.use_amplitude_increment() && use_amplitude_increment,
        amplitude_resolution_range,
        amplitude_register_values: &mut amplitude_register_values,
        phase_resolution_range,
        use_ct_phase: evaluate_use_ct_phase(awg),
        in_branch: false,
    };
    transform_waveforms(node, &mut ctx);
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::ir::compilation_job::{PulseDef, PulseDefKind};
    use crate::signature::PulseSignature;

    fn make_amp_reg_pulse_signature(
        preferred_amplitude_register: Option<u16>,
        pulse_def_kind: PulseDefKind,
    ) -> PulseSignature {
        let pulse = PulseDef {
            uid: "".to_string(),
            kind: pulse_def_kind,
        };
        PulseSignature {
            start: 0,
            pulse: Some(Arc::new(pulse)),
            length: 0,
            phase: 0.0,
            amplitude: None,
            oscillator_frequency: None,
            incr_phase_params: vec![],
            oscillator_phase: None,
            increment_oscillator_phase: None,
            channel: None,
            sub_channel: None,
            id_pulse_params: None,
            markers: vec![],
            preferred_amplitude_register,
        }
    }

    #[test]
    fn test_determine_amplitude_register() {
        let reg = determine_amplitude_register(&mut vec![make_amp_reg_pulse_signature(
            None,
            PulseDefKind::Pulse,
        )]);
        assert_eq!(reg, 0);

        let reg = determine_amplitude_register(&mut vec![
            make_amp_reg_pulse_signature(None, PulseDefKind::Pulse),
            make_amp_reg_pulse_signature(None, PulseDefKind::Pulse),
        ]);
        assert_eq!(reg, 0);

        let reg = determine_amplitude_register(&mut vec![
            make_amp_reg_pulse_signature(Some(1), PulseDefKind::Pulse),
            make_amp_reg_pulse_signature(None, PulseDefKind::Pulse),
        ]);
        assert_eq!(reg, 0);

        let reg = determine_amplitude_register(&mut vec![
            make_amp_reg_pulse_signature(Some(1), PulseDefKind::Pulse),
            make_amp_reg_pulse_signature(Some(2), PulseDefKind::Marker),
        ]);
        assert_eq!(reg, 1);

        let reg = determine_amplitude_register(&mut vec![make_amp_reg_pulse_signature(
            Some(1),
            PulseDefKind::Marker,
        )]);
        assert_eq!(reg, 0);
    }

    mod test_aggregate_ct_amplitude {
        use super::*;
        use crate::ir::compilation_job::{PulseDef, PulseDefKind};
        use crate::signature;
        use std::sync::Arc;

        fn make_signature(
            amplitude: Option<f64>,
            pulse_def_kind: PulseDefKind,
        ) -> signature::PulseSignature {
            let pulse = PulseDef {
                uid: "".to_string(),
                kind: pulse_def_kind,
            };
            signature::PulseSignature {
                start: 0,
                pulse: Some(Arc::new(pulse)),
                length: 0,
                phase: 0.0,
                amplitude,
                oscillator_frequency: None,
                incr_phase_params: vec![],
                oscillator_phase: None,
                increment_oscillator_phase: None,
                channel: None,
                sub_channel: None,
                id_pulse_params: None,
                markers: vec![],
                preferred_amplitude_register: None,
            }
        }

        #[test]
        fn test_no_pulses() {
            assert!(aggregate_ct_amplitude(&mut []).is_none());
        }

        /// Test largest amplitude pulse into command table
        #[test]
        fn test_multiple_pulses_ct_amplitude_selection() {
            let mut pulses = [
                make_signature(Some(0.5), PulseDefKind::Pulse),
                make_signature(Some(0.7), PulseDefKind::Pulse),
            ];
            let ct_amp = aggregate_ct_amplitude(&mut pulses).unwrap();
            assert_eq!(ct_amp, 0.7);
            assert_eq!(pulses[0].amplitude.unwrap(), 0.5 / 0.7);
            assert_eq!(pulses[1].amplitude.unwrap(), 1.0);
        }

        /// Test any of the pulses have no amplitude definition => no command table amplitude
        #[test]
        fn test_no_ct_amplitude() {
            let mut pulses = [
                make_signature(None, PulseDefKind::Pulse),
                make_signature(Some(0.7), PulseDefKind::Pulse),
            ];
            let ct_amp = aggregate_ct_amplitude(&mut pulses);
            assert!(ct_amp.is_none());
        }

        /// Test pulse + marker pulse composite.
        #[test]
        fn test_pulse_with_marker() {
            let mut pulses = [
                make_signature(Some(0.5), PulseDefKind::Pulse),
                make_signature(Some(0.1), PulseDefKind::Marker),
            ];
            let ct_amp = aggregate_ct_amplitude(&mut pulses);
            assert_eq!(ct_amp.unwrap(), 0.5);
        }

        // Test that marker-only pulse signatures should be skipped when computing the CT-amplitude.
        // If the signature contains nothing but marker-only pulses, the CT amplitude should
        // be 1.0.
        #[test]
        fn test_marker_pulse_ignored() {
            let mut pulses = [make_signature(Some(0.01), PulseDefKind::Marker)];
            let ct_amp = aggregate_ct_amplitude(&mut pulses).unwrap();
            assert_eq!(ct_amp, 1.0);
        }
    }

    mod test_evaluate_signature_amplitude {
        use super::*;
        use crate::ir::compilation_job::{PulseDef, PulseDefKind};
        use crate::signature;
        use std::sync::Arc;

        fn make_signature(
            amplitude: f64,
            preferred_amplitude_register: Option<u16>,
        ) -> signature::PulseSignature {
            let pulse = PulseDef {
                uid: "".to_string(),
                kind: PulseDefKind::Pulse,
            };
            signature::PulseSignature {
                start: 0,
                pulse: Some(Arc::new(pulse)),
                length: 0,
                phase: 0.0,
                amplitude: Some(amplitude),
                oscillator_frequency: None,
                incr_phase_params: vec![],
                oscillator_phase: None,
                increment_oscillator_phase: None,
                channel: None,
                sub_channel: None,
                id_pulse_params: None,
                markers: vec![],
                preferred_amplitude_register,
            }
        }

        /// Test no amplitude increments on register 0
        #[test]
        fn test_amplitude_reduction_increment_reg0() {
            let mut amplitude_register_values = AmplitudeRegisterValues::new(1);
            let (reg, amp) = evaluate_signature_amplitude(
                &mut vec![make_signature(0.5, None)],
                true,
                true,
                1 << 18,
                &mut amplitude_register_values,
            );
            assert_eq!(reg, 0);
            assert_eq!(amp, Some(ParameterOperation::SET(0.5)));

            let (reg, amp) = evaluate_signature_amplitude(
                &mut vec![make_signature(0.7, None)],
                true,
                true,
                1 << 18,
                &mut amplitude_register_values,
            );
            assert_eq!(reg, 0);
            assert_eq!(
                amp,
                Some(ParameterOperation::SET(quantize_amplitude_ct(0.7)))
            )
        }

        /// Test zero amplitude increment when amplitude does not change on register != 0
        #[test]
        fn test_amplitude_increment_zero_reg1() {
            let mut amplitude_registers = AmplitudeRegisterValues::new(4);
            let (amp_reg, ct_amp) = evaluate_signature_amplitude(
                &mut vec![make_signature(0.5, Some(1))],
                true,
                true,
                1 << 18,
                &mut amplitude_registers,
            );
            assert_eq!(amp_reg, 1);
            assert_eq!(ct_amp.unwrap(), ParameterOperation::SET(0.5));

            let (amp_reg, ct_amp) = evaluate_signature_amplitude(
                &mut vec![make_signature(0.5, Some(1))],
                true,
                true,
                1 << 18,
                &mut amplitude_registers,
            );
            assert_eq!(amp_reg, 1);
            assert_eq!(ct_amp.unwrap(), ParameterOperation::INCREMENT(0.0));
        }

        /// Test positive amplitude increment happens on register != 0
        #[test]
        fn test_amplitude_positive_increment_reg1() {
            let amp0 = 0.5;
            let amp1 = 0.6;
            let mut amplitude_registers = AmplitudeRegisterValues::new(4);
            let (amp_reg, ct_amp) = evaluate_signature_amplitude(
                &mut vec![make_signature(amp0, Some(1))],
                true,
                true,
                18,
                &mut amplitude_registers,
            );
            assert_eq!(amp_reg, 1);
            assert_eq!(ct_amp.unwrap(), ParameterOperation::SET(amp0));

            let (amp_reg, ct_amp) = evaluate_signature_amplitude(
                &mut vec![make_signature(amp1, Some(1))],
                true,
                true,
                1 << 18,
                &mut amplitude_registers,
            );
            assert_eq!(amp_reg, 1);
            assert_eq!(
                ct_amp.unwrap(),
                ParameterOperation::INCREMENT(quantize_amplitude_ct(amp1 - amp0))
            );
        }

        /// Test negative amplitude increment happens on register != 0
        #[test]
        fn test_amplitude_negative_increment_reg1() {
            let amp0 = 0.5;
            let amp1 = 0.4;
            let mut amplitude_registers = AmplitudeRegisterValues::new(4);
            let (amp_reg, ct_amp) = evaluate_signature_amplitude(
                &mut vec![make_signature(amp0, Some(1))],
                true,
                true,
                1 << 18,
                &mut amplitude_registers,
            );
            assert_eq!(amp_reg, 1);
            assert_eq!(ct_amp.unwrap(), ParameterOperation::SET(amp0));

            let (amp_reg, ct_amp) = evaluate_signature_amplitude(
                &mut vec![make_signature(amp1, Some(1))],
                true,
                true,
                1 << 18,
                &mut amplitude_registers,
            );
            assert_eq!(amp_reg, 1);
            assert_eq!(
                ct_amp.unwrap(),
                ParameterOperation::INCREMENT(quantize_amplitude_ct(amp1 - amp0))
            );
        }

        /// Test do not use amplitude increments
        #[test]
        fn test_no_amplitude_increment() {
            let amp0 = 0.5;
            let mut amplitude_registers = AmplitudeRegisterValues::new(4);
            // Fill amplitude register with some existing value
            amplitude_registers.set(1, 0.2);
            let (_, ct_amp) = evaluate_signature_amplitude(
                &mut vec![make_signature(amp0, Some(1))],
                true,
                false,
                1 << 18,
                &mut amplitude_registers,
            );
            assert_eq!(ct_amp.unwrap(), ParameterOperation::SET(amp0));
            assert_eq!(amplitude_registers.get(1), Some(amp0));
        }

        /// Test no command table amplitude
        #[test]
        fn test_no_command_table_amplitude() {
            let mut amplitude_registers = AmplitudeRegisterValues::new(4);
            let (_, ct_amp) = evaluate_signature_amplitude(
                &mut vec![make_signature(0.5, Some(1))],
                false,
                true,
                1 << 18,
                &mut amplitude_registers,
            );
            assert_eq!(ct_amp, None);
        }
    }

    mod test_handle_signature_phases {

        use super::*;
        use crate::signature::PulseSignature;

        fn make_signature(
            phase: f64,
            oscillator_phase: Option<f64>,
            increment_oscillator_phase: Option<f64>,
            incr_phase_params: Vec<String>,
        ) -> PulseSignature {
            PulseSignature {
                start: 0,
                pulse: None,
                length: 0,
                phase,
                amplitude: None,
                oscillator_frequency: None,
                incr_phase_params,
                oscillator_phase,
                increment_oscillator_phase,
                channel: None,
                sub_channel: None,
                id_pulse_params: None,
                markers: vec![],
                preferred_amplitude_register: None,
            }
        }

        #[test]
        fn test_reduce_phase_double_increment_no_quantization() {
            let amplitude_resolution_range = 0;
            let mut pulses = [
                make_signature(2.0, Some(0.0), Some(4.0), vec![]),
                make_signature(0.0, Some(3.0), Some(3.0), vec![]),
            ];
            let (increment_phase, increment_phase_params) =
                handle_signature_phases(&mut pulses, true, amplitude_resolution_range);
            assert_eq!(increment_phase, Some(7.0));
            assert_eq!(pulses[0].phase, 2.0 - 3.0);
            assert_eq!(pulses[1].phase, 3.0 - 0.0);
            assert_eq!(increment_phase_params, vec![None]);
        }

        #[test]
        fn test_reduce_phase_single_increment_no_quantization() {
            let amplitude_resolution_range = 0;
            let mut pulses = [
                make_signature(2.0, Some(0.0), None, vec![]),
                make_signature(0.0, Some(3.0), Some(3.0), vec![]),
            ];
            let (increment_phase, increment_phase_params) =
                handle_signature_phases(&mut pulses, true, amplitude_resolution_range);
            assert_eq!(increment_phase, Some(3.0));
            assert_eq!(pulses[0].phase, 2.0 - 3.0);
            assert_eq!(pulses[1].phase, 3.0 - 0.0);
            assert_eq!(increment_phase_params, vec![None]);
        }

        /// Test that phase increment parameters are absorbed into the signature.
        #[test]
        fn test_reduce_phase_single_increment_with_sweep_parameter_no_quantization() {
            let amplitude_resolution_range = 0;
            let mut pulses = [
                make_signature(2.0, Some(0.0), None, vec!["param".to_string()]),
                make_signature(0.0, Some(3.0), Some(3.0), vec![]),
            ];
            let (increment_phase, increment_phase_params) =
                handle_signature_phases(&mut pulses, true, amplitude_resolution_range);
            assert_eq!(increment_phase, Some(3.0));
            assert_eq!(pulses[0].phase, 2.0 - 3.0);
            assert!(pulses[0].incr_phase_params.is_empty());
            assert_eq!(pulses[1].phase, 3.0 - 0.0);
            // Magic `None` when increment happens on second pulse
            assert_eq!(
                increment_phase_params,
                vec![None, Some("param".to_string())]
            );
        }

        #[test]
        fn test_reduce_phase_with_quantization() {
            let amplitude_resolution_range = 1 << 24;
            let mut pulses = [
                make_signature(2.0, Some(0.0), Some(4.0), vec![]),
                make_signature(0.0, Some(3.0), Some(3.0), vec![]),
            ];
            let (increment_phase, increment_phase_params) =
                handle_signature_phases(&mut pulses, true, amplitude_resolution_range);
            assert_eq!(increment_phase, Some(quantize_phase_ct(7.0)));
            assert_eq!(
                pulses[0].phase,
                quantize_phase_pulse(3.0 - 4.0, amplitude_resolution_range)
            );
            assert_eq!(
                pulses[1].phase,
                quantize_phase_pulse(3.0 - 0.0, amplitude_resolution_range)
            );
            assert_eq!(increment_phase_params, vec![None]);
        }
    }
}

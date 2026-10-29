// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use laboneq_common::types::SignalKind;
use laboneq_dsl::ExperimentNode;
use laboneq_dsl::operation::{Acquire, AveragingLoop, Operation, PlayPulse};
use laboneq_dsl::types::{
    AcquisitionType, AveragingMode, ComplexOrFloat, ParameterUid, PulseKind, SectionTimingMode,
    ValueOrParameter,
};
use numeric_array::NumericArray;

use crate::error::{Error, Result};

use crate::experiment_validation::traversal::{Validator, VisitContext};
use crate::experiment_validation::{validate_parameters::*, validate_pulses::*};
use crate::signal_view::SignalView;

/// Validates the operations of an experiment.
pub(super) struct OperationValidator {
    amplitude_check_done: Vec<ParameterUid>,
    declared_sweep_parameters: Vec<ParameterUid>,
    found_chunked_sweep: bool,
}

impl OperationValidator {
    pub(super) fn new(ctx: &VisitContext) -> Self {
        OperationValidator {
            amplitude_check_done: Vec::with_capacity(ctx.pulses.len()),
            declared_sweep_parameters: ctx.parameters.keys().copied().collect(),
            found_chunked_sweep: false,
        }
    }
}

impl<'a> Validator<'a> for OperationValidator {
    fn start(&mut self, ctx: &VisitContext<'a>) -> Result<()> {
        check_signals_supported(ctx)
    }

    fn visit(&mut self, node: &ExperimentNode, ctx: &VisitContext<'a>) -> Result<()> {
        validate_strict_timing_only_in_rt(node, ctx)?;
        match &node.kind {
            Operation::PlayPulse(op) => {
                let signal = ctx
                    .signals
                    .get(&op.signal)
                    .ok_or_else(|| Error::new("Signal not found."))?;
                validate_play_pulse(op, ctx, &mut self.amplitude_check_done)?;
                check_no_play_on_acquire_line(signal, ctx)?;
                check_arbitrary_marker_is_valid(op, signal, ctx)?;
                check_phase_on_rf_signal_support(op, signal, ctx)?;
            }
            Operation::Acquire(op) => {
                check_acquire_only_on_acquire_line(op, ctx)?;
            }
            Operation::AveragingLoop(op) => {
                validate_averaging_loop(op)?;
            }
            Operation::Sweep(op) => {
                validate_parallel_sweep(op, ctx)?;
                digest_sweep_parameters(op, &mut self.declared_sweep_parameters);
                validate_chunked_sweep(op, ctx, &mut self.found_chunked_sweep)?;
                validate_no_nt_parameter_in_rt_sweep(op, ctx)?;
            }
            Operation::RealTimeBoundary => {
                assert_eq!(
                    node.children.len(),
                    1,
                    "RealTimeBoundary node must have exactly one child."
                );
            }
            _ => {}
        };
        Ok(())
    }

    fn leave(&mut self, node: &ExperimentNode, ctx: &VisitContext<'a>) -> Result<()> {
        validate_no_mixed_operations(node, ctx)
    }

    fn finish(&mut self, _ctx: &VisitContext<'a>) -> Result<()> {
        validate_sweep_parameters(&mut self.declared_sweep_parameters)
    }
}

/// Refuse a signal that asks for something its device cannot do.
fn check_signals_supported(ctx: &VisitContext) -> Result<()> {
    for signal in ctx.signals.values() {
        if signal.automute() && !signal.device_traits().supports_output_mute {
            return Err(Error::new(format!(
                "Automute is not available on device '{}'",
                signal.device_traits().name
            )));
        }
    }
    Ok(())
}

fn check_phase_on_rf_signal_support(
    pulse: &PlayPulse,
    signal: &SignalView,
    ctx: &VisitContext,
) -> Result<()> {
    if !matches!(signal.signal_kind(), SignalKind::Rf)
        || !signal.is_hardware_modulated()
        || signal.device_traits().supports_baseband_phase_modulation
    {
        return Ok(());
    }
    let valid_amplitude = match &pulse.amplitude {
        ValueOrParameter::Parameter(param_uid) => ctx
            .parameters
            .get(param_uid)
            .is_none_or(|x| matches!(*x.values, NumericArray::Float64(_))),
        ValueOrParameter::Value(ComplexOrFloat::Float(_)) => true,
        _ => false,
    };
    if !valid_amplitude || pulse.phase.is_some() {
        let err_msg = format!(
            "In section '{}', signal '{}': baseband phase modulation \
            not possible for RF signal with HW oscillator.",
            ctx.section_uid()?.0,
            signal.uid().0,
        );
        return Err(Error::new(&err_msg));
    }
    Ok(())
}

fn check_acquire_only_on_acquire_line(acquire: &Acquire, ctx: &VisitContext) -> Result<()> {
    let signal = ctx
        .signals
        .get(&acquire.signal)
        .expect("Expected signal to exist.");
    if !matches!(signal.signal_kind(), SignalKind::Integration) {
        let err_msg = format!(
            "In section '{}', an acquire statement is issued on \
            signal '{}'. `acquire` is only allowed on acquire lines.",
            ctx.section_uid()?.0,
            signal.uid().0,
        );
        return Err(Error::new(&err_msg));
    }
    Ok(())
}

fn check_no_play_on_acquire_line(signal: &SignalView, ctx: &VisitContext) -> Result<()> {
    match signal.signal_kind() {
        SignalKind::Integration => {
            let err_msg = format!(
                "In section {}, a play statement is issued on \
                signal {}. `play` is not allowed on acquire lines.",
                ctx.section_uid()?.0,
                signal.uid().0,
            );
            Err(Error::new(&err_msg))
        }
        _ => Ok(()),
    }
}

fn all_zero_or_one_samples(samples: &NumericArray) -> Result<bool> {
    match samples {
        NumericArray::Float64(v) => Ok(v.iter().all(|&x| x == 0.0 || x == 1.0)),
        NumericArray::Integer64(v) => Ok(v.iter().all(|&x| x == 0 || x == 1)),
        _ => Err(Error::new("Marker waveform cannot contain complex values")),
    }
}

fn check_arbitrary_marker_is_valid(
    pulse: &PlayPulse,
    signal: &SignalView,
    ctx: &VisitContext,
) -> Result<()> {
    if pulse.markers.is_empty() {
        return Ok(());
    }
    let section_uid = ctx.section_uid()?;

    for marker in pulse.markers.iter() {
        let Some(pulse_id) = marker.pulse_id.as_ref() else {
            continue;
        };

        let marker_pulse = ctx
            .pulses
            .get(pulse_id)
            .ok_or_else(|| Error::new("Internal error: pulse not found."))?;
        match &marker_pulse.kind {
            // TODO: should this case be moved to waveform_sampler ?
            PulseKind::Sampled(p) => {
                if all_zero_or_one_samples(&p.samples)? {
                    continue;
                }
                let err_msg = format!(
                    "A pulse in section '{}' attempts to play a sampled arbitrary marker with a sample not set to either 0 or 1. \
                    Please make sure that all samples of your markers are either 0 or 1.",
                    section_uid.0,
                );
                return Err(Error::new(&err_msg));
            }
            PulseKind::Functional(p) => {
                if !p.function.is_constant() {
                    let err_msg = format!(
                        "A pulse '{}' in section '{}' attempts to play an arbitrary marker with a functional pulse \
                        other than `const`. At this time, only constants pulses or sampled pulses are supported",
                        signal.uid().0,
                        section_uid.0,
                    );
                    return Err(Error::new(&err_msg));
                }
            }
            _ => continue,
        }
    }
    Ok(())
}

/// Validates that in real-time sections, no mixing of sections and (pulse) operations occurs.
///
/// This is an arbitrary restriction as scheduler currently does not support mixing of
/// operations and sections in real-time.
/// TODO: remove this restriction when scheduler supports it.
fn validate_no_mixed_operations(node: &ExperimentNode, ctx: &VisitContext) -> Result<()> {
    if !ctx.inside_rt_bound || node.children.len() <= 1 {
        return Ok(());
    }
    // Reserve is allowed to be mixed with sections
    let operations = node
        .children
        .iter()
        .filter(|child| !matches!(&child.kind, Operation::Reserve(_)));
    let (mut has_pulses, mut has_sections) = (false, false);
    for child in operations {
        if child.kind.section_info().is_none() {
            has_pulses = true;
        } else {
            has_sections = true;
        }
    }
    if has_pulses && has_sections {
        let msg = format!(
            "Sections and pulses cannot be mixed in section '{}'.",
            ctx.section_uid.map(|s| s.0.to_string()).unwrap_or_default(),
        );
        return Err(Error::new(&msg));
    }
    Ok(())
}

/// Rejects strict timing outside the real-time boundary, as near-time sections
/// cannot be strictly controlled.
fn validate_strict_timing_only_in_rt(node: &ExperimentNode, ctx: &VisitContext) -> Result<()> {
    if ctx.inside_rt_bound {
        return Ok(());
    }
    let (uid, timing_mode) = match &node.kind {
        Operation::Section(s) => (s.uid, s.section_timing_mode),
        Operation::Sweep(s) => (s.uid, s.section_timing_mode),
        Operation::Case(s) => (s.uid, s.section_timing_mode),
        _ => return Ok(()),
    };
    if timing_mode == SectionTimingMode::Strict {
        return Err(Error::new(format!(
            "Section '{}': SectionTimingMode.STRICT is only allowed inside the real-time loop",
            uid.0,
        )));
    }
    Ok(())
}

fn validate_averaging_loop(op: &AveragingLoop) -> Result<()> {
    if matches!(op.acquisition_type, AcquisitionType::Raw)
        && matches!(op.averaging_mode, AveragingMode::Sequential)
        && op.count.get() > 1
    {
        return Err(Error::new(
            "Sequential averaging is not supported for raw acquisitions.",
        ));
    }
    Ok(())
}

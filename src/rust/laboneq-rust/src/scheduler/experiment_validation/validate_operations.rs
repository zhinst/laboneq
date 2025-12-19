// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use pyo3::prelude::*;
use std::collections::{HashMap, HashSet};

use laboneq_common::types::DeviceKind;
use laboneq_scheduler::experiment::ExperimentNode;
use laboneq_scheduler::experiment::types::{
    Acquire, ComplexOrFloat, MarkerSelector, Operation, PlayPulse, SectionUid, ValueOrParameter,
};
use numeric_array::NumericArray;

use crate::error::{Error, Result};

use crate::scheduler::{
    experiment::SignalKind,
    experiment_validation::{
        ExperimentContext, ParamsContext, ValidationContext, validate_parameters::*,
        validate_pulses::*, validate_signals::*,
    },
    pulse::{PulseFunction, PulseKind},
};

/// Validates [Operation::_]s in an experiment.
pub(super) fn validate_experiment_operations(ctx: &ExperimentContext) -> Result<()> {
    let mut ctx_validator = ValidationContext {
        amplitude_check_done: Vec::with_capacity(ctx.pulses.len()),
        signal_pulse_map: HashMap::with_capacity(ctx.signals.len()),
        signal_check_done: Vec::with_capacity(ctx.signals.len()),
        section_uid: None,
        traversal_done: false,
    };

    let mut ctx_params = ParamsContext {
        inside_rt_bound: false,
        found_chunked_sweep: false,
        declared_sweep_parameters: ctx.parameters.keys().collect::<Vec<_>>(),
        rt_sweep_parameters: HashSet::with_capacity(ctx.parameters.len()),
        awgs_with_section_trigger: HashMap::with_capacity(ctx.signals.len()),
        awgs_with_automute: HashMap::with_capacity(ctx.signals.len()),
        awgs_with_ppc_sweeps: HashMap::with_capacity(ctx.signals.len()),
    };

    // TODO: we currently keep track of node-relevant visiting information, e.g. parent sections, rt / nt boundary,
    //  while traversing the experiment tree. it would be beneficial refactor this logic out of validation.
    visit_node(
        ctx.root_node,
        ctx,
        &mut ctx_validator,
        &mut ctx_params,
        None,
    )?;
    validate_sweep_parameters(&mut ctx_params)?;
    ctx_validator.traversal_done = true;
    check_ppc_sweeper(ctx, &ctx_validator, &mut ctx_params)?;
    Ok(())
}

fn visit_node<'a>(
    node: &'a ExperimentNode,
    ctx: &ExperimentContext<'a>,
    ctx_validator: &mut ValidationContext,
    ctx_params: &mut ParamsContext<'a>,
    parent_section_uid: Option<SectionUid>,
) -> Result<()> {
    let next_uid = node
        .kind
        .section_info()
        .map(|s| *s.uid)
        .or(parent_section_uid);

    let prev_uid = std::mem::replace(&mut ctx_validator.section_uid, next_uid);
    let mut children_visited = false;

    digest_rt_parameters(node, ctx_params);

    match &node.kind {
        Operation::PlayPulse(op) => {
            validate_play_pulse(op, ctx, ctx_validator)?;
            shfqa_unique_measure_pulse(op, ctx, ctx_validator)?;
            check_no_play_on_acquire_line(op, ctx, ctx_validator)?;
            check_arbitrary_marker_is_valid(op, ctx, ctx_validator)?;
            check_phase_on_rf_signal_support(op, ctx, ctx_validator)?;
            check_markers(op, ctx, ctx_validator)?;
        }
        Operation::Acquire(op) => {
            check_acquire_only_on_acquire_line(op, ctx, ctx_validator)?;
        }
        Operation::AveragingLoop(op) => {
            validate_experiment_signals(op, ctx, ctx_validator, ctx_params)?;
        }
        Operation::Section(op) => {
            digest_awg_triggers(op, ctx, ctx_params);
        }
        Operation::Sweep(op) => {
            digest_sweep_parameters(op, ctx_params);
            validate_chunked_sweep(op, ctx_params)?;
        }
        Operation::RealTimeBoundary => {
            ctx_params.enter_rt_bound();
            for child in node.children.iter() {
                visit_node(child, ctx, ctx_validator, ctx_params, next_uid)?;
            }
            ctx_params.exit_rt_bound();
            children_visited = true;
        }
        _ => {}
    };

    if !children_visited {
        for child in node.children.iter() {
            visit_node(child, ctx, ctx_validator, ctx_params, next_uid)?;
        }
    }

    ctx_validator.section_uid = prev_uid;
    Ok(())
}

fn shfqa_unique_measure_pulse(
    pulse: &PlayPulse,
    ctx: &ExperimentContext,
    ctx_validator: &mut ValidationContext,
) -> Result<()> {
    let signal = ctx
        .signals
        .get(&pulse.signal)
        .ok_or_else(|| Error::new("Signal not found."))?;

    let Some(pulse_uid) = pulse.pulse.as_ref() else {
        return Ok(());
    };
    if matches!(signal.signal_kind(), SignalKind::Integration)
        || !matches!(signal.device_kind(), DeviceKind::Shfqa)
    {
        return Ok(());
    }

    ctx_validator
        .signal_pulse_map
        .entry(signal.uid())
        .or_insert_with_key(|_key| HashSet::new())
        .insert(*pulse_uid);

    if ctx_validator
        .signal_pulse_map
        .get(&signal.uid())
        .is_none_or(|v| v.len() > 1)
    {
        let err_msg = format!(
            "Multiple different pulses are being played on signal {}. SHFQA \
            generators can only hold a single pulse waveform. Therefore, playing \
            multiple readout pulses represented by different Python objects is \
            not possible on a SHFQA measurement line.",
            signal.uid().0,
        );
        return Err(Error::new(&err_msg));
    }

    Ok(())
}

fn check_markers(
    pulse: &PlayPulse,
    ctx: &ExperimentContext,
    ctx_validator: &mut ValidationContext,
) -> Result<()> {
    let signal = ctx
        .signals
        .get(&pulse.signal)
        .expect("Expected signal to exist.");
    let Some(pulse_uid) = pulse.pulse.as_ref() else {
        return Ok(());
    };

    let section_uid = ctx_validator
        .section_uid
        .as_ref()
        .ok_or(Error::new("Internal error: section not found."))?;
    if matches!(signal.device_kind(), DeviceKind::Hdawg)
        && matches!(signal.signal_kind(), SignalKind::Rf)
        && signal.channels().len() == 1
        && pulse
            .markers
            .iter()
            .any(|m| matches!(m.marker_selector, MarkerSelector::M2))
    {
        let err_msg = format!(
            "Single channel RF Pulse {} referenced in section {} \
            has marker 2 enabled. Please only use marker 1 on RF channels.",
            pulse_uid.0, section_uid.0,
        );
        return Err(Error::new(&err_msg));
    }
    if signal.device_kind().is_qa_device() && !pulse.markers.is_empty() {
        let err_msg = format!(
            "Pulse {} referenced in section {} \
            has markers but is to be played on a QA device. QA \
            devices do not support markers.",
            pulse_uid.0, section_uid.0,
        );
        return Err(Error::new(&err_msg));
    }
    Ok(())
}

fn check_phase_on_rf_signal_support(
    pulse: &PlayPulse,
    ctx: &ExperimentContext,
    ctx_validator: &mut ValidationContext,
) -> Result<()> {
    let signal = ctx
        .signals
        .get(&pulse.signal)
        .expect("Expected signal to exist.");

    if !matches!(signal.signal_kind(), SignalKind::Rf)
        || !signal.is_hardware_modulated()
        || matches!(signal.device_kind(), DeviceKind::PrettyPrinterDevice)
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
        let section_uid = ctx_validator
            .section_uid
            .as_ref()
            .ok_or(Error::new("Internal error: section not found."))?;
        let err_msg = format!(
            "In section {}, signal {}: baseband phase modulation \
            not possible for RF signal with HW oscillator.",
            section_uid.0,
            signal.uid().0,
        );
        return Err(Error::new(&err_msg));
    }
    Ok(())
}

fn check_acquire_only_on_acquire_line(
    acquire: &Acquire,
    ctx: &ExperimentContext,
    ctx_validator: &mut ValidationContext,
) -> Result<()> {
    let signal = ctx
        .signals
        .get(&acquire.signal)
        .expect("Expected signal to exist.");
    if !matches!(signal.signal_kind(), SignalKind::Integration) {
        let section_uid = ctx_validator
            .section_uid
            .as_ref()
            .ok_or(Error::new("Internal error: section not found."))?;
        let err_msg = format!(
            "In section {}, an acquire statement is issued on \
            signal {}. acquire is only allowed on acquire lines.",
            section_uid.0,
            signal.uid().0,
        );
        return Err(Error::new(&err_msg));
    }
    Ok(())
}

fn check_no_play_on_acquire_line(
    pulse: &PlayPulse,
    ctx: &ExperimentContext,
    ctx_validator: &mut ValidationContext,
) -> Result<()> {
    let signal = ctx
        .signals
        .get(&pulse.signal)
        .expect("Expected signal to exist.");

    let section_uid = ctx_validator
        .section_uid
        .as_ref()
        .ok_or(Error::new("Internal error: section not found."))?;
    match signal.signal_kind() {
        SignalKind::Integration => {
            let err_msg = format!(
                "In section {}, a play statement is issued on \
                signal {}. play is not allowed on acquire lines.",
                section_uid.0,
                signal.uid().0,
            );
            Err(Error::new(&err_msg))
        }
        _ => Ok(()),
    }
}

fn all_zero_or_one_samples(samples: &Py<PyAny>) -> Result<bool> {
    Python::attach(|py| {
        let v = NumericArray::from_py(samples.bind(py))?;
        match v {
            NumericArray::Float64(v) => Ok(v.iter().all(|&x| x == 0.0 || x == 1.0)),
            NumericArray::Integer64(v) => Ok(v.iter().all(|&x| x == 0 || x == 1)),
            _ => Err(Error::new("Marker waveform cannot contain complex values")),
        }
    })
}

fn check_arbitrary_marker_is_valid(
    pulse: &PlayPulse,
    ctx: &ExperimentContext,
    ctx_validator: &mut ValidationContext,
) -> Result<()> {
    let signal = ctx
        .signals
        .get(&pulse.signal)
        .expect("Expected signal to exist.");
    let section_uid = ctx_validator
        .section_uid
        .as_ref()
        .ok_or(Error::new("Internal error: section not found."))?;

    for marker in pulse.markers.iter() {
        let Some(pulse_id) = marker.pulse_id.as_ref() else {
            continue;
        };

        let marker_pulse = ctx
            .pulses
            .get(pulse_id)
            .ok_or(Error::new("Internal error: pulse not found."))?;
        match &marker_pulse.kind {
            // TODO: should this case be moved to waveform_sampler ?
            PulseKind::Sampled(p) => {
                if all_zero_or_one_samples(&p.samples)? {
                    continue;
                }
                let err_msg = format!(
                    "A pulse in section {} attempts to play a sampled arbitrary marker with a sample not set to either 0 or 1. \
                    Please make sure that all samples of your markers are either 0 or 1.",
                    section_uid.0,
                );
                return Err(Error::new(&err_msg));
            }
            PulseKind::Functional(p) => match p.function {
                PulseFunction::Constant => continue,
                PulseFunction::Custom { ref function } => {
                    if function != PulseFunction::CONSTANT_PULSE_NAME {
                        let err_msg = format!(
                            "A pulse {} in section {} attempts to play an arbitrary marker with a pulse functional \
                            other than `const'. At this time, only constants pulses or sampled pulses are supported",
                            signal.uid().0,
                            section_uid.0,
                        );
                        return Err(Error::new(&err_msg));
                    }
                }
            },
            _ => continue,
        }
    }
    Ok(())
}

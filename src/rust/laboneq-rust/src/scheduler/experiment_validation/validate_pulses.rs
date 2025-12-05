// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use core::panic;
use std::fmt::Display;

use laboneq_scheduler::experiment::ExperimentNode;
use laboneq_scheduler::experiment::types::{
    ComplexOrFloat, Operation, ParameterUid, PlayPulse, ValueOrParameter,
};
use numeric_array::NumericArray;

use crate::error::{Error, Result};
use crate::scheduler::experiment::Experiment;

/// Validate [Operation::PlayPulse]s in the experiment.
///
/// - Ensures that the amplitude of each pulse does not exceed unity when specified as a value or parameter.
pub(super) fn validate_play_pulse_operations(experiment: &Experiment) -> Result<()> {
    let mut ctx = Context {
        experiment,
        amplitude_check_done: Vec::new(),
    };
    for section in &experiment.sections {
        visit_node(section, &mut ctx)?;
    }
    Ok(())
}

struct Context<'a> {
    experiment: &'a Experiment,
    amplitude_check_done: Vec<ParameterUid>,
}

fn visit_node(node: &ExperimentNode, ctx: &mut Context) -> Result<()> {
    match &node.kind {
        Operation::PlayPulse(op) => validate_play_pulse(op, ctx)?,
        _ => {
            for child in node.children.iter() {
                visit_node(child, ctx)?;
            }
        }
    }
    Ok(())
}

fn validate_play_pulse(op: &PlayPulse, ctx: &mut Context) -> Result<()> {
    match &op.amplitude {
        ValueOrParameter::Parameter(param_uid) => {
            if ctx.amplitude_check_done.contains(param_uid) {
                return Ok(());
            }
            let parameter = ctx
                .experiment
                .parameters
                .get(param_uid)
                .expect("Expected parameter to exist.");
            maximum_numeric_array_amplitude_magnitude(&parameter.values)?
                .map(|mag| {
                    if amplitude_exceeds_unity(mag) {
                        return Err(create_amplitude_exceeded_error(mag, op));
                    }
                    Ok(())
                })
                .transpose()?;
            ctx.amplitude_check_done.push(*param_uid);
        }
        ValueOrParameter::Value(amp) => {
            match amp {
                ComplexOrFloat::Float(v) => {
                    if amplitude_exceeds_unity(*v) {
                        return Err(create_amplitude_exceeded_error(v, op));
                    }
                }
                ComplexOrFloat::Complex(v) => {
                    if amplitude_exceeds_unity(v.norm()) {
                        return Err(create_amplitude_exceeded_error(v, op));
                    }
                }
            };
        }
        _ => panic!("Internal Error: Expected value not to be resolved."),
    }
    Ok(())
}

fn create_amplitude_exceeded_error(amp: impl Display, op: &PlayPulse) -> Error {
    Error::new(format!(
        "Magnitude of amplitude {} exceeding unity for play operation on pulse '{}' on signal '{}'",
        amp,
        op.pulse.expect("Expected pulse when amplitude is set").0,
        op.signal.0
    ))
}

fn amplitude_exceeds_unity(value: f64) -> bool {
    value.abs() > 1.0 + 1e-9
}

fn maximum_numeric_array_amplitude_magnitude(array: &NumericArray) -> Result<Option<f64>> {
    match array {
        NumericArray::Integer64(_) => Err(Error::new("Amplitude cannot be an integer.")),
        NumericArray::Float64(vec) => Ok(vec
            .iter()
            .map(|&x| x.abs())
            .fold(None, |acc, x| Some(acc.map_or(x, |a| a.max(x))))),
        NumericArray::Complex64(vec) => Ok(vec
            .iter()
            .map(|&x| x.norm())
            .fold(None, |acc, x| Some(acc.map_or(x, |a| a.max(x))))),
    }
}

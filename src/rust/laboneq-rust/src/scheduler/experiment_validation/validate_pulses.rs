// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use core::panic;
use std::fmt::Display;

use laboneq_scheduler::experiment::types::{ComplexOrFloat, PlayPulse, ValueOrParameter};
use numeric_array::NumericArray;

use crate::error::{Error, Result};
use crate::scheduler::experiment_validation::{ExperimentContext, ValidationContext};

pub(super) fn validate_play_pulse(
    op: &PlayPulse,
    ctx: &ExperimentContext,
    ctx_validator: &mut ValidationContext,
) -> Result<()> {
    match &op.amplitude {
        ValueOrParameter::Parameter(param_uid) => {
            if ctx_validator.amplitude_check_done.contains(param_uid) {
                return Ok(());
            }
            let parameter = ctx
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
            ctx_validator.amplitude_check_done.push(*param_uid);
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

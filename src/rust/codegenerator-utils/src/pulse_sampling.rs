// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use laboneq_common::named_id::NamedIdStore;
use laboneq_dsl::operation::ExternalOrValue;
use laboneq_dsl::types::{self, NumericLiteral, PulseFunction, PulseParameterUid, PulseUid};
use laboneq_pulse_sampling::builtin::{BuiltinPulse, ParameterValue, Parameters};
use laboneq_pulse_sampling::{EnvelopeShape, FunctionalShape};
use num_complex::Complex64;
use std::collections::HashMap;

/// Resolves the pulse `parameters` that sampling reads: those of standard pulses,
/// and the `amplitude` scaling factor, `None` when it is not a number.
pub fn resolve_pulse_parameters(
    parameters: Option<&HashMap<PulseParameterUid, ExternalOrValue>>,
    id_store: &NamedIdStore,
) -> (Parameters, Option<Complex64>) {
    let mut resolved = Parameters::default();
    let mut amplitude = Some(Complex64::ONE);
    for (uid, parameter) in parameters.into_iter().flatten() {
        let slot = match id_store.resolve_unchecked(*uid) {
            "amplitude" => {
                amplitude = amplitude_factor(parameter);
                continue;
            }
            "sigma" => &mut resolved.sigma,
            "beta" => &mut resolved.beta,
            "width" => &mut resolved.width,
            "order" => &mut resolved.order,
            "zero_boundaries" => &mut resolved.zero_boundaries,
            // reserved names for functional pulse arguments
            "x" | "length" | "sampling_rate" => {
                resolved.reuses_argument_name = true;
                continue;
            }
            _ => continue,
        };
        *slot = Some(value(parameter));
    }
    (resolved, amplitude)
}

/// Returns the factor by which an `amplitude` pulse parameter scales a pulse:
/// 0 or 1 for a boolean, `None` when not a number.
fn amplitude_factor(parameter: &ExternalOrValue) -> Option<Complex64> {
    match parameter {
        ExternalOrValue::ValueOrParameter(value) => value.value().map(|v| v.to_complex()),
        ExternalOrValue::Bool(value) => Some(Complex64::from(f64::from(u8::from(*value)))),
        _ => None,
    }
}

/// A functional pulse as specified in the experiment: its uid, function, and parameters.
#[derive(Debug, Clone, Copy)]
pub struct PulseSpec<'a> {
    /// Names the pulse in error messages, if it has a name.
    pub pulse_id: Option<PulseUid>,
    pub function: &'a PulseFunction,
    /// Pulse parameters, as pulse functions receive them.
    pub parameters: Option<&'a HashMap<PulseParameterUid, ExternalOrValue>>,
}

impl PulseSpec<'_> {
    /// Returns the shape of this pulse for a length of `length` samples.
    pub fn shape<'s>(self, length: usize) -> EnvelopeShape<'s, Self> {
        EnvelopeShape::Functional(FunctionalShape {
            builtin: builtin(self.function),
            pulse: self,
            length,
        })
    }
}

fn value(value: &ExternalOrValue) -> ParameterValue {
    match value {
        ExternalOrValue::ValueOrParameter(value) => match value.value() {
            Some(NumericLiteral::Int(value)) => ParameterValue::Int(value),
            Some(NumericLiteral::Float(value)) => ParameterValue::Float(value),
            Some(NumericLiteral::Complex(_)) | None => ParameterValue::Opaque,
        },
        ExternalOrValue::Bool(value) => ParameterValue::Bool(*value),
        ExternalOrValue::ExternalParameter(_) | ExternalOrValue::None => ParameterValue::Opaque,
    }
}

/// Returns the [`BuiltinPulse`] for a standard `function`, `None` for a custom one.
fn builtin(function: &PulseFunction) -> Option<BuiltinPulse> {
    match function {
        PulseFunction::Builtin(pulse) => Some(match pulse {
            types::BuiltinPulse::Const => BuiltinPulse::Const,
            types::BuiltinPulse::Gaussian => BuiltinPulse::Gaussian,
            types::BuiltinPulse::Drag => BuiltinPulse::Drag,
            types::BuiltinPulse::GaussianSquare => BuiltinPulse::GaussianSquare,
        }),
        PulseFunction::Custom { .. } => None,
    }
}

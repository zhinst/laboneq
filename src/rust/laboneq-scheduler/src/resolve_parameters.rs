// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use anyhow::Context;
use laboneq_dsl::operation::PulseParameterValue;
use laboneq_dsl::types::{ParameterUid, SweepParameter, ValueOrParameter};
use laboneq_units::tinysample::seconds_to_tinysamples;

use crate::error::{Error, Result};
use crate::parameter_resolver::ParameterResolver;
use crate::utils::round_to_grid;
use crate::{ParameterStore, ScheduledNode};
use laboneq_ir::IrKind;

/// This function modifies the IR in place to resolve all real-time parameter references to their concrete values.
///
/// This function assumes that all loops have been fully unrolled prior to calling it.
pub(crate) fn resolve_parameters(
    ir: &mut ScheduledNode,
    parameters: &HashMap<ParameterUid, SweepParameter>,
    nt_parameters: &ParameterStore,
) -> Result<()> {
    let resolver = ParameterResolver::new(parameters, nt_parameters);
    resolve_parameters_impl(ir, &resolver)
}

fn resolve_parameters_impl(ir: &mut ScheduledNode, resolver: &ParameterResolver) -> Result<()> {
    match &mut ir.kind {
        IrKind::Loop(obj) => {
            let parameters = &obj.parameters();
            let mut resolver = resolver.child_scope(parameters)?;
            // Check whether the loops are fully unrolled. Currently partial unrolling is not supported.
            if !parameters.is_empty() && ir.children.len() != obj.iterations.get() as usize {
                return Err(Error::new(format!(
                    "Expected loop to be unrolled. Mismatch between loop iterations ({}) and number of children ({})",
                    obj.iterations.get(),
                    ir.children.len()
                )));
            }
            for (iteration, child) in ir.children.iter_mut().enumerate() {
                for param in parameters.iter() {
                    resolver.set_iteration(*param, iteration)?;
                }
                resolve_parameters_impl(child.node.make_mut(), &resolver)?;
            }
            Ok(())
        }
        _ => {
            resolve_parameter_fields(ir, resolver)?;
            for child in ir.children.iter_mut() {
                resolve_parameters_impl(child.node.make_mut(), resolver)?;
            }
            Ok(())
        }
    }
}

/// Resolves parameters of a node in-place.
fn resolve_parameter_fields(node: &mut ScheduledNode, resolver: &ParameterResolver) -> Result<()> {
    match &mut node.kind {
        IrKind::SetOscillatorFrequency(obj) => {
            for (_, value) in &mut obj.values {
                if let ValueOrParameter::Parameter(param_uid) = value {
                    *value = ValueOrParameter::Value(
                        resolver
                            .get_value(param_uid)?
                            .try_into()
                            .map_err(Error::new)
                            .with_context(
                                || "Oscillator frequency must be a real number (integer or float).",
                            )?,
                    );
                }
            }
        }
        IrKind::PpcStep(obj) => {
            if let Some(ValueOrParameter::Parameter(param_uid)) = &obj.pump_power {
                obj.pump_power = Some(ValueOrParameter::Value(
                    resolver
                        .get_value(param_uid)?
                        .try_into()
                        .map_err(Error::new)
                        .with_context(
                            || "PPC pump power must be a real number (integer or float).",
                        )?,
                ));
            }
            if let Some(ValueOrParameter::Parameter(param_uid)) = &obj.pump_frequency {
                obj.pump_frequency = Some(ValueOrParameter::Value(
                    resolver
                        .get_value(param_uid)?
                        .try_into()
                        .map_err(Error::new)
                        .with_context(
                            || "PPC pump frequency must be a real number (integer or float).",
                        )?,
                ));
            }
            if let Some(ValueOrParameter::Parameter(param_uid)) = &obj.probe_power {
                obj.probe_power = Some(ValueOrParameter::Value(
                    resolver
                        .get_value(param_uid)?
                        .try_into()
                        .map_err(Error::new)
                        .with_context(
                            || "PPC probe power must be a real number (integer or float).",
                        )?,
                ));
            }
            if let Some(ValueOrParameter::Parameter(param_uid)) = &obj.probe_frequency {
                obj.probe_frequency = Some(ValueOrParameter::Value(
                    resolver
                        .get_value(param_uid)?
                        .try_into()
                        .map_err(Error::new)
                        .with_context(
                            || "PPC probe frequency must be a real number (integer or float).",
                        )?,
                ));
            }
            if let Some(ValueOrParameter::Parameter(param_uid)) = &obj.cancellation_phase {
                obj.cancellation_phase = Some(ValueOrParameter::Value(
                    resolver
                        .get_value(param_uid)?
                        .try_into()
                        .map_err(Error::new)
                        .with_context(
                            || "PPC cancellation phase must be a real number (integer or float).",
                        )?,
                ));
            }
            if let Some(ValueOrParameter::Parameter(param_uid)) = &obj.cancellation_attenuation {
                obj.cancellation_attenuation = Some(ValueOrParameter::Value(
                    resolver.get_value(param_uid)?.try_into().map_err(Error::new).with_context(
                        || "PPC cancellation attenuation must be a real number (integer or float).",
                    )?,
                ));
            }
        }
        IrKind::Acquire(obj) => {
            for p in &mut obj.pulse_parameters {
                for value in p.values_mut() {
                    if let PulseParameterValue::ValueOrParameter(ValueOrParameter::Parameter(
                        param_uid,
                    )) = value
                    {
                        *value = PulseParameterValue::ValueOrParameter(ValueOrParameter::Value(
                            resolver.get_value(param_uid)?,
                        ));
                    }
                }
            }
            for p in &mut obj.parameters {
                for value in p.values_mut() {
                    if let PulseParameterValue::ValueOrParameter(ValueOrParameter::Parameter(
                        param_uid,
                    )) = value
                    {
                        *value = PulseParameterValue::ValueOrParameter(ValueOrParameter::Value(
                            resolver.get_value(param_uid)?,
                        ));
                    }
                }
            }
        }
        IrKind::Delay { .. } => {
            if let Some(param_uid) = node.schedule.length_param {
                let length_step_seconds: f64 = resolver
                    .get_value(&param_uid)?
                    .try_into()
                    .map_err(Error::new)
                    .with_context(|| "Delay must be a real number (integer or float).")?;
                let length_tinysample = seconds_to_tinysamples(length_step_seconds.into());
                node.schedule.resolve_length(
                    round_to_grid(length_tinysample.value(), node.schedule.grid.value()).into(),
                );
            }
        }
        IrKind::PlayPulse(obj) => {
            if let ValueOrParameter::Parameter(param_uid) = &obj.amplitude {
                obj.amplitude = ValueOrParameter::ResolvedParameter {
                    value: resolver
                        .get_value(param_uid)?
                        .try_into()
                        .map_err(Error::new)
                        .with_context(|| "Pulse amplitude must be complex or float value.")?,
                    uid: *param_uid,
                };
            }
            if let Some(phase) = obj.phase.as_mut()
                && let ValueOrParameter::Parameter(param_uid) = phase
            {
                *phase = ValueOrParameter::Value(
                    resolver
                        .get_value(param_uid)?
                        .try_into()
                        .map_err(Error::new)
                        .with_context(|| "Pulse phase must be a numeric value.")?,
                );
            }
            if let Some(increment) = obj.increment_oscillator_phase.as_mut()
                && let ValueOrParameter::Parameter(param_uid) = increment
            {
                *increment = ValueOrParameter::ResolvedParameter {
                    value: resolver
                        .get_value(param_uid)?
                        .try_into()
                        .map_err(Error::new)
                        .with_context(
                            || "Pulse increment oscillator phase must be a numeric value.",
                        )?,
                    uid: *param_uid,
                };
            }
            if let Some(set_phase) = obj.set_oscillator_phase.as_mut()
                && let ValueOrParameter::Parameter(param_uid) = set_phase
            {
                *set_phase = ValueOrParameter::Value(
                    resolver
                        .get_value(param_uid)?
                        .try_into()
                        .map_err(Error::new)
                        .with_context(|| "Pulse set oscillator phase must be a numeric value.")?,
                );
            }
            for value in obj.pulse_parameters.values_mut() {
                if let PulseParameterValue::ValueOrParameter(ValueOrParameter::Parameter(
                    param_uid,
                )) = value
                {
                    *value = PulseParameterValue::ValueOrParameter(ValueOrParameter::Value(
                        resolver.get_value(param_uid)?,
                    ));
                }
            }
            for value in obj.parameters.values_mut() {
                if let PulseParameterValue::ValueOrParameter(ValueOrParameter::Parameter(
                    param_uid,
                )) = value
                {
                    *value = PulseParameterValue::ValueOrParameter(ValueOrParameter::Value(
                        resolver.get_value(param_uid)?,
                    ));
                }
            }
            if let Some(param_uid) = node.schedule.length_param {
                let length_step_seconds: f64 = resolver
                    .get_value(&param_uid)?
                    .try_into()
                    .map_err(Error::new)
                    .with_context(
                        || "Play pulse length must be a real number (integer or float).",
                    )?;
                let length_tinysample = seconds_to_tinysamples(length_step_seconds.into());
                node.schedule.resolve_length(
                    round_to_grid(length_tinysample.value(), node.schedule.grid.value()).into(),
                );
            }
        }
        IrKind::ChangeOscillatorPhase(obj) => {
            if let Some(increment) = obj.increment.as_mut()
                && let ValueOrParameter::Parameter(param_uid) = increment
            {
                *increment = ValueOrParameter::ResolvedParameter {
                    value: resolver
                        .get_value(param_uid)?
                        .try_into()
                        .map_err(Error::new)
                        .with_context(
                            || "Change oscillator phase increment must be a numeric value.",
                        )?,
                    uid: *param_uid,
                };
            }
            if let Some(set_phase) = obj.set.as_mut()
                && let ValueOrParameter::Parameter(param_uid) = set_phase
            {
                *set_phase = ValueOrParameter::ResolvedParameter {
                    value: resolver
                        .get_value(param_uid)?
                        .try_into()
                        .map_err(Error::new)
                        .with_context(
                            || "Change oscillator phase set value must be a numeric value.",
                        )?,
                    uid: *param_uid,
                };
            }
        }
        _ => {}
    }
    Ok(())
}

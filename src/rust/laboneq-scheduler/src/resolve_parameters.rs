// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use anyhow::Context;

use crate::error::{Error, Result};
use crate::experiment::sweep_parameter::SweepParameter;
use crate::experiment::types::{ParameterUid, ValueOrParameter};
use crate::ir::IrKind;
use crate::parameter_resolver::ParameterResolver;
use crate::{ParameterStore, ScheduledNode};

/// This function modifies the IR in place to resolve all real-time parameter references to their concrete values.
///
/// This function assumes that all loops have been fully unrolled prior to calling it.
pub fn resolve_parameters(
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
            let mut resolver = resolver.child_scope();
            // Check whether the loops are fully unrolled. Currently partial unrolling is not supported.
            if !obj.parameters.is_empty() && ir.children.len() != obj.iterations {
                return Err(Error::new(format!(
                    "Expected loop to be unrolled. Mismatch between loop iterations ({}) and number of children ({})",
                    obj.iterations,
                    ir.children.len()
                )));
            }
            for (iteration, child) in ir.children.iter_mut().enumerate() {
                for param in obj.parameters.iter() {
                    resolver.set_iteration(*param, iteration);
                }
                resolve_parameters_impl(child.node.make_mut(), &resolver)?;
            }
            Ok(())
        }
        _ => {
            resolve_parameter_fields(&mut ir.kind, resolver)?;
            for child in ir.children.iter_mut() {
                resolve_parameters_impl(child.node.make_mut(), resolver)?;
            }
            Ok(())
        }
    }
}

/// Resolves parameters of a node in-place.
fn resolve_parameter_fields(ir: &mut IrKind, resolver: &ParameterResolver) -> Result<()> {
    match ir {
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
        _ => {}
    }
    Ok(())
}

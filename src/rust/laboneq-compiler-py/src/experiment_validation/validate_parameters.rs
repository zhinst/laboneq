// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use crate::error::{Error, Result};
use crate::experiment_validation::{ExperimentContext, ParamsContext, ValidationContext};
use crate::signal_view::SignalView;
use laboneq_common::types::DeviceKind;
use laboneq_dsl::ExperimentNode;
use laboneq_dsl::operation::{Section, Sweep};
use laboneq_dsl::types::ValueOrParameter;

/// Validates sweep parameters in an [`Experiment`].
pub(super) fn validate_sweep_parameters(ctx: &mut ParamsContext) -> Result<()> {
    if !ctx.declared_sweep_parameters.is_empty() {
        ctx.declared_sweep_parameters.sort();
        return Err(Error::new(format!(
            "The experiment contains sweep parameters that are not defined in any sweeps: {}.",
            ctx.declared_sweep_parameters
                .iter()
                .map(|p| format!("'{}'", p.0))
                .collect::<Vec<_>>()
                .join(", ")
        )));
    }
    Ok(())
}

pub(super) fn digest_sweep_parameters(node: &Sweep, ctx: &mut ParamsContext) {
    let params_set = node
        .parameters
        .iter()
        .collect::<std::collections::HashSet<_>>();
    ctx.declared_sweep_parameters
        .retain(|p| !params_set.contains(p));
}

pub(super) fn digest_rt_parameters<'a>(node: &'a ExperimentNode, ctx: &mut ParamsContext<'a>) {
    if ctx.inside_rt_bound
        && let Some(l) = &node.kind.loop_info()
    {
        ctx.rt_sweep_parameters.extend(l.parameters.iter());
    }
}

pub(super) fn digest_awg_triggers<'a>(
    section: &'a Section,
    ctx: &ExperimentContext<'a>,
    ctx_params: &mut ParamsContext<'a>,
) {
    for trigger in &section.triggers {
        let Some(trigger_signal) = ctx.signals.get(&trigger.signal) else {
            continue;
        };
        if matches!(trigger_signal.device_kind(), DeviceKind::Shfqa) {
            ctx_params
                .awgs_with_section_trigger
                .insert(trigger_signal.awg_key(), trigger_signal);
        }
    }
}

fn digest_ppc_params<'a>(signal: &'a SignalView, ctx_params: &mut ParamsContext<'a>) {
    let Some(amplifier_pump) = signal.amplifier_pump() else {
        return;
    };

    for param in amplifier_pump.values_or_parameters() {
        let Some(p) = param else {
            continue;
        };
        match p {
            ValueOrParameter::Parameter(uid) | ValueOrParameter::ResolvedParameter { uid, .. }
                if ctx_params.rt_sweep_parameters.contains(uid) =>
            {
                ctx_params
                    .awgs_with_ppc_sweeps
                    .insert(signal.awg_key(), signal);
            }
            _ => {}
        }
    }
}

pub(super) fn check_ppc_sweeper<'a>(
    ctx: &ExperimentContext<'a>,
    ctx_validator: &ValidationContext,
    ctx_params: &mut ParamsContext<'a>,
) -> Result<()> {
    if !ctx_validator.traversal_done {
        return Err(Error::new("Experiment tree traversal not yet done."));
    }

    for signal in ctx.signals.values() {
        digest_ppc_params(signal, ctx_params);
    }

    let mut conflicts = ctx_params
        .awgs_with_automute
        .keys()
        .filter_map(|&k| ctx_params.awgs_with_ppc_sweeps.get(k).copied())
        .collect::<Vec<&SignalView<'_>>>();

    if !conflicts.is_empty() {
        let msg = format!(
            "Signals on the following channels drive both SHFPPC sweeps, and use the \
            output auto-muting feature: {}",
            conflicts
                .iter()
                .map(|s| format!("{}", s.uid().0))
                .collect::<Vec<String>>()
                .join(", ")
        );
        return Err(Error::new(msg));
    }

    conflicts = ctx_params
        .awgs_with_section_trigger
        .keys()
        .filter_map(|&k| ctx_params.awgs_with_ppc_sweeps.get(k).copied())
        .collect::<Vec<&SignalView<'_>>>();

    if !conflicts.is_empty() {
        let msg = format!(
            "Signals on the following channels drive both SHFPPC sweeps, and use \
            section triggers: {}",
            conflicts
                .into_iter()
                .map(|s| format!("{}", s.uid().0))
                .collect::<Vec<String>>()
                .join(", ")
        );
        return Err(Error::new(msg));
    }
    Ok(())
}

pub(super) fn validate_chunked_sweep(sweep: &Sweep, ctx_params: &mut ParamsContext) -> Result<()> {
    if !sweep.is_chunked() {
        return Ok(());
    }

    if !ctx_params.inside_rt_bound {
        let err_msg = "Sweeps that are not inside real-time execution cannot be chunked.";
        return Err(Error::new(err_msg));
    }
    if ctx_params.found_chunked_sweep {
        let err_msg = "Found multiple chunked sweeps.";
        return Err(Error::new(err_msg));
    }
    ctx_params.found_chunked_sweep = true;
    Ok(())
}

pub(super) fn validate_no_nt_parameter_in_rt_sweep(
    sweep: &Sweep,
    ctx: &ExperimentContext,
    ctx_params: &ParamsContext,
) -> Result<()> {
    if !ctx_params.inside_rt_bound {
        return Ok(());
    }
    for param_uid in &sweep.parameters {
        if ctx.nt_only_parameters.contains(param_uid) {
            return Err(Error::new(format!(
                "Parameter '{}' can't be swept in real-time, it is bound to a value \
                 that can only be set in near-time.",
                param_uid.0,
            )));
        }
    }
    Ok(())
}

pub(super) fn validate_parallel_sweep(sweep: &Sweep, ctx: &ExperimentContext) -> Result<()> {
    let mut param_counts = sweep
        .parameters
        .iter()
        .filter_map(|p| ctx.parameters.get(p).map(|p| p.len()));

    if let Some(first_count) = param_counts.next()
        && !param_counts.all(|count| count == first_count)
    {
        let err_msg = format!(
            "Parallel executed sweep parameters must be of same length in sweep '{}'.",
            sweep.uid.0,
        );
        return Err(Error::new(err_msg));
    }
    Ok(())
}

// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use crate::error::{Error, Result};
use crate::experiment_validation::traversal::VisitContext;
use laboneq_dsl::operation::Sweep;
use laboneq_dsl::types::ParameterUid;

/// Validates sweep parameters in an [`Experiment`].
pub(super) fn validate_sweep_parameters(declared_parameters: &mut [ParameterUid]) -> Result<()> {
    if !declared_parameters.is_empty() {
        declared_parameters.sort();
        return Err(Error::new(format!(
            "The experiment contains sweep parameters that are not defined in any sweeps: {}.",
            declared_parameters
                .iter()
                .map(|p| format!("'{}'", p.0))
                .collect::<Vec<_>>()
                .join(", ")
        )));
    }
    Ok(())
}

pub(super) fn digest_sweep_parameters(node: &Sweep, declared_parameters: &mut Vec<ParameterUid>) {
    let params_set = node
        .parameters
        .iter()
        .collect::<std::collections::HashSet<_>>();
    declared_parameters.retain(|p| !params_set.contains(p));
}

pub(super) fn validate_chunked_sweep(
    sweep: &Sweep,
    ctx: &VisitContext,
    found_chunked_sweep: &mut bool,
) -> Result<()> {
    if !sweep.is_chunked() {
        return Ok(());
    }

    if !ctx.inside_rt_bound {
        let err_msg = "Sweeps that are not inside real-time execution cannot be chunked.";
        return Err(Error::new(err_msg));
    }
    if *found_chunked_sweep {
        let err_msg = "Found multiple chunked sweeps.";
        return Err(Error::new(err_msg));
    }
    *found_chunked_sweep = true;
    Ok(())
}

pub(super) fn validate_no_nt_parameter_in_rt_sweep(
    sweep: &Sweep,
    ctx: &VisitContext,
) -> Result<()> {
    if !ctx.inside_rt_bound {
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

pub(super) fn validate_parallel_sweep(sweep: &Sweep, ctx: &VisitContext) -> Result<()> {
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

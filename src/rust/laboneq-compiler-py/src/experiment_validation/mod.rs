// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use crate::error::Result;

use crate::compiler_backend::{CompilerBackend, ProcessedExperiment};
use crate::experiment::Experiment;
use crate::real_time_loop::real_time_loop_properties;
use crate::signal_view::signal_views;
use laboneq_ir::signal::Signal;
use std::collections::HashSet;

mod validate_parameters;
use laboneq_dsl::types::ParameterUid;

mod traversal;
mod validate_operations;
mod validate_pulses;
mod validate_unique_sections;

use traversal::{VisitContext, validate_tree};
use validate_operations::OperationValidator;
use validate_unique_sections::validate_unique_sections;

/// Validates an [`Experiment`], against the compiler's own rules and the
/// backend's.
pub(super) fn validate_experiment<B: CompilerBackend>(
    experiment: &Experiment,
    setup_signals: &[Signal],
    backend: &B,
    backend_data: &B::Output,
) -> Result<()> {
    validate_unique_sections(&experiment.root)?;
    let acquisition_type = real_time_loop_properties(&experiment.root)?.acquisition_type;
    let nt_only_parameters = collect_nt_only_parameters(setup_signals);
    let parameters = experiment.parameters.iter().map(|p| (p.uid, p)).collect();
    let signals = signal_views(setup_signals);
    let ctx = VisitContext {
        pulses: &experiment.pulses,
        parameters: &parameters,
        signals: &signals,
        nt_only_parameters: &nt_only_parameters,
        section_uid: None,
        inside_rt_bound: false,
    };
    validate_tree(&experiment.root, ctx, &mut OperationValidator::new(&ctx))?;

    let processed = ProcessedExperiment {
        root: &experiment.root,
        parameters: experiment
            .parameters
            .iter()
            .map(|p| (p.uid, p.inner_values().as_ref()))
            .collect(),
        acquisition_type,
    };
    Ok(backend.validate(&processed, backend_data)?)
}

/// Return the UIDs of parameters bound to near-time-only calibration fields on a signal.
fn nt_only_parameter_uids(signal: &Signal) -> impl Iterator<Item = ParameterUid> + '_ {
    let Signal {
        amplitude,
        lo_frequency,
        voltage_offset,
        port_delay,
        added_outputs,
        // Not NT-only — listed explicitly so the compiler catches new fields:
        uid: _,
        awg_key: _,
        device_uid: _,
        physical_device_uid: _,
        traits: _,
        sampling_rate: _,
        port_mode: _,
        kind: _,
        oscillator: _,
        amplifier_pump: _,
        automute: _,
        range: _,
        precompensation: _,
        thresholds: _,
        mixer_calibration: _,
        start_delay: _,
        signal_alignment_delay: _,
        signal_delay: _,
    } = signal;

    let scalar_uids = [
        amplitude.as_ref().and_then(|v| v.parameter_uid()),
        lo_frequency.as_ref().and_then(|v| v.parameter_uid()),
        voltage_offset.as_ref().and_then(|v| v.parameter_uid()),
        port_delay.as_ref().and_then(|v| v.parameter_uid()),
    ]
    .into_iter()
    .flatten();

    let output_uids = added_outputs.iter().flat_map(|output| {
        [
            output
                .amplitude_scaling
                .as_ref()
                .and_then(|v| v.parameter_uid()),
            output.phase_shift.as_ref().and_then(|v| v.parameter_uid()),
        ]
        .into_iter()
        .flatten()
    });

    scalar_uids.chain(output_uids)
}

/// Collect the UIDs of all parameters bound to near-time sweeps only.
fn collect_nt_only_parameters(signals: &[Signal]) -> HashSet<ParameterUid> {
    signals.iter().flat_map(nt_only_parameter_uids).collect()
}

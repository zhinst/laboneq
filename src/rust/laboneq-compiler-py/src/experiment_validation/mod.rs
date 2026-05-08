// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use crate::error::Result;

use crate::experiment::Experiment;
use crate::signal_view::{SignalView, signal_views};
use laboneq_ir::signal::Signal;
use laboneq_ir::system::DeviceSetup;
use std::collections::{HashMap, HashSet};

mod validate_parameters;
use laboneq_common::types::AwgKey;
use laboneq_dsl::{
    ExperimentNode,
    types::{ParameterUid, PulseDef, PulseUid, SectionUid, SignalUid, SweepParameter},
};

mod validate_operations;
mod validate_pulses;
mod validate_signals;
mod validate_unique_sections;

use validate_operations::validate_experiment_operations;
use validate_unique_sections::validate_unique_sections;

/// Validates an [`Experiment`].
pub(super) fn validate_experiment(
    experiment: &Experiment,
    device_setup: &DeviceSetup,
) -> Result<()> {
    validate_unique_sections(&experiment.root)?;
    let nt_only_parameters = collect_nt_only_parameters(device_setup);
    let ctx = ExperimentContext {
        root_node: &experiment.root,
        pulses: &experiment.pulses,
        parameters: &experiment.parameters,
        signals: &signal_views(device_setup),
        nt_only_parameters: &nt_only_parameters,
    };
    validate_experiment_operations(&ctx)?;
    Ok(())
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
        sampling_rate: _,
        port_mode: _,
        ports: _,
        kind: _,
        oscillator: _,
        amplifier_pump: _,
        automute: _,
        range: _,
        precompensation: _,
        thresholds: _,
        mixer_calibration: _,
        start_delay: _,
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
fn collect_nt_only_parameters(setup: &DeviceSetup) -> HashSet<ParameterUid> {
    setup.signals().flat_map(nt_only_parameter_uids).collect()
}

struct ExperimentContext<'a> {
    root_node: &'a ExperimentNode,
    pulses: &'a HashMap<PulseUid, PulseDef>,
    parameters: &'a HashMap<ParameterUid, SweepParameter>,
    signals: &'a HashMap<SignalUid, SignalView<'a>>,
    nt_only_parameters: &'a HashSet<ParameterUid>,
}

struct ValidationContext {
    amplitude_check_done: Vec<ParameterUid>,
    signal_check_done: Vec<SignalUid>,
    signal_pulse_map: HashMap<SignalUid, HashSet<PulseUid>>,
    section_uid: Option<SectionUid>,
    traversal_done: bool,
}

struct ParamsContext<'a> {
    inside_rt_bound: bool,
    found_chunked_sweep: bool,
    declared_sweep_parameters: Vec<&'a ParameterUid>,
    rt_sweep_parameters: HashSet<&'a ParameterUid>,
    awgs_with_section_trigger: HashMap<&'a AwgKey, &'a SignalView<'a>>,
    awgs_with_automute: HashMap<&'a AwgKey, &'a SignalView<'a>>,
    awgs_with_ppc_sweeps: HashMap<&'a AwgKey, &'a SignalView<'a>>,
}

impl ParamsContext<'_> {
    fn enter_rt_bound(&mut self) {
        self.inside_rt_bound = true;
    }

    fn exit_rt_bound(&mut self) {
        self.inside_rt_bound = false;
    }
}

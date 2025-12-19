// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use crate::{
    error::Result,
    scheduler::{
        experiment::{DeviceSetup, Experiment},
        pulse::PulseDef,
        signal_view::{SignalView, signal_views},
    },
};
use std::collections::{HashMap, HashSet};

mod validate_parameters;
use laboneq_common::types::AwgKey;
use laboneq_scheduler::experiment::{
    ExperimentNode,
    sweep_parameter::SweepParameter,
    types::{ParameterUid, PulseUid, SectionUid, SignalUid},
};

mod validate_operations;
mod validate_pulses;
mod validate_signals;

use validate_operations::validate_experiment_operations;

/// Validates an [`Experiment`].
pub(super) fn validate_experiment(
    experiment: &Experiment,
    device_setup: &DeviceSetup,
) -> Result<()> {
    let ctx = ExperimentContext {
        root_node: &experiment.root,
        pulses: &experiment.pulses,
        parameters: &experiment.parameters,
        signals: &signal_views(device_setup),
    };
    validate_experiment_operations(&ctx)?;
    Ok(())
}

struct ExperimentContext<'a> {
    root_node: &'a ExperimentNode,
    pulses: &'a HashMap<PulseUid, PulseDef>,
    parameters: &'a HashMap<ParameterUid, SweepParameter>,
    signals: &'a HashMap<SignalUid, SignalView<'a>>,
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

// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

//! Shared Python conversion helpers used by the Cap'n Proto serializer/deserializer path.

use crate::error::Result;
use pyo3::intern;
use pyo3::prelude::*;
use std::collections::HashMap;

#[derive(Debug, Hash, Eq, PartialEq)]
pub(crate) enum DslType {
    LinearSweepParameter,
    SweepParameter,
    Parameter,
    Sweep,
    Section,
    Delay,
    Reserve,
    Acquire,
    PlayPulse,
    Match,
    Case,
    PulseFunctional,
    AcquireLoopRt,
    Call,
    PrngSetup,
    PrngLoop,
    SetNode,
    ResetOscillatorPhase,
}

pub(crate) struct DslTypes<'a> {
    type_map: HashMap<DslType, Bound<'a, PyAny>>,
}

impl<'a> DslTypes<'a> {
    pub(crate) fn new(py: Python<'a>) -> Result<DslTypes<'a>> {
        let linear_sweep_parameter_py = py
            .import(intern!(py, "laboneq.dsl.parameter"))?
            .getattr(intern!(py, "LinearSweepParameter"))?;
        let sweep_parameter_py: Bound<'_, PyAny> = py
            .import(intern!(py, "laboneq.dsl.parameter"))?
            .getattr(intern!(py, "SweepParameter"))?;
        let parameter_py: Bound<'_, PyAny> = py
            .import(intern!(py, "laboneq.dsl.parameter"))?
            .getattr(intern!(py, "Parameter"))?;
        let sweep_py = py
            .import(intern!(py, "laboneq.dsl.experiment.section"))?
            .getattr(intern!(py, "Sweep"))?;
        let section_py = py
            .import(intern!(py, "laboneq.dsl.experiment.section"))?
            .getattr(intern!(py, "Section"))?;
        let delay_py = py
            .import(intern!(py, "laboneq.dsl.experiment.delay"))?
            .getattr(intern!(py, "Delay"))?;
        let reserve_py = py
            .import(intern!(py, "laboneq.dsl.experiment.reserve"))?
            .getattr(intern!(py, "Reserve"))?;
        let acquire_py = py
            .import(intern!(py, "laboneq.dsl.experiment.acquire"))?
            .getattr(intern!(py, "Acquire"))?;
        let play_pulse_py = py
            .import(intern!(py, "laboneq.dsl.experiment.play_pulse"))?
            .getattr(intern!(py, "PlayPulse"))?;
        let match_py = py
            .import(intern!(py, "laboneq.dsl.experiment.section"))?
            .getattr(intern!(py, "Match"))?;
        let case_py = py
            .import(intern!(py, "laboneq.dsl.experiment.section"))?
            .getattr(intern!(py, "Case"))?;
        let pulse_functional_py = py
            .import(intern!(py, "laboneq.dsl.experiment.pulse"))?
            .getattr(intern!(py, "PulseFunctional"))?;
        let acquire_loop_rt_py = py
            .import(intern!(py, "laboneq.dsl.experiment.section"))?
            .getattr(intern!(py, "AcquireLoopRt"))?;
        let neartime_callback = py
            .import(intern!(py, "laboneq.dsl.experiment.call"))?
            .getattr(intern!(py, "Call"))?;
        let prng_setup_py = py
            .import(intern!(py, "laboneq.dsl.experiment.section"))?
            .getattr(intern!(py, "PRNGSetup"))?;
        let prng_loop_py = py
            .import(intern!(py, "laboneq.dsl.experiment.section"))?
            .getattr(intern!(py, "PRNGLoop"))?;
        let set_node_py = py
            .import(intern!(py, "laboneq.dsl.experiment.set_node"))?
            .getattr(intern!(py, "SetNode"))?;
        let reset_oscillator_phase_py = py
            .import(intern!(py, "laboneq.dsl.experiment.reset_oscillator_phase"))?
            .getattr(intern!(py, "ResetOscillatorPhase"))?;

        let type_map = HashMap::from([
            (DslType::LinearSweepParameter, linear_sweep_parameter_py),
            (DslType::SweepParameter, sweep_parameter_py),
            (DslType::Parameter, parameter_py),
            (DslType::Sweep, sweep_py),
            (DslType::Match, match_py),
            (DslType::Section, section_py),
            (DslType::Delay, delay_py),
            (DslType::Reserve, reserve_py),
            (DslType::Acquire, acquire_py),
            (DslType::PlayPulse, play_pulse_py),
            (DslType::Case, case_py),
            (DslType::PulseFunctional, pulse_functional_py),
            (DslType::AcquireLoopRt, acquire_loop_rt_py),
            (DslType::Call, neartime_callback),
            (DslType::PrngSetup, prng_setup_py),
            (DslType::PrngLoop, prng_loop_py),
            (DslType::SetNode, set_node_py),
            (DslType::ResetOscillatorPhase, reset_oscillator_phase_py),
        ]);

        Ok(Self { type_map })
    }

    pub(crate) fn laboneq_type(&self, dsl_type: DslType) -> &Bound<'a, PyAny> {
        self.type_map
            .get(&dsl_type)
            .unwrap_or_else(|| panic!("DSL type not found: {dsl_type:?}"))
    }
}

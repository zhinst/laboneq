// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

//! Shared Python conversion helpers used by the Cap'n Proto serializer/deserializer path.

use crate::error::Result;
use pyo3::intern;
use pyo3::prelude::*;

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
    linear_sweep_parameter: Bound<'a, PyAny>,
    sweep_parameter: Bound<'a, PyAny>,
    parameter: Bound<'a, PyAny>,
    sweep: Bound<'a, PyAny>,
    section: Bound<'a, PyAny>,
    delay: Bound<'a, PyAny>,
    reserve: Bound<'a, PyAny>,
    acquire: Bound<'a, PyAny>,
    play_pulse: Bound<'a, PyAny>,
    match_: Bound<'a, PyAny>,
    case: Bound<'a, PyAny>,
    pulse_functional: Bound<'a, PyAny>,
    acquire_loop_rt: Bound<'a, PyAny>,
    neartime_callback: Bound<'a, PyAny>,
    prng_setup: Bound<'a, PyAny>,
    prng_loop: Bound<'a, PyAny>,
    set_node: Bound<'a, PyAny>,
    reset_oscillator_phase: Bound<'a, PyAny>,
}

impl<'a> DslTypes<'a> {
    pub(crate) fn new(py: Python<'a>) -> Result<DslTypes<'a>> {
        let linear_sweep_parameter = py
            .import(intern!(py, "laboneq.dsl.parameter"))?
            .getattr(intern!(py, "LinearSweepParameter"))?;
        let sweep_parameter: Bound<'_, PyAny> = py
            .import(intern!(py, "laboneq.dsl.parameter"))?
            .getattr(intern!(py, "SweepParameter"))?;
        let parameter: Bound<'_, PyAny> = py
            .import(intern!(py, "laboneq.dsl.parameter"))?
            .getattr(intern!(py, "Parameter"))?;
        let sweep = py
            .import(intern!(py, "laboneq.dsl.experiment.section"))?
            .getattr(intern!(py, "Sweep"))?;
        let section = py
            .import(intern!(py, "laboneq.dsl.experiment.section"))?
            .getattr(intern!(py, "Section"))?;
        let delay = py
            .import(intern!(py, "laboneq.dsl.experiment.delay"))?
            .getattr(intern!(py, "Delay"))?;
        let reserve = py
            .import(intern!(py, "laboneq.dsl.experiment.reserve"))?
            .getattr(intern!(py, "Reserve"))?;
        let acquire = py
            .import(intern!(py, "laboneq.dsl.experiment.acquire"))?
            .getattr(intern!(py, "Acquire"))?;
        let play_pulse = py
            .import(intern!(py, "laboneq.dsl.experiment.play_pulse"))?
            .getattr(intern!(py, "PlayPulse"))?;
        let match_ = py
            .import(intern!(py, "laboneq.dsl.experiment.section"))?
            .getattr(intern!(py, "Match"))?;
        let case = py
            .import(intern!(py, "laboneq.dsl.experiment.section"))?
            .getattr(intern!(py, "Case"))?;
        let pulse_functional = py
            .import(intern!(py, "laboneq.dsl.experiment.pulse"))?
            .getattr(intern!(py, "PulseFunctional"))?;
        let acquire_loop_rt = py
            .import(intern!(py, "laboneq.dsl.experiment.section"))?
            .getattr(intern!(py, "AcquireLoopRt"))?;
        let neartime_callback = py
            .import(intern!(py, "laboneq.dsl.experiment.call"))?
            .getattr(intern!(py, "Call"))?;
        let prng_setup = py
            .import(intern!(py, "laboneq.dsl.experiment.section"))?
            .getattr(intern!(py, "PRNGSetup"))?;
        let prng_loop = py
            .import(intern!(py, "laboneq.dsl.experiment.section"))?
            .getattr(intern!(py, "PRNGLoop"))?;
        let set_node = py
            .import(intern!(py, "laboneq.dsl.experiment.set_node"))?
            .getattr(intern!(py, "SetNode"))?;
        let reset_oscillator_phase = py
            .import(intern!(py, "laboneq.dsl.experiment.reset_oscillator_phase"))?
            .getattr(intern!(py, "ResetOscillatorPhase"))?;

        Ok(Self {
            linear_sweep_parameter,
            sweep_parameter,
            parameter,
            sweep,
            match_,
            section,
            delay,
            reserve,
            acquire,
            play_pulse,
            case,
            pulse_functional,
            acquire_loop_rt,
            neartime_callback,
            prng_setup,
            prng_loop,
            set_node,
            reset_oscillator_phase,
        })
    }

    pub(crate) fn laboneq_type(&self, dsl_type: DslType) -> &Bound<'a, PyAny> {
        match dsl_type {
            DslType::LinearSweepParameter => &self.linear_sweep_parameter,
            DslType::SweepParameter => &self.sweep_parameter,
            DslType::Parameter => &self.parameter,
            DslType::Sweep => &self.sweep,
            DslType::Section => &self.section,
            DslType::Delay => &self.delay,
            DslType::Reserve => &self.reserve,
            DslType::Acquire => &self.acquire,
            DslType::PlayPulse => &self.play_pulse,
            DslType::Match => &self.match_,
            DslType::Case => &self.case,
            DslType::PulseFunctional => &self.pulse_functional,
            DslType::AcquireLoopRt => &self.acquire_loop_rt,
            DslType::Call => &self.neartime_callback,
            DslType::PrngSetup => &self.prng_setup,
            DslType::PrngLoop => &self.prng_loop,
            DslType::SetNode => &self.set_node,
            DslType::ResetOscillatorPhase => &self.reset_oscillator_phase,
        }
    }
}

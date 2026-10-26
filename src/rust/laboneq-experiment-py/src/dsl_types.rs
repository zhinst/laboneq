// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

//! Python DSL classes, imported once per serialization.

use pyo3::intern;
use pyo3::prelude::*;

use crate::Result;

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
    HqcsDoUntilSection,
    HqcsSend,
    HqcsMarkStale,
    HqcsEnsure,
    HqcsVariable,
    HqcsPredicate,
    HqcsIsLive,
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
    hqcs_do_until: Bound<'a, PyAny>,
    hqcs_send: Bound<'a, PyAny>,
    hqcs_mark_stale: Bound<'a, PyAny>,
    hqcs_ensure: Bound<'a, PyAny>,
    hqcs_variable: Bound<'a, PyAny>,
    hqcs_predicate: Bound<'a, PyAny>,
    hqcs_is_live: Bound<'a, PyAny>,
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
        let hqcs_do_until = py
            .import(intern!(py, "laboneq.dsl.experiment.do_until"))?
            .getattr(intern!(py, "DoUntilSection"))?;
        let hqcs_variable = py
            .import(intern!(py, "laboneq.dsl.variable"))?
            .getattr(intern!(py, "Variable"))?;
        let hqcs_send = py
            .import(intern!(py, "laboneq.dsl.coprocessor.operations"))?
            .getattr(intern!(py, "_Send"))?;
        let hqcs_mark_stale = py
            .import(intern!(py, "laboneq.dsl.coprocessor.operations"))?
            .getattr(intern!(py, "_MarkStale"))?;
        let hqcs_ensure = py
            .import(intern!(py, "laboneq.dsl.coprocessor.operations"))?
            .getattr(intern!(py, "_Ensure"))?;
        let hqcs_predicate = py
            .import(intern!(py, "laboneq.dsl.coprocessor.predicate"))?
            .getattr(intern!(py, "_Predicate"))?;
        let hqcs_is_live = py
            .import(intern!(py, "laboneq.dsl.coprocessor.predicate"))?
            .getattr(intern!(py, "_IsLive"))?;

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
            hqcs_do_until,
            hqcs_send,
            hqcs_mark_stale,
            hqcs_ensure,
            hqcs_variable,
            hqcs_predicate,
            hqcs_is_live,
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
            DslType::HqcsDoUntilSection => &self.hqcs_do_until,
            DslType::HqcsSend => &self.hqcs_send,
            DslType::HqcsMarkStale => &self.hqcs_mark_stale,
            DslType::HqcsEnsure => &self.hqcs_ensure,
            DslType::HqcsVariable => &self.hqcs_variable,
            DslType::HqcsPredicate => &self.hqcs_predicate,
            DslType::HqcsIsLive => &self.hqcs_is_live,
        }
    }
}

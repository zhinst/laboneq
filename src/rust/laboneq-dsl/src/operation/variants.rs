// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::num::NonZeroU32;

use crate::types::{
    AcquisitionType, AveragingMode, ComplexOrFloat, ExternalParameterUid, HandleUid, Marker,
    MatchTarget, NumericLiteral, ParameterUid, PrngSampleUid, PulseParameterUid, PulseUid,
    RepetitionMode, SectionAlignment, SectionUid, SignalUid, Trigger, ValueOrParameter,
};
use laboneq_units::duration::{Duration, Second};

impl Reserve {
    pub fn new(signal: SignalUid) -> Self {
        Self { signal }
    }
}

impl ResetOscillatorPhase {
    pub fn new(signals: Vec<SignalUid>) -> Self {
        Self { signals }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Delay {
    pub signal: SignalUid,
    pub time: ValueOrParameter<Duration<Second>>,
    pub precompensation_clear: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum PulseParameterValue {
    // External parameter UID points to an arbitrary value
    // resolved at sampling time, in case of the pulse being played
    // is a Python function.
    ExternalParameter(ExternalParameterUid),
    ValueOrParameter(ValueOrParameter<NumericLiteral>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct PlayPulse {
    pub signal: SignalUid,
    pub pulse: Option<PulseUid>,
    pub amplitude: ValueOrParameter<ComplexOrFloat>,
    pub phase: Option<ValueOrParameter<f64>>,
    pub increment_oscillator_phase: Option<ValueOrParameter<f64>>,
    pub set_oscillator_phase: Option<ValueOrParameter<f64>>,
    pub length: Option<ValueOrParameter<Duration<Second>>>,
    pub parameters: HashMap<PulseParameterUid, PulseParameterValue>,
    pub pulse_parameters: HashMap<PulseParameterUid, PulseParameterValue>,
    pub markers: Vec<Marker>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Acquire {
    pub signal: SignalUid,
    pub handle: HandleUid,
    pub length: Option<Duration<Second, f64>>,
    pub kernel: Vec<PulseUid>,
    pub parameters: Vec<HashMap<PulseParameterUid, PulseParameterValue>>,
    pub pulse_parameters: Vec<HashMap<PulseParameterUid, PulseParameterValue>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Section {
    pub uid: SectionUid,
    pub alignment: SectionAlignment,
    pub length: Option<Duration<Second, f64>>,
    pub play_after: Vec<SectionUid>,
    pub triggers: Vec<Trigger>,
    pub on_system_grid: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PrngSetup {
    pub uid: SectionUid,
    pub range: u32,
    pub seed: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PrngLoop {
    pub uid: SectionUid,
    pub count: NonZeroU32,
    pub sample_uid: PrngSampleUid,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Reserve {
    pub signal: SignalUid,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ResetOscillatorPhase {
    /// If empty, will reset all oscillators within a Section
    pub signals: Vec<SignalUid>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Chunking {
    Count { count: NonZeroU32 },
    Auto,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Sweep {
    pub uid: SectionUid,
    pub parameters: Vec<ParameterUid>,
    pub count: NonZeroU32,
    pub alignment: SectionAlignment,
    pub reset_oscillator_phase: bool,
    pub chunking: Option<Chunking>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Case {
    pub uid: SectionUid,
    pub state: NumericLiteral,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Match {
    pub uid: SectionUid,
    pub target: MatchTarget,
    pub local: Option<bool>,
    pub play_after: Vec<SectionUid>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AveragingLoop {
    pub uid: SectionUid,
    pub count: NonZeroU32,
    pub acquisition_type: AcquisitionType,
    pub averaging_mode: AveragingMode,
    pub repetition_mode: RepetitionMode,
    pub reset_oscillator_phase: bool,
    pub alignment: SectionAlignment,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Operation {
    Root,
    Section(Section),
    PrngSetup(PrngSetup),
    PrngLoop(PrngLoop),
    Reserve(Reserve),
    Sweep(Sweep),
    PlayPulse(PlayPulse),
    Acquire(Acquire),
    Delay(Delay),
    AveragingLoop(AveragingLoop),
    RealTimeBoundary,
    Match(Match),
    ResetOscillatorPhase(ResetOscillatorPhase),
    Case(Case),
    /// Near-time callback is an external function call that is executed between near-time steps.
    NearTimeCallback,
    SetNode,
}

pub struct SectionInfo<'a> {
    pub uid: &'a SectionUid,
}

pub struct LoopInfo<'a> {
    pub uid: &'a SectionUid,
    pub count: NonZeroU32,
    pub parameters: &'a [ParameterUid],
    pub reset_oscillator_phase: bool,
    pub alignment: &'a SectionAlignment,
    pub repetition_mode: Option<RepetitionMode>,
}

impl Operation {
    pub fn signals(&self) -> Vec<SignalUid> {
        match self {
            Operation::PlayPulse(p) => vec![p.signal],
            Operation::Acquire(a) => vec![a.signal],
            Operation::Delay(d) => vec![d.signal],
            Operation::ResetOscillatorPhase(r) => r.signals.to_vec(),
            Operation::Reserve(r) => vec![r.signal],
            Operation::Section(obj) => obj.triggers.iter().map(|t| t.signal).collect(),
            _ => vec![],
        }
    }

    pub fn section_info<'a>(self: &'a Operation) -> Option<SectionInfo<'a>> {
        match self {
            Operation::Section(s) => SectionInfo { uid: &s.uid }.into(),
            Operation::PrngSetup(s) => SectionInfo { uid: &s.uid }.into(),
            Operation::PrngLoop(s) => SectionInfo { uid: &s.uid }.into(),
            Operation::Sweep(s) => SectionInfo { uid: &s.uid }.into(),
            Operation::AveragingLoop(s) => SectionInfo { uid: &s.uid }.into(),
            Operation::Match(s) => SectionInfo { uid: &s.uid }.into(),
            Operation::Case(s) => SectionInfo { uid: &s.uid }.into(),
            Operation::Root
            | Operation::Reserve(_)
            | Operation::PlayPulse(_)
            | Operation::Acquire(_)
            | Operation::Delay(_)
            | Operation::ResetOscillatorPhase(_)
            | Operation::RealTimeBoundary
            | Operation::NearTimeCallback
            | Operation::SetNode => None,
        }
    }

    pub fn loop_info<'a>(self: &'a Operation) -> Option<LoopInfo<'a>> {
        match self {
            Operation::PrngLoop(s) => LoopInfo {
                uid: &s.uid,
                count: s.count,
                parameters: &[],
                reset_oscillator_phase: false,
                alignment: &SectionAlignment::Left,
                repetition_mode: None,
            }
            .into(),
            Operation::Sweep(s) => LoopInfo {
                uid: &s.uid,
                count: s.count,
                parameters: &s.parameters,
                reset_oscillator_phase: s.reset_oscillator_phase,
                alignment: &s.alignment,
                repetition_mode: None,
            }
            .into(),
            Operation::AveragingLoop(s) => LoopInfo {
                uid: &s.uid,
                count: s.count,
                parameters: &[],
                reset_oscillator_phase: s.reset_oscillator_phase,
                alignment: &s.alignment,
                repetition_mode: Some(s.repetition_mode),
            }
            .into(),
            Operation::Root
            | Operation::Section(_)
            | Operation::PrngSetup(_)
            | Operation::Match(_)
            | Operation::Case(_)
            | Operation::Reserve(_)
            | Operation::PlayPulse(_)
            | Operation::Acquire(_)
            | Operation::Delay(_)
            | Operation::ResetOscillatorPhase(_)
            | Operation::RealTimeBoundary
            | Operation::NearTimeCallback
            | Operation::SetNode => None,
        }
    }

    /// Validate if the operation is compatible with real-time execution.
    pub fn validate_real_time_compatible(&self) -> Result<(), &'static str> {
        match self {
            Operation::SetNode => Err(
                "'Set node' is a near-time operation and cannot be part of real-time execution.",
            ),
            Operation::NearTimeCallback => Err(
                "Near-time callback is a near-time operation and cannot be part of real-time execution.",
            ),
            Operation::Root
            | Operation::AveragingLoop(_)
            | Operation::Sweep(_)
            | Operation::PrngLoop(_)
            | Operation::Section(_)
            | Operation::PrngSetup(_)
            | Operation::Match(_)
            | Operation::Case(_)
            | Operation::Reserve(_)
            | Operation::PlayPulse(_)
            | Operation::Acquire(_)
            | Operation::Delay(_)
            | Operation::ResetOscillatorPhase(_)
            | Operation::RealTimeBoundary => Ok(()),
        }
    }
}

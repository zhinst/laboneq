// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::num::NonZeroU32;

use crate::types::{
    AcquisitionType, AveragingMode, ComplexOrFloat, ExternalParameterUid, HandleUid, Marker,
    MatchTarget, NumericLiteral, ParameterUid, PrngSampleUid, PulseParameterUid, PulseUid,
    RepetitionMode, SectionAlignment, SectionTimingMode, SectionUid, SignalUid, Trigger,
    ValueOrParameter,
};
use laboneq_common::named_id::NamedId;
use laboneq_common::types::Literal;
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

/// Either a constant/parametric value of type [`NumericLiteral`], or an external parameter.
///
/// The values can be parametrized, therefore the minimum requirement for the value type is [`NumericLiteral`], which represents
/// all the possible parametrized types (those of sweep parameters).
#[derive(Debug, Clone, PartialEq)]
pub enum ExternalOrValue {
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
    pub parameters: HashMap<PulseParameterUid, ExternalOrValue>,
    pub pulse_parameters: HashMap<PulseParameterUid, ExternalOrValue>,
    pub markers: Vec<Marker>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Acquire {
    pub signal: SignalUid,
    pub handle: HandleUid,
    pub length: Option<Duration<Second, f64>>,
    pub kernel: Vec<PulseUid>,
    pub parameters: Vec<HashMap<PulseParameterUid, ExternalOrValue>>,
    pub pulse_parameters: Vec<HashMap<PulseParameterUid, ExternalOrValue>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Section {
    pub uid: SectionUid,
    pub alignment: SectionAlignment,
    pub length: Option<Duration<Second, f64>>,
    pub play_after: Vec<SectionUid>,
    pub triggers: Vec<Trigger>,
    pub on_system_grid: bool,
    pub section_timing_mode: SectionTimingMode,
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
pub struct Sweep {
    pub uid: SectionUid,
    /// List of all parameters that are swept over in this loop.
    /// Contains both directly defined parameters and derived parameters.
    pub parameters: Vec<ParameterUid>,
    /// Parameters that are directly defined in the sweep.
    /// Subset of `parameters`.
    pub direct_parameters: Vec<ParameterUid>,
    pub count: NonZeroU32,
    pub alignment: SectionAlignment,
    pub reset_oscillator_phase: bool,
    pub chunk_count: NonZeroU32,
    pub auto_chunking: bool,
    pub section_timing_mode: SectionTimingMode,
}

impl Sweep {
    /// A sweep is considered chunked if either auto-chunking is enabled or a chunk count greater than 1 is specified.
    pub fn is_chunked(&self) -> bool {
        self.auto_chunking || self.chunk_count.get() > 1
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Case {
    pub uid: SectionUid,
    pub state: NumericLiteral,
    pub section_timing_mode: SectionTimingMode,
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
    pub section_timing_mode: SectionTimingMode,
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
    NearTimeCallback(NearTimeCallback),
    SetNode(SetNode),
}

#[derive(Debug, Clone, PartialEq)]
pub struct NearTimeCallback {
    pub callback_id: NamedId,
    pub args: Vec<ValueEntry>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ValueEntry {
    pub key: NamedId,
    pub value: ExternalOrValue,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SetNode {
    pub path: NamedId,
    pub value: ValueOrParameter<Literal>,
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
    pub section_timing_mode: &'a SectionTimingMode,
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
            | Operation::NearTimeCallback(_)
            | Operation::SetNode(_) => None,
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
                section_timing_mode: &SectionTimingMode::Relaxed,
            }
            .into(),
            Operation::Sweep(s) => LoopInfo {
                uid: &s.uid,
                count: s.count,
                parameters: &s.parameters,
                reset_oscillator_phase: s.reset_oscillator_phase,
                alignment: &s.alignment,
                repetition_mode: None,
                section_timing_mode: &s.section_timing_mode,
            }
            .into(),
            Operation::AveragingLoop(s) => LoopInfo {
                uid: &s.uid,
                count: s.count,
                parameters: &[],
                reset_oscillator_phase: s.reset_oscillator_phase,
                alignment: &s.alignment,
                repetition_mode: Some(s.repetition_mode),
                section_timing_mode: &s.section_timing_mode,
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
            | Operation::NearTimeCallback(_)
            | Operation::SetNode(_) => None,
        }
    }

    /// Validate if the operation is compatible with real-time execution.
    pub fn validate_real_time_compatible(&self) -> Result<(), &'static str> {
        match self {
            Operation::SetNode(_) => Err(
                "'Set node' is a near-time operation and cannot be part of real-time execution.",
            ),
            Operation::NearTimeCallback(_) => Err(
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

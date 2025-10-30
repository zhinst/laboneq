// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use laboneq_common::named_id::NamedId;
use laboneq_units::duration::{Duration, Seconds};
use num_complex::Complex64;
use std::{collections::HashMap, sync::Arc};

use crate::error;

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct SectionUid(pub NamedId);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PulseUid(pub NamedId);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OscillatorUid(pub NamedId);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ParameterUid(pub NamedId);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HandleUid(pub NamedId);

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct DeviceUid(pub NamedId);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SignalUid(pub NamedId);

#[macro_export]
macro_rules! impl_from_named_id {
    ($t:ty) => {
        impl From<laboneq_common::named_id::NamedId> for $t {
            fn from(value: laboneq_common::named_id::NamedId) -> Self {
                Self(value)
            }
        }

        impl From<$t> for laboneq_common::named_id::NamedId {
            fn from(value: $t) -> Self {
                value.0
            }
        }
    };
}

impl_from_named_id!(SignalUid);
impl_from_named_id!(OscillatorUid);
impl_from_named_id!(ParameterUid);
impl_from_named_id!(HandleUid);
impl_from_named_id!(DeviceUid);
impl_from_named_id!(PulseUid);
impl_from_named_id!(SectionUid);

pub type UserRegister = u16;

/// UID of an external parameter.
///
/// This is used to uniquely identify external parameters in the experiment.
/// The parameter the UID refers to is not accessed by the compiler itself.
#[derive(Debug, Hash, Eq, PartialEq, Clone, Copy)]
pub struct ExternalParameterUid(pub u64);

// Common type definitions

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Float(f64),
    Int(i64),
    Complex(Complex64),
    Bool(bool),
    String(String),
    ParameterUid(ParameterUid),
}

#[derive(Debug, Clone, PartialEq, Copy)]
pub enum NumericLiteral {
    Float(f64),
    Int(i64),
    Complex(Complex64),
}

impl TryFrom<Value> for NumericLiteral {
    type Error = &'static str;

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        match value {
            Value::Float(v) => Ok(NumericLiteral::Float(v)),
            Value::Int(v) => Ok(NumericLiteral::Int(v)),
            Value::Complex(v) => Ok(NumericLiteral::Complex(v)),
            _ => Err("Value is not numeric literal"),
        }
    }
}

impl TryInto<f64> for NumericLiteral {
    type Error = &'static str;
    fn try_into(self) -> Result<f64, Self::Error> {
        match self {
            NumericLiteral::Float(v) => Ok(v),
            NumericLiteral::Int(v) => Ok(v as f64),
            NumericLiteral::Complex(_) => Err("Cannot convert complex to f64"),
        }
    }
}

impl TryInto<Complex64> for NumericLiteral {
    type Error = &'static str;

    fn try_into(self) -> Result<Complex64, Self::Error> {
        match self {
            NumericLiteral::Float(v) => Ok(Complex64::new(v, 0.0)),
            NumericLiteral::Int(v) => Ok(Complex64::new(v as f64, 0.0)),
            NumericLiteral::Complex(v) => Ok(v),
        }
    }
}

impl TryInto<RealValue> for NumericLiteral {
    type Error = &'static str;

    fn try_into(self) -> Result<RealValue, Self::Error> {
        match self {
            NumericLiteral::Float(v) => Ok(RealValue::Float(v)),
            NumericLiteral::Int(v) => Ok(RealValue::Int(v)),
            _ => Err("Value is not real value"),
        }
    }
}

impl TryInto<f64> for RealValue {
    type Error = &'static str;

    fn try_into(self) -> Result<f64, Self::Error> {
        match self {
            RealValue::Float(v) => Ok(v),
            RealValue::Int(v) => Ok(v as f64),
            _ => Err("Value is not a float"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Copy)]
pub enum NumericValue {
    Float(f64),
    Int(i64),
    Complex(Complex64),
    ParameterUid(ParameterUid),
}

impl TryFrom<Value> for NumericValue {
    type Error = &'static str;

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        match value {
            Value::Float(v) => Ok(NumericValue::Float(v)),
            Value::Int(v) => Ok(NumericValue::Int(v)),
            Value::Complex(v) => Ok(NumericValue::Complex(v)),
            Value::ParameterUid(v) => Ok(NumericValue::ParameterUid(v)),
            _ => Err("Value is not numeric"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Copy)]
pub enum RealValue {
    Float(f64),
    Int(i64),
    ParameterUid(ParameterUid),
}

impl TryFrom<Value> for RealValue {
    type Error = &'static str;

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        match value {
            Value::Float(v) => Ok(RealValue::Float(v)),
            Value::Int(v) => Ok(RealValue::Int(v)),
            Value::ParameterUid(v) => Ok(RealValue::ParameterUid(v)),
            Value::Complex(v) => {
                if v.im == 0.0 {
                    Ok(RealValue::Float(v.re))
                } else {
                    Err("Value is not real numeric")
                }
            }
            _ => Err("Value is not real numeric"),
        }
    }
}

// Information living outside the tree

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PulseLength {
    Seconds(Duration<Seconds, f64>),
    Samples(usize),
}

#[derive(Debug, Clone, PartialEq)]
pub struct PulseRef {
    pub uid: PulseUid,
    pub length: PulseLength,
}

#[derive(Debug, Clone, PartialEq, Copy)]
pub enum OscillatorKind {
    Hardware,
    Software,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Oscillator {
    pub uid: OscillatorUid, // NOTE: Needed for legacy reasons, should be removed in future
    pub frequency: RealValue,
    pub kind: OscillatorKind,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExperimentSignal {
    pub uid: SignalUid,
}

// IR definition

#[derive(Debug, Clone, PartialEq, Copy)]
pub enum SectionAlignment {
    Left,
    Right,
}

#[derive(Debug, Clone, PartialEq, Copy)]
pub enum RepetitionMode {
    Fastest,
    Constant { time: f64 },
    Auto,
}

#[derive(Debug, Clone, PartialEq, Copy)]
pub enum AcquisitionType {
    Integration,
    SpectroscopyIq,
    SpectroscopyPsd,
    Spectroscopy,
    Discrimination,
    Raw,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Trigger {
    pub signal: SignalUid,
    pub state: u16,
}

#[derive(Debug, Clone, PartialEq, Copy)]
pub enum MarkerSelector {
    M1,
    M2,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Marker {
    pub marker_selector: MarkerSelector,
    pub enable: bool,
    pub start: Option<Duration<Seconds, f64>>,
    pub length: Option<Duration<Seconds, f64>>,
    pub pulse_id: Option<PulseUid>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Delay {
    pub signal: SignalUid,
    pub time: RealValue,
    pub precompensation_clear: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum PulseParameterValue {
    // External parameter UID points to an arbitrary value
    // resolved at sampling time, in case of the pulse being played
    // is a Python function.
    ExternalParameter(ExternalParameterUid),
    Parameter(ParameterUid),
}

#[derive(Debug, Clone, PartialEq)]
pub struct PlayPulse {
    pub signal: SignalUid,
    pub pulse: Option<PulseUid>,
    pub precompensation_clear: bool,
    pub amplitude: NumericValue,
    pub phase: Option<RealValue>,
    pub increment_oscillator_phase: Option<RealValue>,
    pub set_oscillator_phase: Option<RealValue>,
    pub length: Option<RealValue>,
    pub parameters: HashMap<Arc<String>, PulseParameterValue>,
    pub pulse_parameters: HashMap<Arc<String>, PulseParameterValue>,
    pub markers: Vec<Marker>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Acquire {
    pub signal: SignalUid,
    pub handle: HandleUid,
    pub length: Option<Duration<Seconds, f64>>,
    pub kernel: Vec<PulseUid>,
    pub parameters: Vec<HashMap<Arc<String>, PulseParameterValue>>,
    pub pulse_parameters: Vec<HashMap<Arc<String>, PulseParameterValue>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Section {
    pub uid: SectionUid,
    pub alignment: SectionAlignment,
    pub length: Option<Duration<Seconds, f64>>,
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
    pub count: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Reserve {
    pub signal: SignalUid,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ResetOscillatorPhase {
    /// If None, reset all oscillators within a Section
    pub signal: Option<SignalUid>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Chunking {
    Count { count: usize },
    Auto,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Sweep {
    pub uid: SectionUid,
    pub parameters: Vec<ParameterUid>,
    pub count: u32,
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
pub enum MatchTarget {
    Handle(HandleUid),
    UserRegister(UserRegister),
    /// PRNG Loop UID
    PrngSample(SectionUid),
    SweepParameter(ParameterUid),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Match {
    pub uid: SectionUid,
    pub target: MatchTarget,
    pub local: Option<bool>,
    pub play_after: Vec<SectionUid>,
}

#[derive(Debug, Clone, PartialEq, Copy)]
pub enum AveragingMode {
    Sequential,
    Cyclic,
    SingleShot,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AveragingLoop {
    pub uid: SectionUid,
    pub count: u32,
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
    pub count: u32,
    pub parameters: &'a [ParameterUid],
}

impl Operation {
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
            }
            .into(),
            Operation::Sweep(s) => LoopInfo {
                uid: &s.uid,
                count: s.count,
                parameters: &s.parameters,
            }
            .into(),
            Operation::AveragingLoop(s) => LoopInfo {
                uid: &s.uid,
                count: s.count,
                parameters: &[],
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
    pub fn validate_real_time_compatible(&self) -> error::Result<()> {
        match self {
            Operation::SetNode => Err(error::Error::new(
                "Set node is near-time operation and cannot be part of real-time execution.",
            )),
            Operation::NearTimeCallback => Err(error::Error::new(
                "Near-time callback is near-time operation and cannot be part of real-time execution.",
            )),
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

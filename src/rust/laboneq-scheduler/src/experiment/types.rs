// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use laboneq_common::named_id::NamedId;
use laboneq_units::duration::{Duration, Second};
use num_complex::Complex64;
use num_traits::ToPrimitive;
use std::collections::HashMap;

use crate::error;

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct SectionUid(pub NamedId);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct PulseUid(pub NamedId);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct OscillatorUid(pub NamedId);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct ParameterUid(pub NamedId);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct HandleUid(pub NamedId);

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct DeviceUid(pub NamedId);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct SignalUid(pub NamedId);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PulseParameterUid(pub NamedId);

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
impl_from_named_id!(PulseParameterUid);

pub type UserRegister = u16;

/// UID of an external parameter.
///
/// This is used to uniquely identify external parameters in the experiment.
/// The parameter the UID refers to is not accessed by the compiler itself.
#[derive(Debug, Hash, Eq, PartialEq, Clone, Copy)]
pub struct ExternalParameterUid(pub u64);

impl From<u64> for ExternalParameterUid {
    fn from(value: u64) -> Self {
        ExternalParameterUid(value)
    }
}

// Common type definitions

#[derive(Debug, Clone, PartialEq, Copy)]
pub enum ValueOrParameter<T> {
    Value(T),
    Parameter(ParameterUid),
    // Value resolved from parameter
    ResolvedParameter { value: T, uid: ParameterUid },
}

impl TryFrom<ValueOrParameter<f64>> for f64 {
    type Error = &'static str;

    fn try_from(value: ValueOrParameter<f64>) -> Result<Self, Self::Error> {
        match value {
            ValueOrParameter::Value(v) => Ok(v),
            ValueOrParameter::Parameter(_) => Err("Cannot convert Parameter to f64"),
            ValueOrParameter::ResolvedParameter { value, uid: _ } => Ok(value),
        }
    }
}

impl TryFrom<NumericLiteral> for ValueOrParameter<f64> {
    type Error = &'static str;

    fn try_from(value: NumericLiteral) -> Result<ValueOrParameter<f64>, Self::Error> {
        Ok(ValueOrParameter::Value(value.try_into()?))
    }
}

impl TryFrom<NumericLiteral> for ValueOrParameter<Complex64> {
    type Error = &'static str;

    fn try_from(value: NumericLiteral) -> Result<ValueOrParameter<Complex64>, Self::Error> {
        Ok(ValueOrParameter::Value(value.try_into()?))
    }
}

impl From<f64> for ValueOrParameter<f64> {
    fn from(value: f64) -> Self {
        ValueOrParameter::Value(value)
    }
}

#[derive(Debug, Clone, Copy)]
pub enum NumericLiteral {
    Float(f64),
    Int(i64),
    Complex(Complex64),
}

impl Eq for NumericLiteral {}

impl std::fmt::Display for NumericLiteral {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NumericLiteral::Float(v) => write!(f, "{}", v),
            NumericLiteral::Int(v) => write!(f, "{}", v),
            NumericLiteral::Complex(v) => write!(f, "{} + {}j", v.re, v.im),
        }
    }
}

impl TryFrom<NumericLiteral> for f64 {
    type Error = &'static str;
    fn try_from(value: NumericLiteral) -> Result<Self, Self::Error> {
        match value {
            NumericLiteral::Float(v) => Ok(v),
            NumericLiteral::Int(v) => v
                .to_f64()
                .ok_or("Integer value is too large to convert to f64"),
            NumericLiteral::Complex(v) => {
                if v.im == 0.0 {
                    Ok(v.re)
                } else {
                    Err("Cannot convert complex to f64")
                }
            }
        }
    }
}

impl From<f64> for NumericLiteral {
    fn from(value: f64) -> Self {
        NumericLiteral::Float(value)
    }
}

impl TryFrom<NumericLiteral> for usize {
    type Error = &'static str;
    fn try_from(value: NumericLiteral) -> Result<usize, Self::Error> {
        match value {
            NumericLiteral::Float(v) => {
                if v.fract() == 0.0 {
                    usize::try_from(v as i64)
                        .map_err(|_| "Cannot convert negative float to unsigned integer")
                } else {
                    Err("Cannot convert float to unsigned integer")
                }
            }
            NumericLiteral::Int(v) => usize::try_from(v)
                .map_err(|_| "Cannot convert negative integer to unsigned integer"),
            NumericLiteral::Complex(_) => Err("Cannot convert complex to unsigned integer"),
        }
    }
}

impl TryFrom<NumericLiteral> for Complex64 {
    type Error = &'static str;
    fn try_from(value: NumericLiteral) -> Result<Complex64, Self::Error> {
        match value {
            NumericLiteral::Float(v) => Ok(Complex64::new(v, 0.0)),
            NumericLiteral::Int(v) => Ok(Complex64::new(v as f64, 0.0)),
            NumericLiteral::Complex(v) => Ok(v),
        }
    }
}

impl PartialEq for NumericLiteral {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (NumericLiteral::Float(a), NumericLiteral::Float(b)) => a == b,
            (NumericLiteral::Int(a), NumericLiteral::Int(b)) => a == b,
            (NumericLiteral::Complex(a), NumericLiteral::Complex(b)) => a == b,
            (NumericLiteral::Float(a), NumericLiteral::Int(b)) => *a == *b as f64,
            (NumericLiteral::Int(a), NumericLiteral::Float(b)) => *a as f64 == *b,
            _ => false,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum ComplexOrFloat {
    Float(f64),
    Complex(Complex64),
}

impl Eq for ComplexOrFloat {}

impl std::fmt::Display for ComplexOrFloat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ComplexOrFloat::Float(v) => write!(f, "{}", v),
            ComplexOrFloat::Complex(v) => write!(f, "{} + {}j", v.re, v.im),
        }
    }
}

impl PartialEq for ComplexOrFloat {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (ComplexOrFloat::Float(a), ComplexOrFloat::Float(b)) => a == b,
            (ComplexOrFloat::Complex(a), ComplexOrFloat::Complex(b)) => a == b,
            _ => false,
        }
    }
}

impl TryFrom<NumericLiteral> for ComplexOrFloat {
    type Error = &'static str;
    fn try_from(value: NumericLiteral) -> Result<ComplexOrFloat, Self::Error> {
        match value {
            NumericLiteral::Float(v) => Ok(ComplexOrFloat::Float(v)),
            NumericLiteral::Complex(v) => Ok(ComplexOrFloat::Complex(v)),
            NumericLiteral::Int(_) => Err("Cannot convert integer to complex or float value"),
        }
    }
}

// Information living outside the tree

#[derive(Debug, Clone, PartialEq, Copy)]
pub enum OscillatorKind {
    Hardware,
    Software,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Oscillator {
    pub uid: OscillatorUid, // NOTE: Needed for legacy reasons, should be removed in future
    pub frequency: ValueOrParameter<f64>,
    pub kind: OscillatorKind,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AmplifierPump {
    pub device: DeviceUid,
    pub channel: u16,
    pub pump_power: Option<ValueOrParameter<f64>>,
    pub pump_frequency: Option<ValueOrParameter<f64>>,
    pub probe_power: Option<ValueOrParameter<f64>>,
    pub probe_frequency: Option<ValueOrParameter<f64>>,
    pub cancellation_phase: Option<ValueOrParameter<f64>>,
    pub cancellation_attenuation: Option<ValueOrParameter<f64>>,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Trigger {
    pub signal: SignalUid,
    pub state: u8,
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
    pub start: Option<Duration<Second, f64>>,
    pub length: Option<Duration<Second, f64>>,
    pub pulse_id: Option<PulseUid>,
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
    pub length: Option<ValueOrParameter<f64>>,
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
    pub count: u32,
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

impl MatchTarget {
    pub fn description(&self) -> &'static str {
        match self {
            MatchTarget::Handle(_) => "acquisition handle",
            MatchTarget::UserRegister(_) => "user register",
            MatchTarget::PrngSample(_) => "PRNG sample",
            MatchTarget::SweepParameter(_) => "sweep parameter",
        }
    }
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
    pub reset_oscillator_phase: bool,
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
            }
            .into(),
            Operation::Sweep(s) => LoopInfo {
                uid: &s.uid,
                count: s.count,
                parameters: &s.parameters,
                reset_oscillator_phase: s.reset_oscillator_phase,
            }
            .into(),
            Operation::AveragingLoop(s) => LoopInfo {
                uid: &s.uid,
                count: s.count,
                parameters: &[],
                reset_oscillator_phase: s.reset_oscillator_phase,
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

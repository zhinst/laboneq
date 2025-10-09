// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use laboneq_common::named_id::NamedId;
use laboneq_common::types::DeviceKind as DeviceKindCommon;
use laboneq_units::duration::{Duration, Seconds};
use num_complex::Complex64;
use numeric_array::NumericArray;
use std::{collections::HashMap, sync::Arc};

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

#[derive(Debug, Clone, PartialEq)]
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

#[derive(Debug, Clone, PartialEq)]
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

#[derive(Debug, Clone, PartialEq)]
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

#[derive(Debug, Clone, PartialEq)]
pub enum DeviceKind {
    Hdawg,
    Shfqa,
    Shfsg,
    Uhfqa,
    PrettyPrinterDevice,
}

impl From<DeviceKindCommon> for DeviceKind {
    fn from(kind: DeviceKindCommon) -> Self {
        match kind {
            DeviceKindCommon::Hdawg => DeviceKind::Hdawg,
            DeviceKindCommon::Uhfqa => DeviceKind::Uhfqa,
            DeviceKindCommon::Shfsg => DeviceKind::Shfsg,
            DeviceKindCommon::Shfqa => DeviceKind::Shfqa,
            DeviceKindCommon::PrettyPrinterDevice => DeviceKind::PrettyPrinterDevice,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Device {
    pub uid: DeviceUid,
    pub kind: DeviceKind,
}

#[derive(Debug, Clone, PartialEq)]
pub enum OscillatorKind {
    Hardware,
    Software,
}

#[derive(Debug, Clone)]
pub struct Oscillator {
    pub uid: OscillatorUid,
    pub frequency: Option<RealValue>,
    pub kind: OscillatorKind,
}

#[derive(Debug, Clone)]
pub struct ExperimentSignal {
    pub uid: SignalUid,
}

#[derive(Debug, Clone)]
pub enum ParameterKind {
    Linear {
        start: NumericLiteral,
        stop: NumericLiteral,
        count: usize,
    },
    Array {
        values: NumericArray,
    },
}

#[derive(Debug, Clone)]
pub struct Parameter {
    pub uid: ParameterUid,
    pub kind: ParameterKind,
}

// IR definition

#[derive(Debug, Clone)]
pub enum SectionAlignment {
    Left,
    Right,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ExecutionType {
    RealTime,
    NearTime,
}

#[derive(Debug, Clone)]
pub enum RepetitionMode {
    Fastest,
    Constant { time: f64 },
    Auto,
}

#[derive(Debug, Clone)]
pub enum AcquisitionType {
    Integration,
    SpectroscopyIq,
    SpectroscopyPsd,
    Spectroscopy,
    Discrimination,
    Raw,
}

#[derive(Debug, Clone)]
pub struct Trigger {
    pub signal: SignalUid,
    pub state: u16,
}

#[derive(Debug, Clone)]
pub enum MarkerSelector {
    M1,
    M2,
}

#[derive(Debug, Clone)]
pub struct Marker {
    pub marker_selector: MarkerSelector,
    pub enable: bool,
    pub start: Option<Duration<Seconds, f64>>,
    pub length: Option<Duration<Seconds, f64>>,
    pub pulse_id: Option<PulseUid>,
}

#[derive(Debug, Clone)]
pub struct Delay {
    pub signal: SignalUid,
    pub time: RealValue,
    pub precompensation_clear: bool,
}

#[derive(Debug, Clone)]
pub enum PulseParameterValue {
    // External parameter UID points to an arbitrary value
    // resolved at sampling time, in case of the pulse being played
    // is a Python function.
    ExternalParameter(ExternalParameterUid),
    Parameter(ParameterUid),
}

#[derive(Debug, Clone)]
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

#[derive(Debug, Clone)]
pub struct Acquire {
    pub signal: SignalUid,
    pub handle: HandleUid,
    pub length: Option<Duration<Seconds, f64>>,
    pub kernel: Vec<PulseUid>,
    pub parameters: Vec<HashMap<Arc<String>, PulseParameterValue>>,
    pub pulse_parameters: Vec<HashMap<Arc<String>, PulseParameterValue>>,
}

#[derive(Debug, Clone)]
pub struct Section {
    pub uid: SectionUid,
    pub alignment: SectionAlignment,
    pub length: Option<Duration<Seconds, f64>>,
    pub play_after: Vec<SectionUid>,
    pub triggers: Vec<Trigger>,
    pub on_system_grid: bool,
}

#[derive(Debug, Clone)]
pub struct Prng {
    pub range: u32,
    pub seed: u32,
}

#[derive(Debug, Clone)]
pub struct PrngSetup {
    pub uid: SectionUid,
    pub prng: Prng,
}

#[derive(Debug, Clone)]
pub struct PrngLoop {
    pub uid: SectionUid,
    pub prng: Prng,
    pub count: u32,
}

#[derive(Debug, Clone)]
pub struct Reserve {
    pub signal: SignalUid,
}

#[derive(Debug, Clone)]
pub struct ResetOscillatorPhase {
    pub signal: Option<SignalUid>,
}

#[derive(Debug, Clone)]
pub struct Sweep {
    pub uid: SectionUid,
    pub parameters: Vec<ParameterUid>,
    pub alignment: SectionAlignment,
    pub reset_oscillator_phase: bool,
    pub execution_type: ExecutionType,
}

#[derive(Debug, Clone)]
pub struct Case {
    pub uid: SectionUid,
    pub state: u16,
}

#[derive(Debug, Clone)]
pub enum MatchTarget {
    Handle(HandleUid),
    UserRegister(UserRegister),
    /// PRNG Loop UID
    PrngSample(SectionUid),
    SweepParameter(ParameterUid),
}

#[derive(Debug, Clone)]
pub struct Match {
    pub uid: SectionUid,
    pub target: MatchTarget,
    pub local: Option<bool>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum AveragingMode {
    Sequential,
    Cyclic,
    SingleShot,
}

#[derive(Debug, Clone)]
pub struct AcquireLoopRt {
    pub uid: SectionUid,
    pub count: u32,
    pub acquisition_type: AcquisitionType,
    pub averaging_mode: AveragingMode,
    pub repetition_mode: RepetitionMode,
    pub reset_oscillator_phase: bool,
}

#[derive(Debug, Clone)]
pub enum IrVariant {
    Root,
    Section(Section),
    PrngSetup(PrngSetup),
    PrngLoop(PrngLoop),
    Reserve(Reserve),
    Sweep(Sweep),
    PlayPulse(PlayPulse),
    Acquire(Acquire),
    Delay(Delay),
    AcquireLoopRt(AcquireLoopRt),
    Match(Match),
    ResetOscillatorPhase(ResetOscillatorPhase),
    Case(Case),
    NotYetImplemented,
}

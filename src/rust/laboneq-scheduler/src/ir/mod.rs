// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

//! Intermediate representation (IR) of the real-time portion of an experiment structure.

use std::collections::{HashMap, HashSet};

use laboneq_dsl::{
    operation::PulseParameterValue,
    types::{
        AveragingMode, ComplexOrFloat, DeviceUid, HandleUid, Marker, ParameterUid, PrngSampleUid,
        PulseParameterUid, PulseUid, SectionUid, SignalUid, ValueOrParameter,
    },
};
use laboneq_units::tinysample::TinySamples;

// Re-export for convenience
pub use laboneq_dsl::types::MatchTarget;

pub(crate) mod builders;

#[derive(Debug, Clone, PartialEq)]
pub enum IrKind {
    Root,
    InitialOscillatorFrequency(InitialOscillatorFrequency),
    InitialLocalOscillatorFrequency(InitialLocalOscillatorFrequency),
    InitialVoltageOffset(InitialVoltageOffset),
    Loop(Loop),
    LoopIterationPreamble,
    LoopIteration,
    PlayPulse(PlayPulse),
    ChangeOscillatorPhase(ChangeOscillatorPhase),
    Acquire(Acquire),
    Section(Section),
    SetOscillatorFrequency(SetOscillatorFrequency),
    /// Reset the phase of the specified oscillator signals.
    ResetOscillatorPhase {
        signals: Vec<SignalUid>,
    },
    PpcStep(PpcStep),
    Match(Match),
    Case(Case),
    Delay {
        signal: SignalUid,
    },
    ClearPrecompensation {
        signal: SignalUid,
    },
}

pub struct SectionInfo<'a> {
    pub uid: &'a SectionUid,
}

impl IrKind {
    pub fn section_info<'a>(self: &'a IrKind) -> Option<SectionInfo<'a>> {
        match self {
            IrKind::Loop(obj) => Some(SectionInfo { uid: &obj.uid }),
            IrKind::Section(obj) => Some(SectionInfo { uid: &obj.uid }),
            IrKind::Match(obj) => Some(SectionInfo { uid: &obj.uid }),
            IrKind::Case(obj) => Some(SectionInfo { uid: &obj.uid }),
            IrKind::Root => None,
            IrKind::InitialOscillatorFrequency(_) => None,
            IrKind::InitialLocalOscillatorFrequency(_) => None,
            IrKind::InitialVoltageOffset(_) => None,
            IrKind::LoopIterationPreamble => None,
            IrKind::LoopIteration => None,
            IrKind::PlayPulse(_) => None,
            IrKind::Acquire(_) => None,
            IrKind::SetOscillatorFrequency(_) => None,
            IrKind::ResetOscillatorPhase { .. } => None,
            IrKind::PpcStep(_) => None,
            IrKind::Delay { .. } => None,
            IrKind::ClearPrecompensation { .. } => None,
            IrKind::ChangeOscillatorPhase(_) => None,
        }
    }

    pub fn signals(self: &IrKind) -> HashSet<&SignalUid> {
        match self {
            IrKind::Section(obj) => obj.triggers.iter().map(|trig| &trig.signal).collect(),
            IrKind::InitialOscillatorFrequency(obj) => {
                obj.values.iter().map(|(sig, _)| sig).collect()
            }
            IrKind::InitialLocalOscillatorFrequency(obj) => HashSet::from_iter([&obj.signal]),
            IrKind::InitialVoltageOffset(obj) => HashSet::from_iter([&obj.signal]),
            IrKind::PlayPulse(obj) => HashSet::from_iter([&obj.signal]),
            IrKind::Acquire(obj) => HashSet::from_iter([&obj.signal]),
            IrKind::SetOscillatorFrequency(obj) => obj.values.iter().map(|(sig, _)| sig).collect(),
            IrKind::ResetOscillatorPhase { signals } => HashSet::from_iter(signals.iter()),
            IrKind::PpcStep(obj) => HashSet::from_iter([&obj.signal]),
            IrKind::Delay { signal } => HashSet::from_iter([signal]),
            IrKind::ClearPrecompensation { signal } => HashSet::from_iter([signal]),
            IrKind::ChangeOscillatorPhase(obj) => HashSet::from_iter([&obj.signal]),
            IrKind::Loop(_)
            | IrKind::LoopIterationPreamble
            | IrKind::LoopIteration
            | IrKind::Root
            | IrKind::Match(_)
            | IrKind::Case(_) => HashSet::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct InitialOscillatorFrequency {
    pub values: Vec<(SignalUid, ValueOrParameter<f64>)>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct InitialLocalOscillatorFrequency {
    pub signal: SignalUid,
    pub value: ValueOrParameter<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct InitialVoltageOffset {
    pub signal: SignalUid,
    pub value: ValueOrParameter<f64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoopKind {
    Averaging { mode: AveragingMode },
    Sweeping { parameters: Vec<ParameterUid> },
    Prng { sample_uid: PrngSampleUid },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Loop {
    pub uid: SectionUid,
    pub iterations: usize,
    pub kind: LoopKind,
}

impl Loop {
    /// Whether or not the loop is compressed.
    ///
    /// A loop is considered compressed if it has no sweep parameters
    /// and more than one iteration.
    pub fn compressed(&self) -> bool {
        match &self.kind {
            LoopKind::Averaging { .. } => self.iterations > 1,
            LoopKind::Prng { .. } => self.iterations > 1,
            LoopKind::Sweeping { parameters } => parameters.is_empty() && self.iterations > 1,
        }
    }

    pub fn parameters(&self) -> &[ParameterUid] {
        static EMPTY_VEC: &[ParameterUid] = &[];
        match &self.kind {
            LoopKind::Sweeping { parameters } => parameters,
            _ => EMPTY_VEC,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SetOscillatorFrequency {
    pub values: Vec<(SignalUid, ValueOrParameter<f64>)>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PlayPulse {
    pub signal: SignalUid,
    pub pulse: PulseUid,
    pub amplitude: ValueOrParameter<ComplexOrFloat>,
    pub phase: Option<ValueOrParameter<f64>>,
    pub increment_oscillator_phase: Option<ValueOrParameter<f64>>,
    pub set_oscillator_phase: Option<ValueOrParameter<f64>>,
    pub parameters: HashMap<PulseParameterUid, PulseParameterValue>,
    pub pulse_parameters: HashMap<PulseParameterUid, PulseParameterValue>,
    pub markers: Vec<Marker>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ChangeOscillatorPhase {
    pub signal: SignalUid,
    pub increment: Option<ValueOrParameter<f64>>,
    pub set: Option<ValueOrParameter<f64>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Acquire {
    pub signal: SignalUid,
    pub handle: HandleUid,
    pub integration_length: TinySamples,
    pub kernels: Vec<PulseUid>,
    pub parameters: Vec<HashMap<PulseParameterUid, PulseParameterValue>>,
    pub pulse_parameters: Vec<HashMap<PulseParameterUid, PulseParameterValue>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Trigger {
    pub signal: SignalUid,
    pub state: u8,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrngSetup {
    pub range: u32,
    pub seed: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Section {
    pub uid: SectionUid,
    pub triggers: Vec<Trigger>,
    pub prng_setup: Option<PrngSetup>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PpcStep {
    pub signal: SignalUid,
    pub device: DeviceUid,
    pub channel: u16,
    pub trigger_duration: TinySamples,
    pub pump_power: Option<ValueOrParameter<f64>>,
    pub pump_frequency: Option<ValueOrParameter<f64>>,
    pub probe_power: Option<ValueOrParameter<f64>>,
    pub probe_frequency: Option<ValueOrParameter<f64>>,
    pub cancellation_phase: Option<ValueOrParameter<f64>>,
    pub cancellation_attenuation: Option<ValueOrParameter<f64>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Match {
    pub uid: SectionUid,
    pub target: MatchTarget,
    pub local: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Case {
    pub uid: SectionUid,
    // The state value that this case matches on
    // e.g., the iteration index for sweep parameter matches
    pub state: usize,
}

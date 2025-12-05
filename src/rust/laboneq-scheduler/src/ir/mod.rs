// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

//! Intermediate representation (IR) of the real-time portion of an experiment structure.

use std::collections::{HashMap, HashSet};

use laboneq_units::tinysample::TinySamples;

use crate::experiment::types::{
    DeviceUid, HandleUid, ParameterUid, PulseParameterUid, PulseParameterValue, PulseUid,
    SectionUid, SignalUid, ValueOrParameter,
};
// Re-export for convenience
pub use crate::experiment::types::MatchTarget;

#[derive(Debug, Clone, PartialEq)]
pub enum IrKind {
    Root,
    InitialOscillatorFrequency(InitialOscillatorFrequency),
    InitialLocalOscillatorFrequency(InitialLocalOscillatorFrequency),
    InitialVoltageOffset(InitialVoltageOffset),
    Loop(Loop),
    LoopIterationPreamble,
    LoopIteration,
    Reserve {
        signal: SignalUid,
    },
    PlayPulse(PlayPulse),
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
    // Placeholder for unimplemented variants
    NotYetImplemented,
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
            IrKind::Reserve { .. } => None,
            IrKind::PlayPulse(_) => None,
            IrKind::Acquire(_) => None,
            IrKind::SetOscillatorFrequency(_) => None,
            IrKind::ResetOscillatorPhase { .. } => None,
            IrKind::PpcStep(_) => None,
            IrKind::Delay { .. } => None,
            IrKind::ClearPrecompensation { .. } => None,
            IrKind::NotYetImplemented => None,
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
            IrKind::Reserve { signal } => HashSet::from_iter([signal]),
            IrKind::PlayPulse(obj) => HashSet::from_iter([&obj.signal]),
            IrKind::Acquire(obj) => HashSet::from_iter([&obj.signal]),
            IrKind::SetOscillatorFrequency(obj) => obj.values.iter().map(|(sig, _)| sig).collect(),
            IrKind::ResetOscillatorPhase { signals } => HashSet::from_iter(signals.iter()),
            IrKind::PpcStep(obj) => HashSet::from_iter([&obj.signal]),
            IrKind::Delay { signal } => HashSet::from_iter([signal]),
            IrKind::ClearPrecompensation { signal } => HashSet::from_iter([signal]),
            IrKind::Loop(_)
            | IrKind::LoopIterationPreamble
            | IrKind::LoopIteration
            | IrKind::Root
            | IrKind::Match(_)
            | IrKind::Case(_)
            | IrKind::NotYetImplemented => HashSet::new(),
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

#[derive(Debug, Clone, PartialEq)]
pub struct Loop {
    pub uid: SectionUid,
    pub iterations: usize,
    /// Sweep parameters of this loop
    /// All parameters need to be of equal length
    pub parameters: Vec<ParameterUid>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SetOscillatorFrequency {
    pub values: Vec<(SignalUid, ValueOrParameter<f64>)>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PlayPulse {
    pub signal: SignalUid,
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

#[derive(Debug, Clone, PartialEq)]
pub struct Trigger {
    pub signal: SignalUid,
    pub state: u8,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Section {
    pub uid: SectionUid,
    pub triggers: Vec<Trigger>,
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
    pub play_after: Vec<SectionUid>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Case {
    pub uid: SectionUid,
    // The state value that this case matches on
    // e.g., the iteration index for sweep parameter matches
    pub state: usize,
}

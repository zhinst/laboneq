// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::hash::{Hash, Hasher};
use std::rc::Rc;

use super::seqc_tracker::awg::HwOscillator;
use crate::ir::Samples;
use crate::ir::compilation_job::ChannelIndex;
use crate::{
    ir::{
        Match, OscillatorFrequencySweepStep, ParameterOperation,
        experiment::{Handle, SweepCommand},
    },
    signature::WaveformSignature,
};

#[derive(Clone, Debug)]
pub(crate) struct StaticWaveformSignature {
    uid: u64,
    waveform: WaveformSignature,
    signature_string: String,
}

pub(crate) enum PulseSource {
    Pulses,
    Samples,
}

impl StaticWaveformSignature {
    pub(crate) fn new(uid: u64, waveform: WaveformSignature, signature_string: String) -> Self {
        Self {
            uid,
            waveform,
            signature_string,
        }
    }

    pub(crate) fn kind(&self) -> PulseSource {
        match self.waveform {
            WaveformSignature::Pulses { .. } => PulseSource::Pulses,
            WaveformSignature::Samples { .. } => PulseSource::Samples,
        }
    }

    pub(crate) fn length(&self) -> Samples {
        self.waveform.length()
    }

    pub(crate) fn signature_string(&self) -> &str {
        &self.signature_string
    }

    pub(crate) fn is_playzero(&self) -> bool {
        self.waveform.is_playzero()
    }

    pub(crate) fn uid(&self) -> u64 {
        self.uid
    }
}

impl PartialEq for StaticWaveformSignature {
    fn eq(&self, other: &Self) -> bool {
        self.uid == other.uid
    }
}
impl Eq for StaticWaveformSignature {}
impl Hash for StaticWaveformSignature {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.uid().hash(state);
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct PlayWaveEvent {
    pub waveform: Rc<StaticWaveformSignature>,
    pub state: Option<u16>,
    pub hw_oscillator: Option<HwOscillator>,
    pub amplitude_register: u16,
    pub amplitude: Option<ParameterOperation<f64>>,
    pub increment_phase: Option<f64>,
    pub increment_phase_params: Vec<Option<String>>,
    pub channels: Vec<ChannelIndex>,
}

impl Eq for PlayWaveEvent {}
impl Hash for PlayWaveEvent {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.waveform.hash(state);
        self.state.hash(state);
        self.hw_oscillator.hash(state);
        self.amplitude_register.hash(state);
        self.amplitude.hash(state);
        self.increment_phase.map(|v| v.to_bits()).hash(state);
        self.increment_phase_params.hash(state);
        self.channels.hash(state);
    }
}

#[derive(Debug)]
pub(crate) struct AcquireEvent {
    pub channels: Vec<u8>,
}

#[derive(Debug)]
pub(crate) struct MatchEvent {
    pub handle: Option<Handle>,
    pub local: bool,
    pub user_register: Option<u16>,
    pub prng_sample: bool,
    pub section: String, // Only used for error messages
}

impl MatchEvent {
    pub(crate) fn from_ir(event: Match) -> Self {
        MatchEvent {
            handle: event.handle,
            local: event.local,
            user_register: event.user_register,
            prng_sample: event.prng_sample.is_some(),
            section: event.section_info.name.clone(),
        }
    }
}

#[derive(Debug)]
pub(crate) struct ChangeHwOscPhase {
    pub signature: PlayWaveEvent,
}

#[derive(Debug)]
pub(crate) struct PushLoop {
    pub num_repeats: u64,
    pub compressed: bool,
}

#[derive(Debug)]
pub(crate) struct Iterate {
    pub num_repeats: u64,
}

#[derive(Debug)]
pub(crate) struct PrngSetup {
    pub range: u32,
    pub seed: u32,
}

#[derive(Debug)]
pub(crate) struct TriggerOutput {
    pub state: u16,
}

#[derive(Debug)]
pub(crate) struct TriggerOutputBit {
    pub bits: u8,
    pub set: bool,
}

#[derive(Debug)]
pub(crate) struct QaEvent {
    pub acquire_events: Vec<AcquireEvent>,
    pub play_wave_events: Vec<PlayWaveEvent>,
}

#[derive(Debug)]
pub(crate) enum EventType {
    PlayWave(PlayWaveEvent),
    PlayHold(),
    Match(MatchEvent),
    ChangeHwOscPhase(ChangeHwOscPhase),
    InitAmplitudeRegister { signature: PlayWaveEvent },
    ResetPrecompensationFilters { signature: PlayWaveEvent },
    AcquireEvent(),
    PpcSweepStepStart(SweepCommand),
    PpcSweepStepEnd(),
    SetOscillatorFrequency(OscillatorFrequencySweepStep),
    ResetPhase(),
    InitialResetPhase(),
    LoopStepStart(),
    LoopStepEnd(),
    PushLoop(PushLoop),
    Iterate(Iterate),
    PrngSetup(PrngSetup),
    PrngSample(),
    PrngDropSample(),
    // This is a bit of a hack, but we need to be able to consolidate
    // the trigger output events after flattening the tree.
    // The TriggerOutputBit never appears in the final event list.
    TriggerOutputBit(TriggerOutputBit),
    TriggerOutput(TriggerOutput),
    QaEvent(QaEvent),
}

impl Default for EventType {
    fn default() -> Self {
        // Set to any event type, as this is only need for std::mem::take()
        EventType::InitialResetPhase()
    }
}

#[derive(Debug, Default)]
pub(crate) struct AwgEvent {
    pub start: Samples,
    pub end: Samples,
    pub kind: EventType,
}

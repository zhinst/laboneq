// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::hash::{Hash, Hasher};

use super::seqc_tracker::awg::HwOscillator;
use crate::ir::Samples;
use crate::ir::compilation_job::ChannelIndex;
use crate::{
    ir::{
        InitAmplitudeRegister, Match, OscillatorFrequencySweepStep, ParameterOperation,
        PlayAcquire, PlayWave, QaEvent as QaEventIr,
        experiment::{Handle, SweepCommand},
    },
    signature::WaveformSignature,
};

#[derive(Clone, Debug, PartialEq)]
pub struct PlayWaveEvent {
    pub waveform: WaveformSignature,
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

impl PlayWaveEvent {
    pub fn from_ir(
        event: PlayWave,
        state: Option<u16>,
        hw_oscillator: Option<HwOscillator>,
    ) -> Self {
        PlayWaveEvent {
            waveform: event.waveform,
            state,
            hw_oscillator,
            amplitude_register: event.amplitude_register,
            amplitude: event.amplitude,
            increment_phase: event.increment_phase,
            increment_phase_params: event.increment_phase_params,
            channels: event
                .signals
                .first()
                .map_or_else(Vec::new, |sig| sig.channels.clone()),
        }
    }
}

#[derive(Debug)]
pub struct AcquireEvent {
    pub channels: Vec<u8>,
}

impl AcquireEvent {
    pub fn from_ir(event: PlayAcquire) -> Self {
        let channels = event.signal().channels.to_vec();
        AcquireEvent { channels }
    }
}

impl QaEvent {
    pub fn from_ir(event: QaEventIr) -> Self {
        let (acquires, waveforms) = event.into_parts();
        QaEvent {
            acquire_events: acquires.into_iter().map(AcquireEvent::from_ir).collect(),
            play_wave_events: waveforms
                .into_iter()
                .map(|wf| PlayWaveEvent::from_ir(wf, None, None))
                .collect(),
        }
    }
}

#[derive(Debug)]
pub struct MatchEvent {
    pub handle: Option<Handle>,
    pub local: bool,
    pub user_register: Option<u16>,
    pub prng_sample: bool,
    pub section: String, // Only used for error messages
}

impl MatchEvent {
    pub fn from_ir(event: Match) -> Self {
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
pub struct ChangeHwOscPhase {
    pub phase: f64,
    pub hw_oscillator: Option<HwOscillator>,
    pub parameter: Option<String>,
}

#[derive(Debug)]
pub struct PushLoop {
    pub num_repeats: u64,
    pub compressed: bool,
}

#[derive(Debug)]
pub struct Iterate {
    pub num_repeats: u64,
}

#[derive(Debug)]
pub struct PrngSetup {
    pub range: u32,
    pub seed: u32,
}

#[derive(Debug)]
pub struct TriggerOutput {
    pub state: u16,
}

#[derive(Debug)]
pub struct TriggerOutputBit {
    pub bit: u8,
    pub set: bool,
}

#[derive(Debug)]
pub struct QaEvent {
    pub acquire_events: Vec<AcquireEvent>,
    pub play_wave_events: Vec<PlayWaveEvent>,
}

#[derive(Debug)]
pub enum EventType {
    PlayWave(PlayWaveEvent),
    PlayHold(),
    Match(MatchEvent),
    ChangeHwOscPhase(ChangeHwOscPhase),
    InitAmplitudeRegister(InitAmplitudeRegister),
    ResetPrecompensationFilters(Samples),
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
pub struct AwgEvent {
    pub start: Samples,
    pub end: Samples,
    pub kind: EventType,
}

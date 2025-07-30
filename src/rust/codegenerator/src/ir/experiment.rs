// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use num_complex::Complex;

use crate::ir::compilation_job as cjob;
use crate::node;
use crate::signature::WaveformSignature;
use core::panic;
use std::rc::Rc;
use std::sync::Arc;

use super::compilation_job::{PulseDef, Signal};
pub type Samples = i64;
pub type IrNode = node::Node<Samples, NodeKind>;

/// Represents an operation on a parameter.
#[derive(Debug, Clone, PartialEq)]
pub enum ParameterOperation<T> {
    INCREMENT(T),
    SET(T),
}

impl<T: Clone> ParameterOperation<T> {
    pub fn value(&self) -> T {
        match self {
            ParameterOperation::SET(value) => value.clone(),
            ParameterOperation::INCREMENT(value) => value.clone(),
        }
    }

    pub(crate) fn value_mut(&mut self) -> &mut T {
        match self {
            ParameterOperation::SET(value) => value,
            ParameterOperation::INCREMENT(value) => value,
        }
    }
}

pub type SectionId = u32;

#[derive(Debug, Default, PartialEq)]
pub struct SectionInfo {
    pub name: String,
    pub id: SectionId,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Case {
    pub signals: Vec<Rc<Signal>>,
    pub length: Samples,
    pub state: u16,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SignalFrequency {
    pub signal: Rc<Signal>,
    pub frequency: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct InitialOscillatorFrequency {
    values: Vec<SignalFrequency>,
}

impl InitialOscillatorFrequency {
    pub fn new(values: Vec<SignalFrequency>) -> Self {
        InitialOscillatorFrequency { values }
    }

    pub fn iter(&self) -> impl Iterator<Item = &SignalFrequency> {
        self.values.iter()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SetOscillatorFrequency {
    values: Vec<SignalFrequency>,
    iteration: usize,
}

impl SetOscillatorFrequency {
    pub fn new(values: Vec<SignalFrequency>, iteration: usize) -> Self {
        SetOscillatorFrequency { values, iteration }
    }

    pub fn iter(&self) -> impl Iterator<Item = &SignalFrequency> {
        self.values.iter()
    }

    pub fn iteration(&self) -> usize {
        self.iteration
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PhaseReset {
    // Whether or not the phase reset should reset the software oscillators
    // listed in `signals`.
    // If `reset_sw_oscillators` is false, the phase reset will only
    // apply to hardware oscillators.
    pub reset_sw_oscillators: bool,
    pub signals: Vec<Rc<Signal>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PlayPulse {
    pub signal: Rc<Signal>,
    pub length: Samples,
    pub amplitude: Option<Complex<f64>>,
    pub amp_param_name: Option<String>,
    pub phase: f64,
    pub set_oscillator_phase: Option<f64>,
    pub increment_oscillator_phase: Option<f64>,
    pub incr_phase_param_name: Option<String>,
    pub id_pulse_params: Option<u64>,
    // TODO: Consider replacing this with an ID of markers
    pub markers: Vec<cjob::Marker>,
    pub pulse_def: Option<Arc<PulseDef>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AcquirePulse {
    // TODO: Should this just be handle?
    // Currently we restrict kernel per signal, so lets keep it for now
    pub signal: Rc<Signal>,
    /// Integration length
    pub length: Samples,
    /// Single acquire can consist of multiple individual pulses
    /// Length of pulse defs must match pulse params ID
    pub pulse_defs: Vec<Arc<PulseDef>>,
    pub id_pulse_params: Vec<Option<u64>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Match {
    pub section_info: Arc<SectionInfo>,
    pub length: Samples,
    pub handle: Option<String>,
    pub user_register: Option<i64>,
    pub local: bool,
    pub prng_sample: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Loop {
    pub section_info: Arc<SectionInfo>,
    /// Length of the loop in samples
    pub length: Samples,
    /// A flag representing whether the loop is compressed
    pub compressed: bool,
    /// Number of iterations in the loop
    pub count: u64,
}

/// One iteration of an loop.
#[derive(Debug, Clone, PartialEq)]
pub struct LoopIteration {
    /// Length of the iteration in samples
    pub length: Samples,
    /// Parameters used in this iteration
    pub parameters: Vec<Arc<cjob::SweepParameter>>,
    /// PRNG sample name to draw from
    pub prng_sample: Option<String>,
    // Whether or not the iteration is a shadow of previous iteration.
    pub shadow: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PlayWave {
    pub signals: Vec<Rc<Signal>>,
    pub waveform: WaveformSignature,
    pub oscillator: Option<String>,
    pub amplitude_register: u16,
    pub amplitude: Option<ParameterOperation<f64>>,
    pub increment_phase: Option<f64>,
    /// Increment phase parameters.
    /// None indicates a phase increment that is unrelated to any parameters
    pub increment_phase_params: Vec<Option<String>>,
}

impl PlayWave {
    pub fn length(&self) -> Samples {
        self.waveform.length()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PlayHold {
    pub length: Samples,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FrameChange {
    pub length: Samples,
    pub phase: f64,
    pub parameter: Option<String>,
    pub signal: Rc<Signal>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct InitAmplitudeRegister {
    pub register: u16,
    pub value: ParameterOperation<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ResetPrecompensationFilters {
    pub length: Samples,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PlayAcquire {
    signal: Rc<Signal>,
    length: Samples,
    // Acquire pulse definitions
    pulse_defs: Vec<Arc<PulseDef>>,
    id_pulse_params: Vec<Option<u64>>,
    oscillator_frequency: f64,
}

impl PlayAcquire {
    pub(crate) fn new(
        signal: Rc<Signal>,
        length: Samples,
        pulse_defs: Vec<Arc<PulseDef>>,
        oscillator_frequency: f64,
        id_pulse_params: Vec<Option<u64>>,
    ) -> Self {
        PlayAcquire {
            signal,
            length,
            pulse_defs,
            oscillator_frequency,
            id_pulse_params,
        }
    }

    pub fn signal(&self) -> &Signal {
        &self.signal
    }

    pub fn length(&self) -> Samples {
        self.length
    }

    pub fn pulse_defs(&self) -> &[Arc<PulseDef>] {
        &self.pulse_defs
    }

    pub fn id_pulse_params(&self) -> &[Option<u64>] {
        &self.id_pulse_params
    }

    pub fn oscillator_frequency(&self) -> f64 {
        self.oscillator_frequency
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PpcDevice {
    pub device: String,
    pub channel: u16,
}

#[derive(Debug, Clone, PartialEq)]
pub struct QaEvent {
    acquires: Vec<PlayAcquire>,
    play_waves: Vec<PlayWave>,
    length: Samples,
}

impl QaEvent {
    pub fn new(acquires: Vec<PlayAcquire>, waveforms: Vec<PlayWave>) -> Self {
        let length_acquires = acquires.iter().map(|a| a.length()).max().unwrap_or(0);
        let length_waveforms = waveforms.iter().map(|w| w.length()).max().unwrap_or(0);
        let length = length_acquires.max(length_waveforms);
        QaEvent {
            acquires,
            play_waves: waveforms,
            length,
        }
    }

    pub fn acquires(&self) -> &[PlayAcquire] {
        &self.acquires
    }

    pub fn into_parts(self) -> (Vec<PlayAcquire>, Vec<PlayWave>) {
        (self.acquires, self.play_waves)
    }

    pub fn play_waves(&self) -> &[PlayWave] {
        &self.play_waves
    }

    pub fn length(&self) -> Samples {
        self.length
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SweepCommand {
    pub pump_power: Option<f64>,
    pub pump_frequency: Option<f64>,
    pub probe_power: Option<f64>,
    pub probe_frequency: Option<f64>,
    pub cancellation_phase: Option<f64>,
    pub cancellation_attenuation: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PpcSweepStep {
    pub signal: Rc<Signal>,
    pub length: Samples,
    pub sweep_command: SweepCommand,
    pub ppc_device: Arc<PpcDevice>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PrngSetup {
    pub range: u16,
    pub seed: u32,
    pub section_info: Arc<SectionInfo>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PrngSample {
    pub length: Samples,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TriggerBitData {
    pub signal: Rc<Signal>,
    pub bit: u8,
    pub set: bool,
    pub section_info: Arc<SectionInfo>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LinearParameterInfo {
    pub start: f64,
    pub step: f64,
    pub count: usize,
}

/// Represents a single step in the oscillator frequency sweep.
/// This is used to sweep the frequency of an oscillator over multiple iterations.
#[derive(Debug, Clone, PartialEq)]
pub struct OscillatorFrequencySweepStep {
    /// Iteration number in the sweep. 0 means the first iteration.
    pub iteration: usize,
    /// Oscillator index in the sweep.
    pub osc_index: u16,
    /// Information about the parameter that is being swept.
    pub parameter: Arc<LinearParameterInfo>,
}

/// Represents a set of oscillator frequency sweep steps at a given time.
/// This is used to set the frequency of one or multiple oscillators in parallel at a specific time.
#[derive(Debug, Clone, PartialEq)]
pub struct SetOscillatorFrequencySweep {
    pub length: Samples,
    /// List of parallel oscillator frequency steps in this event.
    pub oscillators: Vec<OscillatorFrequencySweepStep>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Section {
    pub length: Samples,
    pub trigger_output: Vec<(Rc<Signal>, u8)>,
    pub prng_setup: Option<PrngSetup>,
    pub section_info: Arc<SectionInfo>,
}

// TODO: Think of separating AWG specific nodes and public nodes, which
// are used to build the experiment.

/// Nodes that can live in the IR tree.
#[derive(Debug, Clone, PartialEq)] // TODO: Add eq trait once all nodes implement it
pub enum NodeKind {
    // IR Nodes
    // IR nodes are consumed by the code generator.
    PlayPulse(PlayPulse),
    AcquirePulse(AcquirePulse),
    Case(Case),
    InitialOscillatorFrequency(InitialOscillatorFrequency),
    SetOscillatorFrequency(SetOscillatorFrequency),
    PhaseReset(PhaseReset),
    PrecompensationFilterReset { signal: Rc<Signal> },
    PpcSweepStep(PpcSweepStep),
    Match(Match),
    Loop(Loop),
    LoopIteration(LoopIteration),
    TriggerSet(TriggerBitData),
    Section(Section),
    // AWG nodes
    // AWG nodes are produced by the code generator.
    PlayWave(PlayWave),
    PlayHold(PlayHold),
    Acquire(PlayAcquire),
    QaEvent(QaEvent),
    FrameChange(FrameChange),
    InitAmplitudeRegister(InitAmplitudeRegister),
    ResetPrecompensationFilters(ResetPrecompensationFilters),
    PpcStep(PpcSweepStep),
    ResetPhase(),
    InitialResetPhase(),
    SetupPrng(PrngSetup),
    DropPrngSetup,
    SetTrigger(TriggerBitData),
    SamplePrng(PrngSample),
    SetOscillatorFrequencySweep(SetOscillatorFrequencySweep),
    // No-op node.
    // Should be treated as such, except it's length must be
    // taken into account.
    // Currently represents both not yet implemented nodes and
    // nodes marked ready for deletion.
    // TODO: Once all the nodes are implemented, add a pass to prune Nop
    // nodes and remove 'length' field.
    Nop { length: Samples },
}

impl NodeKind {
    pub fn set_length(&mut self, value: Samples) {
        match self {
            NodeKind::PlayPulse(x) => x.length = value,
            NodeKind::AcquirePulse(x) => x.length = value,
            NodeKind::Case(x) => x.length = value,
            NodeKind::InitialOscillatorFrequency(_) => {}
            NodeKind::SetOscillatorFrequency(_) => {}
            NodeKind::PhaseReset(_) => {}
            NodeKind::InitialResetPhase() => {}
            NodeKind::PrecompensationFilterReset { .. } => {}
            NodeKind::PpcSweepStep(x) => x.length = value,
            NodeKind::Match(x) => x.length = value,
            NodeKind::Loop(x) => x.length = value,
            NodeKind::LoopIteration(x) => x.length = value,
            NodeKind::Nop { length } => *length = value,
            NodeKind::SetupPrng(_) => {}
            NodeKind::DropPrngSetup => {}
            NodeKind::TriggerSet(_) => {}
            NodeKind::Section(_) => panic!("Can't set length of Section nodes"),
            // Disallow settings of AWG nodes.
            _ => panic!("Can't set length of AWG nodes"),
        }
    }

    pub fn length(&self) -> Samples {
        match self {
            NodeKind::PlayPulse(x) => x.length,
            NodeKind::AcquirePulse(x) => x.length,
            NodeKind::Case(x) => x.length,
            NodeKind::InitialOscillatorFrequency(_) => 0,
            NodeKind::SetOscillatorFrequency(_) => 0,
            NodeKind::PhaseReset(_) => 0,
            NodeKind::InitialResetPhase() => 0,
            NodeKind::PrecompensationFilterReset { .. } => 0,
            NodeKind::PpcSweepStep(x) => x.length,
            NodeKind::Match(x) => x.length,
            NodeKind::Loop(x) => x.length,
            NodeKind::LoopIteration(x) => x.length,
            NodeKind::Nop { length } => *length,
            NodeKind::DropPrngSetup => 0,
            NodeKind::TriggerSet(_) => 0,
            NodeKind::FrameChange(x) => x.length,
            NodeKind::Acquire(x) => x.length,
            NodeKind::PlayWave(x) => x.length(),
            NodeKind::PlayHold(x) => x.length,
            NodeKind::QaEvent(x) => x.length(),
            NodeKind::InitAmplitudeRegister(_) => 0,
            NodeKind::ResetPrecompensationFilters(x) => x.length,
            NodeKind::PpcStep(x) => x.length,
            NodeKind::ResetPhase() => 0,
            NodeKind::SetupPrng(_) => 0,
            NodeKind::SetTrigger(_) => 0,
            NodeKind::SamplePrng(x) => x.length,
            NodeKind::SetOscillatorFrequencySweep(x) => x.length,
            NodeKind::Section(x) => x.length,
        }
    }
}

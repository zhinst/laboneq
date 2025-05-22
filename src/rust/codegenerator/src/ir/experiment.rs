// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use num_complex::Complex;

use crate::ir::compilation_job as cjob;
use crate::node;
use crate::signature;
use std::rc::Rc;
use std::sync::Arc;
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

#[derive(Debug, Clone)]
pub struct Case {
    pub signals: Vec<Rc<cjob::Signal>>,
    pub length: Samples,
    pub state: u16,
}

#[derive(Debug, Clone)]
pub struct SetOscillatorFrequency {
    // NOTE: Also initial oscillator frequency from Python IR
    pub values: Vec<(Rc<cjob::Signal>, f64)>,
}

#[derive(Debug, Clone)]
pub struct PhaseReset {
    pub reset_sw_oscillators: bool,
}

#[derive(Debug, Clone)]
pub struct PlayPulse {
    pub signal: Rc<cjob::Signal>,
    pub length: Samples,
    pub amplitude: Option<Complex<f64>>,
    pub amp_param_name: Option<String>,
    pub phase: f64,
    pub set_oscillator_phase: Option<f64>,
    pub increment_oscillator_phase: Option<f64>,
    pub incr_phase_param_name: Option<String>,
    pub id_pulse_params: Option<usize>,
    pub markers: Vec<cjob::Marker>,
    pub pulse_def: Option<Arc<cjob::PulseDef>>,
}

#[derive(Debug, Clone)]
pub struct Match {
    pub section: String,
    pub length: Samples,
    pub handle: Option<String>,
    pub user_register: Option<i64>,
    pub local: bool,
    pub prng_sample: Option<String>,
}

#[derive(Debug, Clone)]
pub struct Loop {
    /// Length of the loop in samples
    pub length: Samples,
    /// A flag representing whether the loop is compressed
    pub compressed: bool,
}

/// One iteration of an loop.
#[derive(Debug, Clone)]
pub struct LoopIteration {
    /// Length of the iteration in samples
    pub length: Samples,
    /// Iteration number within the loop
    pub iteration: u64,
    /// Parameters used in this iteration
    pub parameters: Vec<Arc<cjob::SweepParameter>>,
}

#[derive(Debug, Clone)]
pub struct PlayWave {
    pub signals: Vec<Rc<cjob::Signal>>,
    pub waveform: signature::Waveform,
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
        self.waveform.length
    }

    pub fn waveform(&self) -> &signature::Waveform {
        &self.waveform
    }
}

#[derive(Debug, Clone)]
pub struct FrameChange {
    pub length: Samples,
    pub phase: f64,
    pub parameter: Option<String>,
    pub signal: Rc<cjob::Signal>,
}

#[derive(Debug, Clone)]
pub struct InitAmplitudeRegister {
    pub register: u16,
    pub value: ParameterOperation<f64>,
}

// TODO: Think of separating AWG specific nodes and public nodes, which
// are used to build the experiment.

/// Nodes that can live in the IR tree.
#[derive(Debug, Clone)]
pub enum NodeKind {
    // IR Nodes
    // IR nodes are consumed by the code generator.
    PlayPulse(PlayPulse),
    Case(Case),
    SetOscillatorFrequency(SetOscillatorFrequency),
    PhaseReset(PhaseReset),
    Match(Match),
    Loop(Loop),
    LoopIteration(LoopIteration),
    // AWG nodes
    // AWG nodes are produced by the code generator.
    PlayWave(PlayWave),
    FrameChange(FrameChange),
    InitAmplitudeRegister(InitAmplitudeRegister),
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
            NodeKind::Case(x) => x.length = value,
            NodeKind::SetOscillatorFrequency(_) => {}
            NodeKind::PhaseReset(_) => {}
            NodeKind::Match(x) => x.length = value,
            NodeKind::Loop(x) => x.length = value,
            NodeKind::LoopIteration(x) => x.length = value,
            NodeKind::Nop { length } => *length = value,
            // Disallow settings of AWG nodes.
            _ => panic!("Can't set length of AWG nodes"),
        }
    }

    pub fn length(&self) -> Samples {
        match self {
            NodeKind::PlayPulse(x) => x.length,
            NodeKind::Case(x) => x.length,
            NodeKind::SetOscillatorFrequency(_) => 0,
            NodeKind::PhaseReset(_) => 0,
            NodeKind::Match(x) => x.length,
            NodeKind::Loop(x) => x.length,
            NodeKind::LoopIteration(x) => x.length,
            NodeKind::Nop { length } => *length,
            NodeKind::FrameChange(x) => x.length,
            NodeKind::PlayWave(x) => x.length(),
            NodeKind::InitAmplitudeRegister(_) => 0,
        }
    }
}

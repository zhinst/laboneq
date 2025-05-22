// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

pub mod compilation_job;
pub mod experiment;

pub use experiment::{
    Case, FrameChange, InitAmplitudeRegister, IrNode, Loop, LoopIteration, Match, NodeKind,
    ParameterOperation, PhaseReset, PlayPulse, PlayWave, Samples, SetOscillatorFrequency,
};

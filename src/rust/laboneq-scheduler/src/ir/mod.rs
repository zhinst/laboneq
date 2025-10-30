// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

//! Intermediate representation (IR) of the real-time portion of an experiment structure.
use crate::experiment::types::{ParameterUid, RealValue, SectionUid, SignalUid};

#[derive(Debug, Clone, PartialEq)]
pub enum IrKind {
    Root,
    InitialOscillatorFrequency(InitialOscillatorFrequency),
    InitialLocalOscillatorFrequency(InitialLocalOscillatorFrequency),
    InitialVoltageOffset(InitialVoltageOffset),
    Loop(Loop),
    LoopIterationPreamble,
    LoopIteration,
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
            IrKind::Root => None,
            IrKind::InitialOscillatorFrequency(_) => None,
            IrKind::InitialLocalOscillatorFrequency(_) => None,
            IrKind::InitialVoltageOffset(_) => None,
            IrKind::LoopIterationPreamble => None,
            IrKind::LoopIteration => None,
            IrKind::NotYetImplemented => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct InitialOscillatorFrequency {
    pub values: Vec<(SignalUid, RealValue)>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct InitialLocalOscillatorFrequency {
    pub signal: SignalUid,
    pub value: RealValue,
}

#[derive(Debug, Clone, PartialEq)]
pub struct InitialVoltageOffset {
    pub signal: SignalUid,
    pub value: RealValue,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Loop {
    pub uid: SectionUid,
    pub iterations: usize,
    /// Sweep parameters of this loop
    /// All parameters need to be of equal length
    pub parameters: Vec<ParameterUid>,
}

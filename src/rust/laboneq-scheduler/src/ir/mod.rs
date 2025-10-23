// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

//! Intermediate representation (IR) of the real-time portion of an experiment structure.
use crate::experiment::types::{RealValue, SectionUid, SignalUid};

#[derive(Debug, Clone, PartialEq)]
pub enum IrKind {
    Root,
    InitialOscillatorFrequency(InitialOscillatorFrequency),
    InitialLocalOscillatorFrequency(InitialLocalOscillatorFrequency),
    InitialVoltageOffset(InitialVoltageOffset),
    // Placeholder for unimplemented variants
    NotYetImplemented,
}

pub struct SectionInfo<'a> {
    pub uid: &'a SectionUid,
}

impl IrKind {
    pub fn section_info<'a>(self: &'a IrKind) -> Option<SectionInfo<'a>> {
        match self {
            IrKind::Root => None,
            IrKind::InitialOscillatorFrequency(_) => None,
            IrKind::InitialLocalOscillatorFrequency(_) => None,
            IrKind::InitialVoltageOffset(_) => None,
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

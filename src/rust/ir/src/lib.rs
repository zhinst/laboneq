// Copyright 2024 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use common::RuntimeError;
use loop_ir::LoopIr;
use loop_iteration_ir::{LoopIterationIr, LoopIterationPreambleIr};
use oscillator_ir::{InitialOscillatorFrequencyIr, SetOscillatorFrequencyIr};
use pulse_ir::PulseIr;
use section_ir::SectionIr;
use single_awg_ir::SingleAwgIr;

mod common;

pub mod interval_ir;
pub mod loop_ir;
pub mod loop_iteration_ir;
pub mod oscillator_ir;
pub mod pulse_ir;
pub mod section_ir;
pub mod single_awg_ir;

#[derive(Debug)]
pub enum IrNode {
    SectionIr(SectionIr),
    LoopIr(LoopIr),
    LoopIterationPreambleIr(LoopIterationPreambleIr),
    LoopIterationIr(LoopIterationIr),
    PulseIr(PulseIr),
    SetOscillatorFrequencyIr(SetOscillatorFrequencyIr),
    InitialOscillatorFrequencyIr(InitialOscillatorFrequencyIr),
    SingleAwgIr(SingleAwgIr),
}

pub fn deep_copy_ir_node(node: &IrNode) -> Result<IrNode, RuntimeError> {
    let node_copy: IrNode = match node {
        IrNode::SectionIr(section) => IrNode::SectionIr(section.deep_copy()?),
        IrNode::LoopIr(ir_loop) => IrNode::LoopIr(ir_loop.deep_copy()?),
        IrNode::LoopIterationPreambleIr(loop_iteration_preamble) => {
            IrNode::LoopIterationPreambleIr(loop_iteration_preamble.deep_copy()?)
        }
        IrNode::LoopIterationIr(loop_iteration) => {
            IrNode::LoopIterationIr(loop_iteration.deep_copy()?)
        }
        IrNode::PulseIr(pulse) => IrNode::PulseIr(pulse.deep_copy()?),
        IrNode::SetOscillatorFrequencyIr(set_oscillator_frequency) => {
            IrNode::SetOscillatorFrequencyIr(set_oscillator_frequency.deep_copy()?)
        }
        IrNode::InitialOscillatorFrequencyIr(initial_oscillator_frequency) => {
            IrNode::InitialOscillatorFrequencyIr(initial_oscillator_frequency.deep_copy()?)
        }
        IrNode::SingleAwgIr(single_awg) => IrNode::SingleAwgIr(single_awg.deep_copy()?),
    };
    Ok(node_copy)
}

pub trait DeepCopy {
    fn deep_copy(&self) -> Result<Self, RuntimeError>
    where
        Self: Sized;
}

// Copyright 2024 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use loop_iteration_ir::{LoopIterationPreamblePy, LoopIterationPy};
use oscillator_ir::{InitialOscillatorFrequencyPy, SetOscillatorFrequencyPy};
use pulse_ir::PulsePy;
use pyo3::prelude::*;

mod common;
mod interval_ir;
mod loop_ir;
mod loop_iteration_ir;
mod oscillator_ir;
mod pulse_ir;
mod section_ir;
mod single_awg_ir;

use interval_ir::IntervalPy;
use loop_ir::LoopPy;
use section_ir::SectionPy;
use single_awg_ir::SingleAwgPy;

pub fn create_module<'a>(parent_module: &Bound<'a, PyModule>) -> PyResult<Bound<'a, PyModule>> {
    let rust_ir_module = PyModule::new_bound(parent_module.py(), "rust_ir")?;
    rust_ir_module.add_class::<IntervalPy>()?;
    rust_ir_module.add_class::<SectionPy>()?;
    rust_ir_module.add_class::<LoopPy>()?;
    rust_ir_module.add_class::<LoopIterationPreamblePy>()?;
    rust_ir_module.add_class::<LoopIterationPy>()?;
    rust_ir_module.add_class::<InitialOscillatorFrequencyPy>()?;
    rust_ir_module.add_class::<SetOscillatorFrequencyPy>()?;
    rust_ir_module.add_class::<PulsePy>()?;
    rust_ir_module.add_class::<SingleAwgPy>()?;
    Ok(rust_ir_module)
}

// Copyright 2024 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

mod common;
mod interval_ir;
mod loop_ir;
mod section_ir;

use interval_ir::IntervalPy;

use loop_ir::LoopPy;
use pyo3::prelude::*;
use section_ir::SectionPy;

pub fn create_module<'a>(parent_module: &Bound<'a, PyModule>) -> PyResult<Bound<'a, PyModule>> {
    let rust_ir_module = PyModule::new_bound(parent_module.py(), "rust_ir")?;
    rust_ir_module.add_class::<IntervalPy>()?;
    rust_ir_module.add_class::<SectionPy>()?;
    rust_ir_module.add_class::<LoopPy>()?;
    Ok(rust_ir_module)
}

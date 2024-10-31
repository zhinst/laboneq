// Copyright 2024 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use ir::{interval_ir::IntervalIr, loop_ir::LoopIr, IrNode};

use pyo3::exceptions::PySystemError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use std::{
    collections::HashSet,
    sync::{Arc, Mutex},
};

use crate::{
    common::*, impl_extractable_ir_node, impl_interval_methods, impl_python_dunders,
    interval_ir::IntervalPy,
};
use ir::deep_copy_ir_node;

#[pyclass]
#[pyo3(name = "LoopIR")]
#[derive(Clone)]
pub struct LoopPy(pub Arc<Mutex<IrNode>>);

#[pymethods]
impl LoopPy {
    #[new]
    #[pyo3(signature = (interval=None, section="".to_string(), trigger_output=HashSet::new(), prng_setup=None, compressed=false, iterations=0))]
    pub fn new(
        interval: Option<IntervalPy>,
        section: String,
        trigger_output: HashSet<(String, i64)>,
        prng_setup: Option<Py<PyAny>>,
        compressed: bool,
        iterations: i64,
    ) -> Self {
        LoopPy(Arc::new(Mutex::new(IrNode::LoopIr(LoopIr {
            interval: match interval {
                Some(interval) => interval.0.clone(),
                None => Arc::new(Mutex::new(IntervalIr::default())),
            },
            section,
            trigger_output,
            prng_setup,
            compressed,
            iterations,
        }))))
    }

    #[setter]
    pub fn set_section(&mut self, section: String) -> PyResult<()> {
        let mut guard = unlock_mutex(&self.0)?;
        let IrNode::LoopIr(ir_loop) = &mut *guard else {
            panic!("Encountered wrapper with inconsistent internal type");
        };
        ir_loop.section = section;
        Ok(())
    }

    #[getter]
    pub fn get_section(&self) -> PyResult<String> {
        let guard = unlock_mutex(&self.0)?;
        let IrNode::LoopIr(ir_loop) = &*guard else {
            panic!("Encountered wrapper with inconsistent internal type");
        };
        Ok(ir_loop.section.clone())
    }
}

impl_extractable_ir_node!(LoopPy);
impl_interval_methods!(LoopPy, LoopIr);
impl_python_dunders!(LoopPy, LoopIr);

// Copyright 2024 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use ir::{interval_ir::IntervalIr, section_ir::SectionIr, IrNode};

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
#[pyo3(name = "SectionIR")]
#[derive(Clone)]
pub struct SectionPy(pub Arc<Mutex<IrNode>>);

#[pymethods]
impl SectionPy {
    #[new]
    #[pyo3(signature = (interval=None, section="".to_string(), trigger_output=HashSet::new(), prng_setup=None))]
    pub fn new(
        interval: Option<IntervalPy>,
        section: String,
        trigger_output: HashSet<(String, i64)>,
        prng_setup: Option<Py<PyAny>>,
    ) -> Self {
        SectionPy(Arc::new(Mutex::new(IrNode::SectionIr(SectionIr {
            interval: match interval {
                Some(interval) => interval.0.clone(),
                None => Arc::new(Mutex::new(IntervalIr::default())),
            },
            section,
            trigger_output,
            prng_setup,
        }))))
    }

    #[setter]
    pub fn set_section(&mut self, section: String) -> PyResult<()> {
        let mut guard: std::sync::MutexGuard<'_, IrNode> = unlock_mutex(&self.0)?;
        let IrNode::SectionIr(ir_section) = &mut *guard else {
            panic!("Encountered wrapper with inconsistent internal type");
        };
        ir_section.section = section;
        Ok(())
    }

    #[getter]
    pub fn get_section(&self) -> PyResult<String> {
        let guard = unlock_mutex(&self.0)?;
        let IrNode::SectionIr(ir_section) = &*guard else {
            panic!("Encountered wrapper with inconsistent internal type");
        };
        Ok(ir_section.section.clone())
    }
}

impl_extractable_ir_node!(SectionPy);
impl_interval_methods!(SectionPy, SectionIr);
impl_python_dunders!(SectionPy, SectionIr);

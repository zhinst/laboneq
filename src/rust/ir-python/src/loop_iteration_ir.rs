// Copyright 2024 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use ir::loop_iteration_ir::LoopIterationIr;
use ir::{interval_ir::IntervalIr, loop_iteration_ir::LoopIterationPreambleIr, IrNode};

use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::{exceptions::PySystemError, types::PyList};

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
#[pyo3(name = "LoopIterationPreambleIR")]
#[derive(Clone)]
pub struct LoopIterationPreamblePy(pub Arc<Mutex<IrNode>>);

#[pymethods]
impl LoopIterationPreamblePy {
    #[new]
    #[pyo3(signature = (interval=None))]
    pub fn new(interval: Option<IntervalPy>) -> Self {
        LoopIterationPreamblePy(Arc::new(Mutex::new(IrNode::LoopIterationPreambleIr(
            LoopIterationPreambleIr {
                interval: match interval {
                    Some(interval) => interval.0.clone(),
                    None => Arc::new(Mutex::new(IntervalIr::default())),
                },
            },
        ))))
    }
}

impl_extractable_ir_node!(LoopIterationPreamblePy);
impl_interval_methods!(LoopIterationPreamblePy, LoopIterationPreambleIr);
impl_python_dunders!(LoopIterationPreamblePy, LoopIterationPreambleIr);

#[pyclass]
#[pyo3(name = "LoopIterationIR")]
#[derive(Clone)]
pub struct LoopIterationPy(pub Arc<Mutex<IrNode>>);

#[pymethods]
impl LoopIterationPy {
    #[new]
    #[pyo3(signature = (interval=None, section="".to_string(), trigger_output=HashSet::new(), prng_setup=None, iteration=0, sweep_parameters=None, num_repeats=0, shadow=false, prng_sample=None))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        interval: Option<IntervalPy>,
        section: String,
        trigger_output: HashSet<(String, i64)>,
        prng_setup: Option<Py<PyAny>>,
        iteration: i64,
        sweep_parameters: Option<Py<PyList>>,
        num_repeats: i64,
        shadow: bool,
        prng_sample: Option<String>,
        py: Python,
    ) -> Self {
        LoopIterationPy(Arc::new(Mutex::new(IrNode::LoopIterationIr(
            LoopIterationIr {
                interval: match interval {
                    Some(interval) => interval.0.clone(),
                    None => Arc::new(Mutex::new(IntervalIr::default())),
                },
                section,
                trigger_output,
                prng_setup,
                iteration,
                sweep_parameters: match sweep_parameters {
                    Some(sweep_parameters) => sweep_parameters,
                    None => PyList::empty_bound(py).into(),
                },
                num_repeats,
                prng_sample,
                shadow,
            },
        ))))
    }
}

impl_extractable_ir_node!(LoopIterationPy);
impl_interval_methods!(LoopIterationPy, LoopIterationIr);
impl_python_dunders!(LoopIterationPy, LoopIterationIr);

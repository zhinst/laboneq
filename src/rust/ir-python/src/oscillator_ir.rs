// Copyright 2024 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use ir::oscillator_ir::{InitialOscillatorFrequencyIr, SetOscillatorFrequencyIr};
use ir::{interval_ir::IntervalIr, IrNode};

use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::{exceptions::PySystemError, types::PyList};

use std::sync::{Arc, Mutex};

use crate::{
    common::*, impl_extractable_ir_node, impl_interval_methods, impl_python_dunders,
    interval_ir::IntervalPy,
};
use ir::deep_copy_ir_node;

#[pyclass]
#[pyo3(name = "SetOscillatorFrequencyIR")]
#[derive(Clone)]
pub struct SetOscillatorFrequencyPy(pub Arc<Mutex<IrNode>>);

#[pymethods]
impl SetOscillatorFrequencyPy {
    #[new]
    #[pyo3(signature = (section, oscillators, params, values, iteration, interval=None))]
    pub fn new(
        section: String,
        oscillators: Py<PyList>,
        params: Vec<String>,
        values: Vec<f64>,
        iteration: i64,
        interval: Option<IntervalPy>,
    ) -> Self {
        SetOscillatorFrequencyPy(Arc::new(Mutex::new(IrNode::SetOscillatorFrequencyIr(
            SetOscillatorFrequencyIr {
                interval: match interval {
                    Some(interval) => interval.0.clone(),
                    None => Arc::new(Mutex::new(IntervalIr::default())),
                },
                section,
                oscillators,
                params,
                values,
                iteration,
            },
        ))))
    }
}

impl_extractable_ir_node!(SetOscillatorFrequencyPy);
impl_interval_methods!(SetOscillatorFrequencyPy, SetOscillatorFrequencyIr);
impl_python_dunders!(SetOscillatorFrequencyPy, SetOscillatorFrequencyIr);

#[pyclass]
#[pyo3(name = "InitialOscillatorFrequencyIR")]
#[derive(Clone)]
pub struct InitialOscillatorFrequencyPy(pub Arc<Mutex<IrNode>>);

#[pymethods]
impl InitialOscillatorFrequencyPy {
    #[new]
    #[pyo3(signature = (section, oscillators, values, interval=None))]
    pub fn new(
        section: String,
        oscillators: Py<PyList>,
        values: Vec<f64>,
        interval: Option<IntervalPy>,
    ) -> Self {
        InitialOscillatorFrequencyPy(Arc::new(Mutex::new(IrNode::InitialOscillatorFrequencyIr(
            InitialOscillatorFrequencyIr {
                interval: match interval {
                    Some(interval) => interval.0.clone(),
                    None => Arc::new(Mutex::new(IntervalIr::default())),
                },
                section,
                oscillators,
                values,
            },
        ))))
    }
}

impl_extractable_ir_node!(InitialOscillatorFrequencyPy);
impl_interval_methods!(InitialOscillatorFrequencyPy, InitialOscillatorFrequencyIr);
impl_python_dunders!(InitialOscillatorFrequencyPy, InitialOscillatorFrequencyIr);

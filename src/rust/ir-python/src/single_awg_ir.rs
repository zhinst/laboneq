// Copyright 2024 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use ir::single_awg_ir::SingleAwgIr;
use ir::{interval_ir::IntervalIr, IrNode};

use pyo3::exceptions::PySystemError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyType};

use std::sync::{Arc, Mutex};

use crate::{
    common::*, impl_extractable_ir_node, impl_interval_methods, impl_python_dunders,
    interval_ir::IntervalPy,
};
use ir::deep_copy_ir_node;

#[pyclass]
#[pyo3(name = "SingleAwgIR")]
#[derive(Clone)]
pub struct SingleAwgPy(pub Arc<Mutex<IrNode>>);

#[pymethods]
impl SingleAwgPy {
    #[new]
    #[pyo3(signature = (awg, interval=None))]
    pub fn new(awg: Py<PyAny>, interval: Option<IntervalPy>) -> Self {
        SingleAwgPy(Arc::new(Mutex::new(IrNode::SingleAwgIr(SingleAwgIr {
            interval: match interval {
                Some(interval) => interval.0.clone(),
                None => Arc::new(Mutex::new(IntervalIr::default())),
            },
            awg,
        }))))
    }

    #[classmethod]
    pub fn from_root_ir(_cls: &Bound<'_, PyType>, interval: IntervalPy, awg: Py<PyAny>) -> Self {
        SingleAwgPy(Arc::new(Mutex::new(IrNode::SingleAwgIr(SingleAwgIr {
            interval: interval.0.clone(),
            awg,
        }))))
    }
}

impl_extractable_ir_node!(SingleAwgPy);
impl_interval_methods!(SingleAwgPy, SingleAwgIr);
impl_python_dunders!(SingleAwgPy, SingleAwgIr);

// Copyright 2024 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use ir::{interval_ir::IntervalIr, IrNode};

use pyo3::{
    exceptions::{PyIndexError, PySystemError},
    prelude::*,
    types::PyList,
};

use std::{
    collections::HashSet,
    sync::{Arc, Mutex},
};

use crate::{
    common::*,
    loop_ir::LoopPy,
    loop_iteration_ir::{LoopIterationPreamblePy, LoopIterationPy},
    oscillator_ir::{InitialOscillatorFrequencyPy, SetOscillatorFrequencyPy},
    pulse_ir::PulsePy,
    section_ir::SectionPy,
    single_awg_ir::SingleAwgPy,
};

#[pyclass]
#[pyo3(name = "IntervalIR")]
#[derive(Clone)]
pub struct IntervalPy(pub Arc<Mutex<IntervalIr>>);

#[pymethods]
impl IntervalPy {
    #[new]
    #[pyo3(signature = (children=None, length=None, signals=HashSet::new(), children_start=Vec::new()))]
    fn new(
        children: Option<Bound<PyList>>,
        length: Option<i64>,
        signals: HashSet<String>,
        children_start: Vec<i64>,
    ) -> PyResult<Self> {
        let mut rust_children: Vec<Arc<Mutex<IrNode>>> = Vec::new();
        match &children {
            Some(children) => {
                for c_py in children {
                    let c_rs = extract_child(&c_py)?;
                    rust_children.push(c_rs);
                }
            }
            None => {}
        }
        Ok(IntervalPy(Arc::new(Mutex::new(IntervalIr {
            children: rust_children,
            length,
            signals,
            children_start,
        }))))
    }

    //------------------------------------------------------------------------------------
    // note(mr): dangerous interface, left here to demonstrate xfailing python test
    //           - the only way to fix this would be to wrap the children vector into
    //             Arc<Mutex<>>, then wrap this once more into a #[pyclass] in order to send
    //             it to python
    //
    //           - lets not do that since we excluded perfectly emulating python objects
    //             alternative interface below
    //
    //           - we probably need this after all though for the visitors
    #[getter]
    pub fn get_children<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let ret = PyList::empty_bound(py);

        let guard = unlock_mutex(&self.0)?;

        for c in &guard.children {
            let c_guard = c.lock().map_err(|_| {
                PyErr::new::<PySystemError, _>("Unable to lock mutex of shared python reference.")
            })?;
            match &*c_guard {
                IrNode::LoopIr(_) => ret.append(Py::new(py, LoopPy(Arc::clone(c))).unwrap())?,
                IrNode::SectionIr(_) => {
                    ret.append(Py::new(py, SectionPy(Arc::clone(c))).unwrap())?
                }
                IrNode::LoopIterationPreambleIr(_) => {
                    ret.append(Py::new(py, LoopIterationPreamblePy(Arc::clone(c))).unwrap())?
                }
                IrNode::LoopIterationIr(_) => {
                    ret.append(Py::new(py, LoopIterationPy(Arc::clone(c))).unwrap())?
                }
                IrNode::PulseIr(_) => ret.append(Py::new(py, PulsePy(Arc::clone(c))).unwrap())?,
                IrNode::InitialOscillatorFrequencyIr(_) => {
                    ret.append(Py::new(py, InitialOscillatorFrequencyPy(Arc::clone(c))).unwrap())?
                }
                IrNode::SetOscillatorFrequencyIr(_) => {
                    ret.append(Py::new(py, SetOscillatorFrequencyPy(Arc::clone(c))).unwrap())?
                }
                IrNode::SingleAwgIr(_) => {
                    ret.append(Py::new(py, SingleAwgPy(Arc::clone(c))).unwrap())?
                }
            }
        }

        Ok(ret)
    }

    #[setter]
    pub fn set_children(&mut self, children_py: &Bound<PyList>) -> PyResult<()> {
        let mut rust_children: Vec<Arc<Mutex<IrNode>>> = Vec::new();
        for c_py in children_py {
            let c_rs = extract_child(&c_py)?;
            rust_children.push(c_rs);
        }
        let mut guard = unlock_mutex(&self.0)?;
        guard.children = rust_children;
        Ok(())
    }
    //
    //------------------------------------------------------------------------------------

    //------------------------------------------------------------------------------------
    // note(mr): alternative, non-pythonic, but save interface
    //
    pub fn replace_children(&mut self, children_py: &Bound<PyList>) -> PyResult<()> {
        let mut rust_children: Vec<Arc<Mutex<IrNode>>> = Vec::new();
        for c_py in children_py {
            let c_rs = extract_child(&c_py)?;
            rust_children.push(c_rs);
        }
        let mut guard = unlock_mutex(&self.0)?;
        guard.children = rust_children;
        Ok(())
    }

    pub fn get_child_at(&self, idx: usize, py: Python) -> PyResult<Py<PyAny>> {
        let guard = unlock_mutex(&self.0)?;

        if idx >= guard.children.len() {
            return Err(PyIndexError::new_err(format!(
                "supplied index {} exceeds number of children {}",
                idx,
                guard.children.len()
            )));
        }

        let c_guard = guard.children[idx].lock().map_err(|_| {
            PyErr::new::<PySystemError, _>("Unable to lock mutex of shared python reference.")
        })?;
        match &*c_guard {
            IrNode::LoopIr(_) => Ok(Py::new(py, LoopPy(Arc::clone(&guard.children[idx])))
                .unwrap()
                .into_any()),
            IrNode::SectionIr(_) => Ok(Py::new(py, SectionPy(Arc::clone(&guard.children[idx])))
                .unwrap()
                .into_any()),
            IrNode::LoopIterationPreambleIr(_) => Ok(Py::new(
                py,
                LoopIterationPreamblePy(Arc::clone(&guard.children[idx])),
            )
            .unwrap()
            .into_any()),
            IrNode::LoopIterationIr(_) => Ok(Py::new(
                py,
                LoopIterationPy(Arc::clone(&guard.children[idx])),
            )
            .unwrap()
            .into_any()),
            IrNode::PulseIr(_) => Ok(Py::new(py, PulsePy(Arc::clone(&guard.children[idx])))
                .unwrap()
                .into_any()),
            IrNode::InitialOscillatorFrequencyIr(_) => Ok(Py::new(
                py,
                InitialOscillatorFrequencyPy(Arc::clone(&guard.children[idx])),
            )
            .unwrap()
            .into_any()),
            IrNode::SetOscillatorFrequencyIr(_) => Ok(Py::new(
                py,
                SetOscillatorFrequencyPy(Arc::clone(&guard.children[idx])),
            )
            .unwrap()
            .into_any()),
            IrNode::SingleAwgIr(_) => {
                Ok(Py::new(py, SingleAwgPy(Arc::clone(&guard.children[idx])))
                    .unwrap()
                    .into_any())
            }
        }
    }

    pub fn set_child_at(&mut self, idx: usize, child: &Bound<PyAny>) -> PyResult<()> {
        let mut guard = unlock_mutex(&self.0)?;
        if idx >= guard.children.len() {
            return Err(PyIndexError::new_err(format!(
                "supplied index {} exceeds number of children{}",
                idx,
                guard.children.len()
            )));
        }

        let c = extract_child(child)?;
        guard.children[idx] = c;

        Ok(())
    }
    //
    //------------------------------------------------------------------------------------
}

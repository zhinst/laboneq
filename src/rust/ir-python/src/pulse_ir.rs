// Copyright 2024 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use ir::pulse_ir::PulseIr;
use ir::{interval_ir::IntervalIr, IrNode};

use pyo3::exceptions::PySystemError;
use pyo3::prelude::*;
use pyo3::types::{PyComplex, PyDict, PyFloat, PyInt};

use std::sync::{Arc, Mutex};

use crate::{
    common::*, impl_extractable_ir_node, impl_interval_methods, impl_python_dunders,
    interval_ir::IntervalPy,
};
use ir::deep_copy_ir_node;

#[pyclass]
#[pyo3(name = "PulseIR")]
#[derive(Clone)]
pub struct PulsePy(pub Arc<Mutex<IrNode>>);

#[pymethods]
impl PulsePy {
    #[new]
    #[pyo3(signature = (pulse, amplitude, phase, offset, section, is_acquire, markers=None, amp_param_name=None, set_oscillator_phase=None, increment_oscillator_phase=None, incr_phase_param_name=None, play_pulse_params=None, pulse_pulse_params=None, interval=None))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        pulse: PyObject,
        amplitude: Py<PyAny>,
        phase: f64,
        offset: i64,
        section: String,
        is_acquire: bool,
        markers: Option<Py<PyAny>>,
        amp_param_name: Option<String>,
        set_oscillator_phase: Option<f64>,
        increment_oscillator_phase: Option<f64>,
        incr_phase_param_name: Option<String>,
        play_pulse_params: Option<Py<PyDict>>,
        pulse_pulse_params: Option<Py<PyDict>>,
        interval: Option<IntervalPy>,
        py: Python,
    ) -> Self {
        let mut complex = PyComplex::from_doubles_bound(py, 0.0, 0.0);
        if let Ok(py_float) = amplitude.downcast_bound::<PyFloat>(py) {
            let real = py_float.value();
            complex = PyComplex::from_doubles_bound(py, real, 0.0);
        }

        if let Ok(py_int) = amplitude.downcast_bound::<PyInt>(py) {
            let real: i64 = py_int.extract().unwrap();
            complex = PyComplex::from_doubles_bound(py, real as f64, 0.0);
        }

        if let Ok(py_complex) = amplitude.downcast_bound::<PyComplex>(py) {
            complex = PyComplex::from_doubles_bound(py, py_complex.real(), py_complex.imag());
        }

        PulsePy(Arc::new(Mutex::new(IrNode::PulseIr(PulseIr {
            interval: match interval {
                Some(interval) => interval.0.clone(),
                None => Arc::new(Mutex::new(IntervalIr::default())),
            },
            pulse,
            amplitude: complex.into(),
            amp_param_name,
            phase,
            offset,
            set_oscillator_phase,
            increment_oscillator_phase,
            incr_phase_param_name,
            section,
            play_pulse_params,
            pulse_pulse_params,
            is_acquire,
            markers,
        }))))
    }
}

impl_extractable_ir_node!(PulsePy);
impl_interval_methods!(PulsePy, PulseIr);
impl_python_dunders!(PulsePy, PulseIr);

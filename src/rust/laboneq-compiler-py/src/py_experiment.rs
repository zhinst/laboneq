// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use pyo3::prelude::*;
use std::sync::Arc;

use laboneq_ir::system::DeviceSetup;

use crate::experiment::Experiment;
use crate::experiment_context::ExperimentContext;
use crate::py_signal::{
    BounceCompensationPy, ExponentialCompensationPy, FirCompensationPy, HighPassCompensationPy,
    PrecompensationPy,
};
use crate::setup_processor::DelayRegistry;

#[pyclass(name = "Experiment", frozen)]
pub(crate) struct ExperimentPy {
    pub inner: Experiment,
    // NOTE: The usage of Arc here is to allow sharing the id_store across Python bindings
    // Remove when Python bindings are no longer needed
    pub device_setup: Arc<DeviceSetup>,
    pub context: ExperimentContext,
    /// Delay compensation for signals on devices.
    pub delay_compensation: DelayRegistry,
}

#[pymethods]
impl ExperimentPy {
    fn signal_delay_compensation(&self, signal_uid: &str) -> f64 {
        let uid = self.inner.id_store.get(signal_uid).unwrap().into();
        self.delay_compensation.signal_port_delay(uid).into()
    }

    fn signal_precompensation(
        &self,
        py: Python,
        signal_uid: &str,
    ) -> Option<Py<PrecompensationPy>> {
        let uid = self.inner.id_store.get(signal_uid).unwrap().into();
        if let Some(signal) = self.device_setup.signal_by_uid(&uid)
            && let Some(p) = &signal.precompensation
        {
            let precomp_py = PrecompensationPy {
                high_pass: p.high_pass.as_ref().map(|hp| {
                    Py::new(
                        py,
                        HighPassCompensationPy {
                            timeconstant: hp.timeconstant,
                        },
                    )
                    .unwrap()
                }),
                exponential: p
                    .exponential
                    .iter()
                    .map(|exp| {
                        Py::new(
                            py,
                            ExponentialCompensationPy {
                                timeconstant: exp.timeconstant,
                                amplitude: exp.amplitude,
                            },
                        )
                        .unwrap()
                    })
                    .collect(),
                fir: p.fir.as_ref().map(|fir| {
                    Py::new(
                        py,
                        FirCompensationPy {
                            coefficients: fir.coefficients.clone(),
                        },
                    )
                    .unwrap()
                }),
                bounce: p.bounce.as_ref().map(|bounce| {
                    Py::new(
                        py,
                        BounceCompensationPy {
                            delay: bounce.delay,
                            amplitude: bounce.amplitude,
                        },
                    )
                    .unwrap()
                }),
            };
            Some(Py::new(py, precomp_py).unwrap())
        } else {
            None
        }
    }
}

// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use laboneq_dsl::types::{OscillatorKind, ValueOrParameter};
use pyo3::prelude::*;
use std::sync::Arc;

use laboneq_common::compiler_settings::CompilerSettings;
use laboneq_ir::system::DeviceSetup;

use crate::compiler_backend::PreprocessedBackendData;
use crate::experiment::Experiment;
use crate::experiment_context::ExperimentContext;
use crate::py_signal::{
    BounceCompensationPy, ExponentialCompensationPy, FirCompensationPy, HighPassCompensationPy,
    PrecompensationPy,
};
use crate::setup_processor::DelayRegistry;

#[pyclass(name = "Experiment", frozen)]
pub struct ExperimentPy {
    pub(crate) inner: Experiment,
    // NOTE: The usage of Arc here is to allow sharing the id_store across Python bindings
    // Remove when Python bindings are no longer needed
    pub(crate) device_setup: Arc<DeviceSetup>,
    pub(crate) context: ExperimentContext,
    /// Delay compensation for signals on devices.
    pub(crate) delay_compensation: DelayRegistry,
    pub(crate) compiler_settings: CompilerSettings,
    pub(crate) backend_data: Arc<dyn PreprocessedBackendData + Send + Sync>,
}

#[pymethods]
impl ExperimentPy {
    fn signal_delay_compensation(&self, signal_uid: &str) -> f64 {
        let uid = self.inner.id_store.get(signal_uid).unwrap().into();
        self.delay_compensation.signal_port_delay(uid).into()
    }

    fn signal_sampling_rate(&self, signal_uid: &str) -> f64 {
        let uid = self.inner.id_store.get(signal_uid).unwrap().into();
        self.device_setup.signal_by_uid(&uid).unwrap().sampling_rate
    }

    fn signals(&self) -> Vec<String> {
        self.device_setup
            .signals()
            .map(|signal| self.inner.id_store.resolve(signal.uid).unwrap().to_string())
            .collect()
    }

    fn signal_device_uid(&self, signal_uid: &str) -> String {
        let uid = self.inner.id_store.get(signal_uid).unwrap().into();
        let signal = self.device_setup.signal_by_uid(&uid).unwrap();
        let device: &laboneq_ir::system::AwgDevice =
            self.device_setup.device_by_uid(&signal.device_uid).unwrap();
        self.inner
            .id_store
            .resolve(device.uid())
            .unwrap()
            .to_string()
    }

    fn device_lead_delay(&self, device_uid: &str) -> f64 {
        let uid = self.inner.id_store.get(device_uid).unwrap().into();
        self.delay_compensation.device_lead_delay(uid).into()
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

    fn signal_automute(&self, signal_uid: &str) -> bool {
        let uid = self.inner.id_store.get(signal_uid).unwrap().into();
        if let Some(signal) = self.device_setup.signal_by_uid(&uid) {
            signal.automute
        } else {
            false
        }
    }

    /// Return the hardware oscillator information for a signal, if it exists.
    ///
    /// The tuple contains:
    /// (oscillator_uid, fixed frequency, parameter_name)
    ///
    /// The frequency value is present only for fixed frequency oscillators.
    fn signal_hw_oscillator(
        &self,
        signal_uid: &str,
    ) -> Option<(String, Option<f64>, Option<String>)> {
        let uid = self.inner.id_store.get(signal_uid).unwrap().into();
        if let Some(signal) = self.device_setup.signal_by_uid(&uid)
            && let Some(oscillator) = &signal.oscillator
            && oscillator.kind == OscillatorKind::Hardware
        {
            let osc_uid = self
                .inner
                .id_store
                .resolve(oscillator.uid)
                .unwrap()
                .to_string();
            // Get only fixed frequency value, if it's a parameter, return None and the parameter name
            match &oscillator.frequency {
                ValueOrParameter::Value(val) => {
                    return Some((osc_uid, Some(*val), None));
                }
                ValueOrParameter::Parameter(param) => {
                    let param_string = self.inner.id_store.resolve(*param).unwrap().to_string();
                    return Some((osc_uid, None, Some(param_string)));
                }
                ValueOrParameter::ResolvedParameter { value: _, uid } => {
                    let param_string = self.inner.id_store.resolve(*uid).unwrap().to_string();
                    return Some((osc_uid, None, Some(param_string)));
                }
            };
        }
        None
    }
}

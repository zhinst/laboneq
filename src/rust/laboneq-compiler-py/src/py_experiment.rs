// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use pyo3::prelude::*;
use std::sync::Arc;

use laboneq_common::named_id::resolve_ids;
use laboneq_dsl::types::{OscillatorKind, ValueOrParameter};
use laboneq_py_utils::py_export::{acquisition_type_to_py, averaging_mode_to_py};

use laboneq_common::compiler_settings::CompilerSettings;
use laboneq_ir::system::DeviceSetup;

use crate::Error;
use crate::compiler_backend::{DynCompilerBackend, PreprocessedBackendData};
use crate::error::create_error_message;
use crate::experiment::Experiment;
use crate::experiment_context::ExperimentContext;
use crate::py_result_shape::HandleResultShapePy;
use crate::py_signal::{
    BounceCompensationPy, ExponentialCompensationPy, FirCompensationPy, HighPassCompensationPy,
    PrecompensationPy,
};
use crate::result_shape::ResultShapes;
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
    pub(crate) backend: Arc<dyn DynCompilerBackend>,
    pub(crate) backend_data: Arc<dyn PreprocessedBackendData + Send + Sync>,
    pub(crate) result_shapes: ResultShapes,
}

#[pymethods]
impl ExperimentPy {
    fn signal_delay_compensation(&self, signal_uid: &str) -> f64 {
        let uid = self.inner.id_store.get(signal_uid).unwrap().into();
        self.delay_compensation.signal_port_delay(uid).into()
    }

    fn device_lead_delay(&self, device_uid: &str) -> f64 {
        let uid = self.inner.id_store.get(device_uid).unwrap().into();
        self.delay_compensation.device_lead_delay(uid).into()
    }

    fn signal_precompensation(&self, py: Python, signal_uid: &str) -> Option<PrecompensationPy> {
        let uid = self.inner.id_store.get(signal_uid).unwrap().into();
        if let Some(signal) = self.device_setup.signal_by_uid(&uid)
            && let Some(p) = &signal.precompensation
        {
            let precomp_py = PrecompensationPy {
                high_pass: p.high_pass.as_ref().map(|hp| {
                    HighPassCompensationPy {
                        timeconstant: hp.timeconstant,
                    }
                    .into_pyobject(py)
                    .unwrap()
                    .unbind()
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
                    FirCompensationPy {
                        coefficients: fir.coefficients.clone(),
                        strict: fir.strict,
                    }
                    .into_pyobject(py)
                    .unwrap()
                    .unbind()
                }),
                bounce: p.bounce.as_ref().map(|bounce| {
                    BounceCompensationPy {
                        delay: bounce.delay,
                        amplitude: bounce.amplitude,
                    }
                    .into_pyobject(py)
                    .unwrap()
                    .unbind()
                }),
            };
            Some(precomp_py)
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

    fn get_result_shapes(
        &self,
        py: Python,
        combined_output: Bound<'_, PyAny>,
    ) -> PyResult<Vec<HandleResultShapePy>> {
        use crate::py_result_shape::create_result_shape_py;

        let id_store = &self.inner.id_store;

        // Collect raw acquisition lengths for all signal-handle pairs in the result shapes
        let raw_acquisition_lengths: PyResult<Vec<_>> = self
            .result_shapes
            .raw_acquisitions()
            .map(|(signal, handle)| -> PyResult<_> {
                let signal_str = id_store.resolve(signal).expect("Failed to resolve signal");
                let handle_str = id_store.resolve(handle).expect("Failed to resolve handle");
                let raw_acq_len: usize = combined_output
                    .call_method1("get_raw_acquire_length", (signal_str, handle_str))?
                    .extract()?;
                Ok((signal, handle, raw_acq_len))
            })
            .collect();

        // Get the result shapes for the collected acquisition lengths and convert them to Python objects
        self.result_shapes
            .get_shapes(raw_acquisition_lengths?.into_iter())
            .map_err(|e| {
                let msg = create_error_message(e);
                Error::new(resolve_ids(&msg, &self.inner.id_store))
            })?
            .into_iter()
            .map(|shape| create_result_shape_py(py, shape, id_store))
            .collect::<PyResult<Vec<_>>>()
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

    fn rt_loop_properties<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, RtLoopPropertiesPy>> {
        let rt_properties = self.context.rt_properties();

        let out = RtLoopPropertiesPy {
            uid: self
                .inner
                .id_store
                .resolve(rt_properties.uid)
                .unwrap()
                .to_string(),
            acquisition_type: acquisition_type_to_py(&rt_properties.acquisition_type).to_string(),
            averaging_mode: averaging_mode_to_py(&rt_properties.averaging_mode).to_string(),
            count: rt_properties.count.get(),
        };
        out.into_pyobject(py)
    }
}

#[pyclass(name = "RtLoopProperties", frozen)]
struct RtLoopPropertiesPy {
    #[pyo3(get)]
    uid: String,
    #[pyo3(get)]
    acquisition_type: String,
    #[pyo3(get)]
    averaging_mode: String,
    #[pyo3(get)]
    count: u32,
}

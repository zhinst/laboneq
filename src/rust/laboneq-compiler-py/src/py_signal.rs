// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use pyo3::prelude::*;

#[pyclass(name = "AmplifierPump", frozen)]
pub struct AmplifierPumpPy {
    pub device: String,
    pub channel: u16,
    pub pump_power: Py<PyAny>,
    pub pump_frequency: Py<PyAny>,
    pub probe_power: Py<PyAny>,
    pub probe_frequency: Py<PyAny>,
    pub cancellation_phase: Py<PyAny>,
    pub cancellation_attenuation: Py<PyAny>,
}

#[allow(clippy::too_many_arguments)]
#[pymethods]
impl AmplifierPumpPy {
    #[new]
    pub fn new(
        device: String,
        channel: u16,
        pump_power: Py<PyAny>,
        pump_frequency: Py<PyAny>,
        probe_power: Py<PyAny>,
        probe_frequency: Py<PyAny>,
        cancellation_phase: Py<PyAny>,
        cancellation_attenuation: Py<PyAny>,
    ) -> Self {
        Self {
            device,
            channel,
            pump_power,
            pump_frequency,
            probe_power,
            probe_frequency,
            cancellation_phase,
            cancellation_attenuation,
        }
    }
}

#[pyclass(name = "Precompensation", frozen)]
pub struct PrecompensationPy {
    #[pyo3(get)]
    pub high_pass: Option<Py<HighPassCompensationPy>>,
    #[pyo3(get)]
    pub exponential: Vec<Py<ExponentialCompensationPy>>,
    #[pyo3(get)]
    pub fir: Option<Py<FirCompensationPy>>,
    #[pyo3(get)]
    pub bounce: Option<Py<BounceCompensationPy>>,
}

#[pymethods]
impl PrecompensationPy {
    #[new]
    pub fn new(
        high_pass: Option<Py<HighPassCompensationPy>>,
        exponential: Vec<Py<ExponentialCompensationPy>>,
        fir: Option<Py<FirCompensationPy>>,
        bounce: Option<Py<BounceCompensationPy>>,
    ) -> Self {
        Self {
            high_pass,
            exponential,
            fir,
            bounce,
        }
    }
}

#[pyclass(name = "HighPassCompensation", frozen)]
pub struct HighPassCompensationPy {
    #[pyo3(get)]
    pub timeconstant: f64,
}

#[pymethods]
impl HighPassCompensationPy {
    #[new]
    pub fn new(timeconstant: f64) -> Self {
        Self { timeconstant }
    }
}

#[pyclass(name = "ExponentialCompensation", frozen)]
pub struct ExponentialCompensationPy {
    #[pyo3(get)]
    pub timeconstant: f64,
    #[pyo3(get)]
    pub amplitude: f64,
}

#[pymethods]
impl ExponentialCompensationPy {
    #[new]
    pub fn new(timeconstant: f64, amplitude: f64) -> Self {
        Self {
            timeconstant,
            amplitude,
        }
    }
}

#[pyclass(name = "FirCompensation", frozen)]
pub struct FirCompensationPy {
    #[pyo3(get)]
    pub coefficients: Vec<f64>,
}

#[pymethods]
impl FirCompensationPy {
    #[new]
    pub fn new(coefficients: Vec<f64>) -> Self {
        Self { coefficients }
    }
}

#[pyclass(name = "BounceCompensation", frozen)]
pub struct BounceCompensationPy {
    #[pyo3(get)]
    pub delay: f64,
    #[pyo3(get)]
    pub amplitude: f64,
}

#[pymethods]
impl BounceCompensationPy {
    #[new]
    pub fn new(delay: f64, amplitude: f64) -> Self {
        Self { delay, amplitude }
    }
}

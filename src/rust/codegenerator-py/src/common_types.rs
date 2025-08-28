// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use codegenerator::ir::compilation_job::{AwgKey, DeviceKind, MixerType, SignalKind};
use pyo3::prelude::*;

#[allow(clippy::upper_case_acronyms)]
#[pyclass(name = "SignalType", eq)]
#[derive(PartialEq, Clone)]
pub enum SignalTypePy {
    IQ,
    SINGLE,
    INTEGRATION,
}

impl SignalTypePy {
    pub fn from_signal_kind(signal_kind: &SignalKind) -> Self {
        match signal_kind {
            SignalKind::IQ => SignalTypePy::IQ,
            SignalKind::SINGLE => SignalTypePy::SINGLE,
            SignalKind::INTEGRATION => SignalTypePy::INTEGRATION,
        }
    }
}

#[allow(clippy::upper_case_acronyms)]
#[pyclass(name = "DeviceType", eq)]
#[derive(PartialEq, Clone)]
pub enum DeviceTypePy {
    HDAWG,
    SHFQA,
    SHFSG,
    UHFQA,
}

impl DeviceTypePy {
    pub fn from_device_kind(device_kind: &DeviceKind) -> Self {
        match device_kind {
            DeviceKind::HDAWG => DeviceTypePy::HDAWG,
            DeviceKind::SHFQA => DeviceTypePy::SHFQA,
            DeviceKind::SHFSG => DeviceTypePy::SHFSG,
            DeviceKind::UHFQA => DeviceTypePy::UHFQA,
        }
    }
}

#[allow(clippy::upper_case_acronyms)]
#[pyclass(name = "MixerType", eq)]
#[derive(PartialEq, Clone)]
pub enum MixerTypePy {
    /// Mixer performs full complex modulation
    IQ,
    /// Mixer only performs envelope modulation (UHFQA-style)
    UhfqaEnvelope,
}

impl MixerTypePy {
    pub fn from_mixer_type(mixer_type: &MixerType) -> Self {
        match mixer_type {
            MixerType::IQ => MixerTypePy::IQ,
            MixerType::UhfqaEnvelope => MixerTypePy::UhfqaEnvelope,
        }
    }
}

/// Python representation of the [`AwgKey`].
///
/// The class is interchangeable with the Python `AwgKey` class.
#[pyclass(name = "AwgKey", frozen)]
#[derive(PartialEq, Clone)]
pub struct AwgKeyPy {
    awg_key: AwgKey,
}

impl AwgKeyPy {
    pub fn new(awg_key: AwgKey) -> Self {
        Self { awg_key }
    }
}

impl From<AwgKey> for AwgKeyPy {
    fn from(awg_key: AwgKey) -> Self {
        AwgKeyPy::new(awg_key)
    }
}

#[pymethods]
impl AwgKeyPy {
    pub fn __repr__(&self) -> String {
        format!(
            "AwgKey(device_id='{}', awg_id={})",
            self.device_id(),
            self.awg_id()
        )
    }

    fn __eq__(&self, other: &Bound<PyAny>) -> PyResult<bool> {
        let device_id: String = other.getattr("device_id")?.extract()?;
        let awg_id: i32 = other.getattr("awg_id")?.extract()?;
        Ok(self.device_id() == device_id && self.awg_id() == awg_id)
    }

    fn __hash__(&self) -> PyResult<isize> {
        // Use Python's built-in hash function for tuples
        // to ensure consistent hashing behavior with Python `AwgKey`.
        Python::with_gil(|py| (self.device_id(), self.awg_id()).into_pyobject(py)?.hash())
    }

    #[getter]
    pub fn device_id(&self) -> &str {
        self.awg_key.device_name()
    }

    #[getter]
    pub fn awg_id(&self) -> i32 {
        self.awg_key.index().into()
    }
}

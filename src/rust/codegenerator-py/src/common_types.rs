// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use codegenerator::ir::compilation_job::{DeviceKind, MixerType, SignalKind};
use pyo3::prelude::*;

#[allow(clippy::upper_case_acronyms)]
#[pyclass(name = "SignalType", eq)]
#[derive(PartialEq, Clone)]
pub(crate) enum SignalTypePy {
    IQ,
    SINGLE,
    INTEGRATION,
}

impl SignalTypePy {
    pub(crate) fn from_signal_kind(signal_kind: &SignalKind) -> Self {
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
pub(crate) enum DeviceTypePy {
    HDAWG,
    SHFQA,
    SHFSG,
    UHFQA,
}

impl DeviceTypePy {
    pub(crate) fn from_device_kind(device_kind: &DeviceKind) -> Self {
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
pub(crate) enum MixerTypePy {
    /// Mixer performs full complex modulation
    IQ,
    /// Mixer only performs envelope modulation (UHFQA-style)
    UhfqaEnvelope,
}

impl MixerTypePy {
    pub(crate) fn from_mixer_type(mixer_type: &MixerType) -> Self {
        match mixer_type {
            MixerType::IQ => MixerTypePy::IQ,
            MixerType::UhfqaEnvelope => MixerTypePy::UhfqaEnvelope,
        }
    }
}

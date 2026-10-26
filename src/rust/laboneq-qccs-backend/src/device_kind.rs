// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

//! QCCS arbitrary waveform generator (AWG) models.

use std::fmt::Display;

use codegenerator_py::DeviceKind as CodegenDeviceKind;
use laboneq_common::device_traits;
use laboneq_common::device_traits::DeviceTraits;
use laboneq_units::duration::{Frequency, Hertz};

use crate::device_traits::{HDAWG_TRAITS, SHFQA_TRAITS, SHFSG_TRAITS, UHFQA_TRAITS};

/// Device model that can have signals or operations performed on it.
#[derive(Debug, Clone, Eq, PartialEq, Copy, Hash)]
pub enum DeviceKind {
    Hdawg,
    Shfqa,
    Shfsg,
    Uhfqa,
}

impl DeviceKind {
    pub fn is_qa_device(&self) -> bool {
        matches!(self, Self::Shfqa | Self::Uhfqa)
    }

    /// Whether the model is part of the SHF family.
    pub(crate) fn is_shf(&self) -> bool {
        matches!(self, Self::Shfqa | Self::Shfsg)
    }

    /// The rate at which a device of this model plays samples. An HDAWG is clocked
    /// down when it shares the setup with an SHF device.
    pub(crate) fn sampling_rate(&self, setup_has_shf: bool) -> Frequency<Hertz> {
        match self {
            DeviceKind::Shfsg => device_traits::SHFSG_SAMPLING_RATE,
            DeviceKind::Shfqa => device_traits::SHFQA_SAMPLING_RATE,
            DeviceKind::Hdawg if setup_has_shf => device_traits::HDAWG_SAMPLING_RATE_WITH_SHF,
            DeviceKind::Hdawg => device_traits::HDAWG_SAMPLING_RATE_WITHOUT_SHF,
            DeviceKind::Uhfqa => device_traits::UHFQA_SAMPLING_RATE,
        }
    }

    /// What a device of this model can do.
    pub(crate) fn traits(&self) -> &'static DeviceTraits {
        match self {
            DeviceKind::Hdawg => &HDAWG_TRAITS,
            DeviceKind::Shfqa => &SHFQA_TRAITS,
            DeviceKind::Shfsg => &SHFSG_TRAITS,
            DeviceKind::Uhfqa => &UHFQA_TRAITS,
        }
    }

    /// The model as the code generator names it.
    pub(crate) fn for_codegen(&self) -> CodegenDeviceKind {
        match self {
            DeviceKind::Hdawg => CodegenDeviceKind::HDAWG,
            DeviceKind::Shfqa => CodegenDeviceKind::SHFQA,
            DeviceKind::Shfsg => CodegenDeviceKind::SHFSG,
            DeviceKind::Uhfqa => CodegenDeviceKind::UHFQA,
        }
    }
}

impl Display for DeviceKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let out = match self {
            DeviceKind::Hdawg => "HDAWG",
            DeviceKind::Shfqa => "SHFQA",
            DeviceKind::Shfsg => "SHFSG",
            DeviceKind::Uhfqa => "UHFQA",
        };
        write!(f, "{}", out)
    }
}

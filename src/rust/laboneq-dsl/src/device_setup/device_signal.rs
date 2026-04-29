// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use crate::signal_calibration::SignalCalibration;
use crate::types::{DeviceUid, SignalUid};
use laboneq_common::types::SignalKind;

/// Device signal definition, representing a signal in the device setup.
#[derive(Debug, Clone, PartialEq)]
pub struct DeviceSignal {
    // Identification parameters
    pub uid: SignalUid,
    pub device_uid: DeviceUid,

    // Configuration parameters
    pub ports: Vec<String>,
    pub kind: SignalKind,
    pub calibration: SignalCalibration,
}

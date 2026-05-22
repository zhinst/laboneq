// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use crate::signal_calibration::SignalCalibration;
use crate::types::SignalUid;

#[derive(Debug, Clone, PartialEq)]
pub struct ExperimentSignal {
    pub uid: SignalUid,
    pub maps_to: String,
    pub calibration: SignalCalibration,
}

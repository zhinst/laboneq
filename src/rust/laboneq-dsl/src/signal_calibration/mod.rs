// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use crate::types::Quantity;

mod precompensation;

pub use precompensation::*;

pub struct SignalCalibration {
    pub range: Quantity,
    pub precompensation: Option<Precompensation>,
}

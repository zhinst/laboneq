// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use laboneq_units::duration::{Duration, Second};

use crate::types::PulseUid;

#[derive(Debug, Clone, PartialEq, Copy)]
pub enum MarkerSelector {
    M1,
    M2,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Marker {
    pub marker_selector: MarkerSelector,
    pub enable: bool,
    pub start: Option<Duration<Second, f64>>,
    pub length: Option<Duration<Second, f64>>,
    pub pulse_id: Option<PulseUid>,
}

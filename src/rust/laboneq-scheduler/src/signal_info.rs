// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

//! This module defines signal information that is required by
//! the Scheduler.
use crate::ir::SignalUid;
use laboneq_common::types::AwgKey;

pub trait SignalInfo {
    fn uid(&self) -> SignalUid;
    fn awg_key(&self) -> AwgKey;
    fn sampling_rate(&self) -> f64;
}

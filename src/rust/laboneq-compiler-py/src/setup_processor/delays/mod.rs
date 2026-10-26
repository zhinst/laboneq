// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

//! Lining the signals up in time.
//!
//! Signals reach the compiler carrying delays that their backend has already
//! accounted for, so left alone they would start at different times. This
//! module works out how long each one has to wait for the others.

mod calculator;
mod on_device;

#[cfg(test)]
mod tests;

pub(crate) use calculator::{DelayRegistry, SignalDelayProperties, compute_signal_delays};

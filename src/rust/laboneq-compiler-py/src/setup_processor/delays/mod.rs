// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

mod calculator;
mod on_device;

#[cfg(test)]
mod tests;

pub(crate) use calculator::{DelayRegistry, SignalDelayProperties, compute_signal_delays};

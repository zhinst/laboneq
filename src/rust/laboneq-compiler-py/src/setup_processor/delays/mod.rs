// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

mod calculator;
mod lead_delay;
mod on_device;
mod output_routing;
mod precompensation;

#[cfg(test)]
mod tests;

pub(crate) use calculator::{DelayRegistry, compute_delays};

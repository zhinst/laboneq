// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

mod sampling;

pub use sampling::{PulseRequest, sample_pulses};

#[cfg(test)]
mod tests;

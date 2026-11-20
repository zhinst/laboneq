// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

mod dispatch;
mod shape;

pub(crate) use dispatch::{evaluation_points, for_each_envelope};
pub use shape::{EnvelopeShape, FunctionalShape};

#[cfg(test)]
mod tests;

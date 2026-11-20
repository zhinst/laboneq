// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

mod carrier;
mod check;
mod fold;

pub(crate) use carrier::Carriers;
pub(crate) use check::{Peak, SampleCheck};
pub(crate) use fold::modulate_envelope;

#[cfg(test)]
mod tests;

// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use numeric_array::NumericArray;

use crate::builtin::BuiltinPulse;

#[derive(Debug, Clone, Copy)]
pub enum EnvelopeShape<'a, P> {
    Functional(FunctionalShape<P>),
    Sampled(&'a NumericArray),
}

/// A functional pulse of `length` samples, described to an
/// [`ExternalSampler`](crate::external::ExternalSampler) by `pulse`.
/// A `builtin` one is a standard pulse the crate may evaluate itself.
#[derive(Debug, Clone, Copy)]
pub struct FunctionalShape<P> {
    pub pulse: P,
    pub builtin: Option<BuiltinPulse>,
    pub length: usize,
}

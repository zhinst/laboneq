// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::ops::Range;

use num_complex::Complex64;

use laboneq_error::LabOneQError;

use crate::envelope::{self, FunctionalShape};
use crate::samples::SamplesRef;

/// Evaluates functional pulse envelopes not supported by this crate, such as
/// externally registered pulse functions, each described by a `P`.
/// Results are expected to be deterministic.
pub trait ExternalSampler<P>: Send + Sync {
    /// Evaluates every request and hands each envelope onto `f` with the request's `index`.
    /// Returns an error if an evaluated envelope is shorter than its request's slice.
    fn for_each_envelope(
        &self,
        requests: &[ExternalRequest<'_, P>],
        sampling_rate: f64,
        f: &mut EnvelopeFn<'_>,
    ) -> Result<(), LabOneQError>;
}

/// A functional pulse envelope for an [`ExternalSampler`] to evaluate.
#[derive(Debug, Clone)]
pub struct ExternalRequest<'r, P> {
    /// Index of the [`PulseRequest`](crate::PulseRequest) to evaluate.
    pub index: usize,
    pub shape: &'r FunctionalShape<P>,
    /// Pulse samples to evaluate.
    pub slice: Range<usize>,
    /// Pulse amplitude, which pulse functions receive as an argument.
    pub amplitude: Complex64,
}

impl<P> ExternalRequest<'_, P> {
    /// Returns the points to evaluate the pulse function at,
    /// scaled to the [-1, 1) range over the pulse length.
    pub fn evaluation_points(&self, sampling_rate: f64) -> impl Iterator<Item = f64> {
        envelope::evaluation_points(self.shape.length, sampling_rate, self.slice.clone())
    }
}

/// Receives a pulse envelope with the index of its [`PulseRequest`](crate::PulseRequest).
pub type EnvelopeFn<'f> = dyn FnMut(usize, SamplesRef<'_>) -> Result<(), LabOneQError> + 'f;

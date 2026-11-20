// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::ops::Range;

use laboneq_error::LabOneQError;
use numeric_array::NumericArray;

use crate::builtin::{self, Parameters, Scratch};
use crate::external::{EnvelopeFn, ExternalRequest, ExternalSampler};
use crate::pulse::PulseRequest;
use crate::samples::Samples;

use super::EnvelopeShape;

/// Evaluates the envelope of every pulse, handing each onto `f` with its index.
/// Pulses this crate does not support are passed onto `external`.
pub(crate) fn for_each_envelope<P>(
    pulses: &[PulseRequest<'_, P>],
    sampling_rate: f64,
    external: &dyn ExternalSampler<P>,
    f: &mut EnvelopeFn<'_>,
) -> Result<(), LabOneQError> {
    let mut unevaluated = Vec::new();
    let mut scratch = Scratch::default();
    for (index, pulse) in pulses.iter().enumerate() {
        unevaluated.extend(evaluate(index, pulse, sampling_rate, &mut scratch, f)?);
    }
    if !unevaluated.is_empty() {
        external.for_each_envelope(&unevaluated, sampling_rate, f)?;
    }
    Ok(())
}

/// Hands the envelope of `pulse` onto `f`, if this crate supports it.
/// Returns an [`ExternalRequest`] for an [`ExternalSampler`] otherwise.
fn evaluate<'r, P>(
    index: usize,
    pulse: &'r PulseRequest<'_, P>,
    sampling_rate: f64,
    scratch: &mut Scratch,
    f: &mut EnvelopeFn<'_>,
) -> Result<Option<ExternalRequest<'r, P>>, LabOneQError> {
    let shape = match &pulse.shape {
        EnvelopeShape::Functional(shape) => shape,
        EnvelopeShape::Sampled(samples) => {
            debug_assert_eq!(pulse.slice.start, 0, "sampled pulses are not sliced");
            sampled_envelope(index, samples, f)?;
            return Ok(None);
        }
    };
    let parameters = pulse.parameters.unwrap_or(&Parameters::NONE);
    let envelope = shape.builtin.and_then(|builtin| {
        let slice = pulse.slice.clone();
        let length = shape.length;
        builtin::builtin_envelope(builtin, length, slice, parameters, sampling_rate, scratch)
    });
    Ok(match envelope {
        Some(envelope) => {
            f(index, envelope)?;
            None
        }
        None => Some(ExternalRequest {
            index,
            shape,
            slice: pulse.slice.clone(),
            amplitude: pulse.amplitude,
        }),
    })
}

/// Returns the evaluation points defined by a `slice` for a functional pulse
/// of `length` samples. Evaluation points are scaled to the [-1, 1) range over
/// the pulse length.
pub(crate) fn evaluation_points(
    length: usize,
    sampling_rate: f64,
    slice: Range<usize>,
) -> impl Iterator<Item = f64> {
    let length_seconds = length as f64 / sampling_rate;
    // `round` in Python rounds half to even.
    let length_samples = (length_seconds * sampling_rate).round_ties_even();
    slice.map(move |i| i as f64 / length_samples * 2.0 - 1.0)
}

/// Hands the samples of a sampled pulse onto `f`.
fn sampled_envelope(
    index: usize,
    samples: &NumericArray,
    f: &mut EnvelopeFn<'_>,
) -> Result<(), LabOneQError> {
    match samples {
        NumericArray::Float64(samples) => f(index, Samples::Real(samples)),
        NumericArray::Complex64(samples) => f(index, Samples::Complex(samples)),
        NumericArray::Integer64(samples) => {
            let envelope: Vec<_> = samples.iter().map(|&s| s as f64).collect();
            f(index, Samples::Real(&envelope))
        }
    }
}

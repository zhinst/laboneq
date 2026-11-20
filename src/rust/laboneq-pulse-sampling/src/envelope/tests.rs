// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use num_complex::Complex64;

use laboneq_error::LabOneQError;
use numeric_array::NumericArray;

use crate::builtin::BuiltinPulse;
use crate::external::{EnvelopeFn, ExternalRequest, ExternalSampler};
use crate::pulse::PulseRequest;
use crate::samples::{Samples, SamplesRef};

use super::*;

#[test]
fn test_supported_pulses_are_consumed_in_order_and_others_passed_on() {
    let samples = NumericArray::Float64(vec![0.5, 0.25]);
    let pulses = [
        pulse(EnvelopeShape::Sampled(&samples)),
        pulse(custom()),
        pulse(constant()),
        pulse(custom()),
        pulse(constant()),
        pulse(drag()),
    ];

    let mut consumed = vec![];
    for_each_envelope(&pulses, 2e9, &EmptyExternal, &mut |index, envelope| {
        consumed.push((index, owned(envelope)));
        Ok(())
    })
    .unwrap();

    let expected = [
        (0, Samples::Real(vec![0.5, 0.25])),
        (2, Samples::Real(vec![1.0])),
        (4, Samples::Real(vec![1.0])),
    ];
    assert_eq!(consumed[..3], expected);
    assert!(matches!(consumed[3], (5, Samples::Complex(_))));
    let external = [1, 3].map(|idx| (idx, Samples::Real(vec![])));
    assert_eq!(consumed[4..], external);
}

#[test]
fn test_integer_samples_are_converted() {
    let samples = NumericArray::Integer64(vec![2, -1, 3]);
    let mut pulse = pulse(EnvelopeShape::Sampled(&samples));
    pulse.slice = 0..2;

    let mut consumed = vec![];
    for_each_envelope(&[pulse], 2e9, &EmptyExternal, &mut |_, envelope| {
        let Samples::Real(samples) = envelope else {
            panic!("integer samples become real samples");
        };
        consumed.extend_from_slice(samples);
        Ok(())
    })
    .unwrap();

    assert_eq!(consumed, [2.0, -1.0, 3.0]);
}

/// Hands every request back with an empty envelope.
struct EmptyExternal;

impl ExternalSampler<()> for EmptyExternal {
    fn for_each_envelope(
        &self,
        requests: &[ExternalRequest<'_, ()>],
        _sampling_rate: f64,
        f: &mut EnvelopeFn<'_>,
    ) -> Result<(), LabOneQError> {
        requests
            .iter()
            .try_for_each(|request| f(request.index, Samples::Real(&[])))
    }
}

fn pulse(shape: EnvelopeShape<'_, ()>) -> PulseRequest<'_, ()> {
    PulseRequest {
        shape,
        slice: 0..1,
        amplitude: Complex64::new(0.8, 0.0),
        parameters: None,
        phase: 0.0,
        modulation_frequency: 0.0,
        target: 0,
        offset: 0,
    }
}

fn constant() -> EnvelopeShape<'static, ()> {
    functional(Some(BuiltinPulse::Const))
}

fn drag() -> EnvelopeShape<'static, ()> {
    functional(Some(BuiltinPulse::Drag))
}

fn custom() -> EnvelopeShape<'static, ()> {
    functional(None)
}

fn functional(builtin: Option<BuiltinPulse>) -> EnvelopeShape<'static, ()> {
    EnvelopeShape::Functional(FunctionalShape {
        pulse: (),
        builtin,
        length: 1,
    })
}

fn owned(envelope: SamplesRef<'_>) -> Samples<Vec<f64>, Vec<Complex64>> {
    match envelope {
        Samples::Real(samples) => Samples::Real(samples.to_vec()),
        Samples::Complex(samples) => Samples::Complex(samples.to_vec()),
    }
}

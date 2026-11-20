// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use num_complex::Complex64;

use laboneq_error::LabOneQError;

use crate::builtin::BuiltinPulse;
use crate::envelope::{EnvelopeShape, FunctionalShape};
use crate::external::{EnvelopeFn, ExternalRequest, ExternalSampler};
use crate::samples::Samples;

use super::*;

#[test]
fn test_pulses_are_added_into_their_targets_at_their_offsets() {
    let mut first = [0.0; 4];
    let mut second = [0.0; 2];
    let pulses = [
        pulse(constant(), 0.5, 0, 1),
        pulse(constant(), 0.25, 1, 0),
        pulse(constant(), 0.5, 0, 2),
    ];
    let targets = &mut [
        Samples::Real(&mut first[..]),
        Samples::Real(&mut second[..]),
    ];

    sample_pulses(&pulses, targets, 2e9, &OnesExternal, None).unwrap();

    assert_eq!(first, [0.0, 0.5, 1.0, 0.5]);
    assert_eq!(second, [0.25, 0.25]);
}

#[test]
fn test_the_first_pulse_in_order_above_the_magnitude_limit_is_reported() {
    let mut samples = [0.0; 2];
    // The externally evaluated pulse is added last, yet comes first in order.
    let pulses = [
        pulse(custom(), 1.5, 0, 0),
        pulse(constant(), 2.0, 0, 0),
        pulse(constant(), 0.5, 0, 0),
    ];
    let targets = &mut [Samples::Real(&mut samples[..])];

    let exceeding = sample_pulses(&pulses, targets, 2e9, &OnesExternal, Some(1.0)).unwrap();

    assert_eq!(exceeding, Some((0, 1.5)));
    assert_eq!(samples, [4.0, 4.0]);
}

#[test]
fn test_complex_peaks_are_magnitudes() {
    let mut samples = [Complex64::ZERO];
    let pulses = [pulse(constant(), 0.0, 0, 0)].map(|mut pulse| {
        pulse.slice = 0..1;
        pulse.amplitude = Complex64::new(0.6, 0.8);
        pulse
    });
    let targets = &mut [Samples::Complex(&mut samples[..])];

    let within = sample_pulses(&pulses, targets, 2e9, &OnesExternal, Some(1.0)).unwrap();
    let above = sample_pulses(&pulses, targets, 2e9, &OnesExternal, Some(0.99)).unwrap();

    assert_eq!(within, None);
    assert!(matches!(above, Some((0, peak)) if (peak - 1.0).abs() < 1e-12));
}

#[test]
fn test_pulses_are_unchecked_without_a_magnitude_limit() {
    let mut samples = [0.0; 2];
    let pulses = [pulse(constant(), 2.0, 0, 0)];
    let targets = &mut [Samples::Real(&mut samples[..])];

    let exceeding = sample_pulses(&pulses, targets, 2e9, &OnesExternal, None).unwrap();

    assert_eq!(exceeding, None);
    assert_eq!(samples, [2.0, 2.0]);
}

#[test]
#[should_panic]
fn test_a_pulse_beyond_its_target_panics() {
    let mut samples = [0.0; 2];
    let pulses = [pulse(constant(), 1.0, 0, 1)];
    let targets = &mut [Samples::Real(&mut samples[..])];

    let _ = sample_pulses(&pulses, targets, 2e9, &OnesExternal, None);
}

/// Evaluates every request to an envelope of ones.
struct OnesExternal;

impl ExternalSampler<()> for OnesExternal {
    fn for_each_envelope(
        &self,
        requests: &[ExternalRequest<'_, ()>],
        _sampling_rate: f64,
        f: &mut EnvelopeFn<'_>,
    ) -> Result<(), LabOneQError> {
        requests.iter().try_for_each(|request| {
            let ones = vec![1.0; request.slice.len()];
            f(request.index, Samples::Real(&ones))
        })
    }
}

/// Returns a two-sample pulse of `amplitude`, added into `target` from `offset`.
fn pulse(
    shape: EnvelopeShape<'_, ()>,
    amplitude: f64,
    target: usize,
    offset: usize,
) -> PulseRequest<'_, ()> {
    PulseRequest {
        shape,
        slice: 0..2,
        amplitude: Complex64::new(amplitude, 0.0),
        parameters: None,
        phase: 0.0,
        modulation_frequency: 0.0,
        target,
        offset,
    }
}

fn constant() -> EnvelopeShape<'static, ()> {
    functional(Some(BuiltinPulse::Const))
}

fn custom() -> EnvelopeShape<'static, ()> {
    functional(None)
}

fn functional(builtin: Option<BuiltinPulse>) -> EnvelopeShape<'static, ()> {
    EnvelopeShape::Functional(FunctionalShape {
        pulse: (),
        builtin,
        length: 2,
    })
}

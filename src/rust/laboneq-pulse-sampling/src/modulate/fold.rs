// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use num_complex::Complex64;

use crate::samples::{Samples, SamplesMut, SamplesRef};

use super::carrier::{BLOCK, Modulation};
use super::check::SampleCheck;

/// Tolerance for vanishing imaginary parts.
const IMAGINARY_TOLERANCE: f64 = 1e-8;

/// Applies a `modulation` to a precomputed `envelope`, scaled by `amplitude`, and folds
/// (accumulates) the result into a `target` buffer. The `n`-th pulse sample is described by
/// `amplitude * envelope[n] * modulation[n]`.
///
/// Each pulse sample passes through a `check` before being added. A real `target` keeps
/// the real part and logs when a non-vanishing imaginary part is dropped.
/// `envelope` must be at least as long as `target`; a longer `envelope` is cut short.
pub(crate) fn modulate_envelope(
    target: SamplesMut<'_>,
    envelope: SamplesRef<'_>,
    amplitude: Complex64,
    modulation: &Modulation<'_>,
    check: &mut impl SampleCheck,
) {
    assert!(
        envelope.len() >= target.len(),
        "Found a pulse envelope with {} samples, but {} are played",
        envelope.len(),
        target.len()
    );
    if fold_pulse(target, envelope, amplitude, modulation, check) {
        if matches!(modulation, Modulation::Carrier { .. }) {
            let msg = "Complex-valued pulse envelope provided for an rf-signal, \
                imaginary part will be dropped after performing software modulation.";
            log::debug!("{msg}");
        } else {
            let msg = "Complex-valued pulse envelope provided for an rf-signal, imaginary part will be dropped.";
            log::warn!("{msg}");
        }
    }
}

/// Accumulates a modulated pulse `envelope` into a `target` buffer and
/// returns whether a real `target` dropped a non-vanishing imaginary part.
pub(super) fn fold_pulse(
    target: SamplesMut<'_>,
    envelope: SamplesRef<'_>,
    amplitude: Complex64,
    modulation: &Modulation,
    check: &mut impl SampleCheck,
) -> bool {
    let mut imaginary_part_dropped = false;
    let add_real = |sample: &mut f64, value: Complex64| {
        imaginary_part_dropped |= value.im.abs() > IMAGINARY_TOLERANCE || value.im.is_nan();
        check.real(value.re);
        *sample += value.re;
    };
    match (target, envelope) {
        (Samples::Real(target), Samples::Real(envelope))
            if amplitude.im == 0.0 && matches!(modulation, Modulation::None) =>
        {
            let amplitude = amplitude.re;
            for (sample, &e) in target.iter_mut().zip(envelope) {
                let value = amplitude * e;
                check.real(value);
                *sample += value;
            }
        }
        (Samples::Real(target), Samples::Real(envelope)) => {
            let values = envelope.iter().map(|&e| amplitude * Complex64::new(e, 0.0));
            fold(target, values, modulation, add_real);
        }
        (Samples::Real(target), Samples::Complex(envelope)) => {
            let values = envelope.iter().map(|&e| amplitude * e);
            fold(target, values, modulation, add_real);
        }
        (Samples::Complex(target), Samples::Real(envelope)) => {
            let values = envelope.iter().map(|&e| amplitude * Complex64::new(e, 0.0));
            fold(target, values, modulation, |sample, value| {
                check.complex(value);
                *sample += value;
            });
        }
        (Samples::Complex(target), Samples::Complex(envelope)) => {
            let values = envelope.iter().map(|&e| amplitude * e);
            fold(target, values, modulation, |sample, value| {
                check.complex(value);
                *sample += value;
            });
        }
    };
    imaginary_part_dropped
}

/// Modulates envelope `values` and hands them onto `add` with their `target` sample buffer.
#[inline(always)]
fn fold<T>(
    target: &mut [T],
    mut values: impl Iterator<Item = Complex64>,
    modulation: &Modulation,
    mut add: impl FnMut(&mut T, Complex64),
) {
    match modulation {
        Modulation::None => {
            for (sample, value) in target.iter_mut().zip(values) {
                add(sample, value);
            }
        }
        Modulation::Phase(phase) => {
            for (sample, value) in target.iter_mut().zip(values) {
                add(sample, phase * value);
            }
        }
        Modulation::Carrier {
            table,
            angular_frequency,
            phase,
            sampling_rate,
        } => {
            for (m, block) in target.chunks_mut(BLOCK).enumerate() {
                // every `BLOCK`th phasor
                let base = Complex64::cis(
                    -(angular_frequency * ((m * BLOCK) as f64 / sampling_rate) + phase),
                );
                for ((sample, value), carrier) in block.iter_mut().zip(&mut values).zip(*table) {
                    add(sample, base * carrier * value);
                }
            }
        }
    }
}

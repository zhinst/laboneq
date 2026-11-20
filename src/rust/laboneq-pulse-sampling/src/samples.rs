// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::ops::Range;

use num_complex::Complex64;

/// Real or complex samples, owned or borrowed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Samples<R, C> {
    Real(R),
    Complex(C),
}

pub type SamplesRef<'a> = Samples<&'a [f64], &'a [Complex64]>;

pub type SamplesMut<'a> = Samples<&'a mut [f64], &'a mut [Complex64]>;

impl<R: AsRef<[f64]>, C: AsRef<[Complex64]>> Samples<R, C> {
    pub fn len(&self) -> usize {
        match self {
            Samples::Real(samples) => samples.as_ref().len(),
            Samples::Complex(samples) => samples.as_ref().len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn as_samples_ref(&self) -> SamplesRef<'_> {
        match self {
            Samples::Real(samples) => Samples::Real(samples.as_ref()),
            Samples::Complex(samples) => Samples::Complex(samples.as_ref()),
        }
    }
}

impl<R: AsMut<[f64]>, C: AsMut<[Complex64]>> Samples<R, C> {
    pub fn as_samples_mut(&mut self) -> SamplesMut<'_> {
        match self {
            Samples::Real(samples) => Samples::Real(samples.as_mut()),
            Samples::Complex(samples) => Samples::Complex(samples.as_mut()),
        }
    }

    /// Returns a mutable view of the samples in `range`.
    pub(crate) fn slice_mut(&mut self, range: Range<usize>) -> SamplesMut<'_> {
        match self {
            Samples::Real(samples) => Samples::Real(&mut samples.as_mut()[range]),
            Samples::Complex(samples) => Samples::Complex(&mut samples.as_mut()[range]),
        }
    }
}

// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use num_complex::Complex64;

/// Inspects each pulse sample prior to accumulation into the target buffer.
pub(crate) trait SampleCheck {
    fn real(&mut self, value: f64);
    fn complex(&mut self, value: Complex64);
}

/// Checks nothing.
impl SampleCheck for () {
    fn real(&mut self, _value: f64) {}

    fn complex(&mut self, _value: Complex64) {}
}

/// Peak sample magnitude: |x| for real samples, sqrt(I^2+Q^2) for complex ones.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct Peak {
    abs: f64,
    norm_sqr: f64,
}

impl Peak {
    pub(crate) fn exceeds(&self, limit: f64) -> bool {
        self.abs > limit || self.norm_sqr > limit.powi(2)
    }

    pub(crate) fn magnitude(&self) -> f64 {
        self.abs.max(self.norm_sqr.sqrt())
    }
}

impl SampleCheck for Peak {
    fn real(&mut self, value: f64) {
        self.abs = self.abs.max(value.abs());
    }

    fn complex(&mut self, value: Complex64) {
        self.norm_sqr = self.norm_sqr.max(value.norm_sqr());
    }
}

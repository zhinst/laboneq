// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::f64::consts::PI;

use num_complex::Complex64;
use rustc_hash::FxHashMap;

/// Pulse-sample modulation.
pub(crate) enum Modulation<'t> {
    None,
    /// Baseband carrier, phase-offset only.
    Phase(Complex64),
    /// Carrier at `angular_frequency` in rad/s, offset by `phase`
    /// in radians, with its first `BLOCK` samples in `table`.
    Carrier {
        table: &'t [Complex64; BLOCK],
        angular_frequency: f64,
        phase: f64,
        sampling_rate: f64,
    },
}

/// Samples per carrier block, modulated from one phasor and a precomputed
/// shared table. A pulse of `N` samples costs about `N / BLOCK` trigonometric
/// calls, and each pulse frequency `BLOCK` additional calls for its table.
pub(super) const BLOCK: usize = 64;

/// Modulations at one sampling rate, with the carrier tables of the frequencies seen so far.
pub(crate) struct Carriers {
    sampling_rate: f64,
    tables: FxHashMap<u64, [Complex64; BLOCK]>,
}

impl Carriers {
    pub(crate) fn new(sampling_rate: f64) -> Self {
        Carriers {
            sampling_rate,
            tables: FxHashMap::default(),
        }
    }

    /// Returns a [`Modulation`] that modulates the `n`-th sample of a pulse
    /// with `exp(-i (2.0 * pi * frequency * n / sampling_rate + phase))`,
    /// where the `phase` offset is in radians and the `frequency` in Hz.
    pub(crate) fn modulation(&mut self, phase: f64, frequency: f64) -> Modulation<'_> {
        if frequency != 0.0 {
            let angular_frequency = 2.0 * PI * frequency;
            let sampling_rate = self.sampling_rate;
            Modulation::Carrier {
                table: self.table(angular_frequency),
                angular_frequency,
                phase,
                sampling_rate,
            }
        } else if phase != 0.0 {
            Modulation::Phase(Complex64::cis(-phase))
        } else {
            Modulation::None
        }
    }

    /// Returns the first `BLOCK` phases for `angular_frequency`, computed on first use.
    fn table(&mut self, angular_frequency: f64) -> &[Complex64; BLOCK] {
        let sampling_rate = self.sampling_rate;
        self.tables
            .entry(angular_frequency.to_bits())
            .or_insert_with(|| {
                std::array::from_fn(|k| {
                    Complex64::cis(-(angular_frequency * (k as f64 / sampling_rate)))
                })
            })
    }
}

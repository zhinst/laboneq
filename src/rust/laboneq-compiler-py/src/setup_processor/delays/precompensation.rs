// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use laboneq_dsl::signal_calibration::Precompensation;

use crate::setup_processor::precompensation::precompensation_is_active;

/// Base delay introduced by precompensation settings.
pub(super) const PRECOMPENSATION_BASE_DELAY_SAMPLES: i64 = 72;
/// Delay introduced by each exponential compensation term.
pub(super) const PRECOMPENSATION_EXPONENTIAL_DELAY_SAMPLES: i64 = 88;
/// Delay introduced by high-pass compensation.
pub(super) const PRECOMPENSATION_HIGH_PASS_DELAY_SAMPLES: i64 = 96;
/// Delay introduced by bounce compensation.
pub(super) const PRECOMPENSATION_BOUNCE_DELAY_SAMPLES: i64 = 32;
/// Delay introduced by FIR compensation.
pub(super) const PRECOMPENSATION_FIR_DELAY_SAMPLES: i64 = 136;

/// Calculate the delay introduced by precompensation settings.
pub(super) fn precompensation_delay_samples(precompensation: &Precompensation) -> i64 {
    if !precompensation_is_active(precompensation) {
        return 0;
    }

    let mut delay = PRECOMPENSATION_BASE_DELAY_SAMPLES;

    delay += PRECOMPENSATION_EXPONENTIAL_DELAY_SAMPLES * precompensation.exponential.len() as i64;

    if precompensation.high_pass.is_some() {
        delay += PRECOMPENSATION_HIGH_PASS_DELAY_SAMPLES;
    }
    if precompensation.bounce.is_some() {
        delay += PRECOMPENSATION_BOUNCE_DELAY_SAMPLES;
    }
    if precompensation.fir.is_some() {
        delay += PRECOMPENSATION_FIR_DELAY_SAMPLES;
    }
    delay
}

#[cfg(test)]
mod tests {
    use super::*;
    use laboneq_dsl::signal_calibration::{
        BounceCompensation, ExponentialCompensation, FirCompensation, HighPassCompensation,
        Precompensation,
    };

    #[test]
    fn test_precompensation_delay_samples() {
        let precomp = Precompensation {
            exponential: vec![
                ExponentialCompensation {
                    timeconstant: 0.0,
                    amplitude: 0.0,
                },
                ExponentialCompensation {
                    timeconstant: 0.0,
                    amplitude: 0.0,
                },
            ],
            high_pass: HighPassCompensation { timeconstant: 0.0 }.into(),
            bounce: BounceCompensation {
                delay: 0.0,
                amplitude: 0.0,
            }
            .into(),
            fir: FirCompensation {
                coefficients: vec![0.0],
            }
            .into(),
        };
        assert_eq!(
            precompensation_delay_samples(&precomp),
            PRECOMPENSATION_BASE_DELAY_SAMPLES
                + PRECOMPENSATION_EXPONENTIAL_DELAY_SAMPLES * 2
                + PRECOMPENSATION_HIGH_PASS_DELAY_SAMPLES
                + PRECOMPENSATION_BOUNCE_DELAY_SAMPLES
                + PRECOMPENSATION_FIR_DELAY_SAMPLES
        );
    }

    #[test]
    fn test_precompensation_delay_samples_no_precomp() {
        let precomp = Precompensation::default();
        assert_eq!(precompensation_delay_samples(&precomp), 0);
    }
}

// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::fmt;

use laboneq_dsl::signal_calibration::Precompensation;
use laboneq_dsl::types::SignalUid;
use laboneq_error::{WithContext, bail};

use crate::Result;

const DSP_OUTCOMP_BSHIFT_W: u32 = 2;
const DSP_OUTCOMP_BSHIFT_C: u32 = 4;
const DSP_OUTCOMP_COEF_W: u32 = 18;

const EXP_IIR_DSP48_PPL_C: f64 = 2.0;
const HZL_DSP_OUTC_COEF_EPS: f64 = 1.0e-32;
const HZL_DSP_OUTCEXP_MIN_AMP: f64 = -1.0 + 1.0e-6;
const HZL_DSP_OUTCEXP_MAX_ALPHA: f64 = 1.0 - 1.0e-6;
const FPGA_PATHS: f64 = 8.0;

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PrecompensationWarning {
    signal_uid: SignalUid,
    warning: PrecompensationWarningType,
}

impl fmt::Display for PrecompensationWarning {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Signal '{}': {}", self.signal_uid.0, self.warning)
    }
}

#[derive(Debug)]
pub(crate) struct SignalPrecompensation<'a> {
    pub signal_uid: SignalUid,
    pub sampling_rate: f64,
    pub precompensation: &'a Precompensation,
}

pub(crate) struct PrecompensationProcessingResult {
    pub warnings: Vec<PrecompensationWarning>,
}

/// Process the precompensation configuration for the given signals.
///
/// Returns a list of warnings for any parameters that were out of range and will be clamped, or an error for hard limit violations.
pub(crate) fn process_precompensation(
    precompensations: Vec<SignalPrecompensation>,
) -> Result<PrecompensationProcessingResult> {
    let mut collected_warnings = Vec::new();
    for precomp in &precompensations {
        let warnings =
            verify_precompensation_parameters(Some(precomp.precompensation), precomp.sampling_rate)
                .with_context(|| {
                    format!(
                        "Invalid precompensation for signal: '{}'",
                        precomp.signal_uid.0
                    )
                })?;
        warnings.into_iter().for_each(|w| {
            collected_warnings.push(PrecompensationWarning {
                signal_uid: precomp.signal_uid,
                warning: w,
            })
        });
    }
    Ok(PrecompensationProcessingResult {
        warnings: collected_warnings,
    })
}

/// Validate precompensation parameters, returning warnings for
/// out-of-range values that will be clamped, or an error for hard limit
/// violations.
fn verify_precompensation_parameters(
    precompensation: Option<&Precompensation>,
    sampling_rate: f64,
) -> Result<Vec<PrecompensationWarningType>> {
    let Some(precompensation) = precompensation else {
        return Ok(Vec::new());
    };

    let mut warnings: Vec<PrecompensationWarningType> = Vec::new();

    if !precompensation.exponential.is_empty() {
        if precompensation.exponential.len() > 8 {
            bail!("Too many exponential filters. Maximum is 8 exponential filters.");
        }
        for e in &precompensation.exponential {
            if let Some(w) = check_exponential_filter(e.timeconstant, e.amplitude, sampling_rate) {
                warnings.push(w);
            }
        }
    }

    if let Some(hp) = &precompensation.high_pass
        && !(208e-12..=166e-3).contains(&hp.timeconstant)
    {
        warnings.push(PrecompensationWarningType::HighPassTimeconstant);
    }

    if let Some(bounce) = &precompensation.bounce {
        if bounce.delay > 103.3e-9 {
            warnings.push(PrecompensationWarningType::BounceDelay);
        }
        if bounce.amplitude.abs() > 1.0 {
            warnings.push(PrecompensationWarningType::BounceAmplitude);
        }
    }

    if let Some(fir) = &precompensation.fir {
        if fir.coefficients.len() > 40 {
            bail!("Too many coefficients in FIR filter. Maximum is 40 coefficients.");
        }
        if fir.coefficients.iter().any(|c| c.abs() > 4.0) {
            warnings.push(PrecompensationWarningType::FirCoefficients);
        }
    }

    Ok(warnings)
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum PrecompensationWarningType {
    Exponential {
        timeconstant_clamped: f64,
        amplitude_clamped: f64,
    },
    HighPassTimeconstant,
    BounceDelay,
    BounceAmplitude,
    FirCoefficients,
}

impl fmt::Display for PrecompensationWarningType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Exponential {
                timeconstant_clamped,
                amplitude_clamped,
            } => {
                write!(
                    f,
                    "Exponential precompensation values out of range; \
                they will be clamped to timeconstant={timeconstant_clamped:.3e} s, \
                amplitude={amplitude_clamped:.3e}."
                )
            }
            Self::HighPassTimeconstant => write!(
                f,
                "High pass precompensation timeconstant out of range; \
                 will be clamped to [208 ps, 166 ms]."
            ),
            Self::BounceDelay => write!(
                f,
                "Bounce precompensation delay out of range; \
                 will be clamped to 103.3 ns."
            ),
            Self::BounceAmplitude => write!(
                f,
                "Bounce precompensation amplitude out of range; \
                 will be clamped to +/- 1."
            ),
            Self::FirCoefficients => write!(
                f,
                "FIR precompensation coefficients out of range; \
                 will be clamped to +/- 4."
            ),
        }
    }
}

// Out-of-range values are clamped to the nearest valid value with a warning,
// rather than raised as an error. The HDAWG hardware quantization defines a
// natural "closest representable" value, so clamping is physically meaningful.
// (ZQCS takes the opposite approach and errors on out-of-range values, because
// its firmware register bounds have no equivalent physical analogue.)
fn check_exponential_filter(
    timeconstant: f64,
    amplitude: f64,
    sampling_freq: f64,
) -> Option<PrecompensationWarningType> {
    let (tc_clamped, amp_clamped) = clamp_exp_filter_params(timeconstant, amplitude, sampling_freq);
    if !is_close(tc_clamped, timeconstant, 0.001) || !is_close(amp_clamped, amplitude, 0.001) {
        Some(PrecompensationWarningType::Exponential {
            timeconstant_clamped: tc_clamped,
            amplitude_clamped: amp_clamped,
        })
    } else {
        None
    }
}

/// Rounds a filter coefficient to mimic the pseudo-float implementation on the FPGA.
/// Follows firmware notation, hence the unusual structure.
fn round_to_fpga(coef: f64) -> f64 {
    // bit shift; can be 0, -4, -8 or -12
    let bshift = if coef != 0.0 {
        let raw = (-coef.abs().log2()).floor() / DSP_OUTCOMP_BSHIFT_C as f64;
        raw.clamp(0.0, ((1u32 << DSP_OUTCOMP_BSHIFT_W) - 1) as f64) as u32
    } else {
        0
    };
    let shift = DSP_OUTCOMP_COEF_W - 1 + DSP_OUTCOMP_BSHIFT_C * bshift;
    let scale = 1_i64 << shift;
    let max_val = (1_i64 << (DSP_OUTCOMP_COEF_W - 1)) - 1;
    let min_val = -(1_i64 << (DSP_OUTCOMP_COEF_W - 1));
    let coef_int = (coef * scale as f64).round() as i64;
    coef_int.clamp(min_val, max_val) as f64 / scale as f64
}

fn clamp_exp_filter_params(timeconstant: f64, amplitude: f64, sampling_freq: f64) -> (f64, f64) {
    // First stage of clamping
    let timeconstant = if timeconstant <= HZL_DSP_OUTC_COEF_EPS {
        HZL_DSP_OUTC_COEF_EPS
    } else {
        timeconstant
    };
    let amplitude = if amplitude.is_nan() {
        0.0
    } else if amplitude <= HZL_DSP_OUTCEXP_MIN_AMP {
        HZL_DSP_OUTCEXP_MIN_AMP
    } else {
        amplitude
    };

    let alpha = 1.0 - (-1.0 / (sampling_freq * timeconstant * (1.0 + amplitude))).exp();
    let alpha = alpha.min(HZL_DSP_OUTCEXP_MAX_ALPHA);

    // Second stage of clamping
    let scaled_alpha = -FPGA_PATHS * EXP_IIR_DSP48_PPL_C * alpha;
    let (scaled_alpha, alpha) = if scaled_alpha < -1.0 {
        let sa = -1.0_f64;
        let a = -sa / (FPGA_PATHS * EXP_IIR_DSP48_PPL_C);
        (sa, a)
    } else {
        (scaled_alpha, alpha)
    };

    let k = if amplitude > 0.0 {
        amplitude / (1.0 + amplitude - alpha)
    } else {
        amplitude / ((1.0 + amplitude) * (1.0 - alpha))
    };

    // Third stage of clamping: prevent the case where alpha == 0
    let alpha = round_to_fpga(scaled_alpha) / (-FPGA_PATHS * EXP_IIR_DSP48_PPL_C);
    let alpha = if alpha == 0.0 {
        round_to_fpga(-1.0) / (-FPGA_PATHS * EXP_IIR_DSP48_PPL_C)
    } else {
        alpha
    };
    let k = round_to_fpga(k);

    // calc_exp_filter_params_reverse
    let amplitude = if k >= 0.0 {
        k * (1.0 - alpha) / (1.0 - k)
    } else {
        k * (1.0 - alpha) / (1.0 - k + alpha * k)
    };
    let timeconstant = -1.0 / ((1.0 - alpha).ln() * sampling_freq * (1.0 + amplitude));
    (timeconstant, amplitude)
}

fn is_close(a: f64, b: f64, rel_tol: f64) -> bool {
    (a - b).abs() <= rel_tol * a.abs().max(b.abs())
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use super::*;
    use laboneq_dsl::signal_calibration::{
        BounceCompensation, ExponentialCompensation, FirCompensation, HighPassCompensation,
    };

    #[test]
    fn no_precompensation_returns_empty() {
        let result = verify_precompensation_parameters(None, 2.0e9).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn empty_precompensation_returns_empty() {
        let pc = Precompensation::default();
        let result = verify_precompensation_parameters(Some(&pc), 2.0e9).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn too_many_exponential_filters_is_error() {
        let pc = Precompensation {
            exponential: vec![
                ExponentialCompensation {
                    timeconstant: 1e-6,
                    amplitude: 0.1
                };
                9
            ],
            ..Default::default()
        };
        assert!(verify_precompensation_parameters(Some(&pc), 2.0e9).is_err());
    }

    #[test]
    fn too_many_fir_coefficients_is_error() {
        let pc = Precompensation {
            fir: Some(FirCompensation {
                coefficients: vec![0.1; 41],
            }),
            ..Default::default()
        };
        assert!(verify_precompensation_parameters(Some(&pc), 2.0e9).is_err());
    }

    #[test]
    fn high_pass_out_of_range_produces_warning() {
        let pc = Precompensation {
            high_pass: Some(HighPassCompensation { timeconstant: 1.0 }),
            ..Default::default()
        };
        let result = verify_precompensation_parameters(Some(&pc), 2.0e9).unwrap();
        assert_eq!(result, [PrecompensationWarningType::HighPassTimeconstant]);
    }

    #[test]
    fn high_pass_in_range_no_warning() {
        let pc = Precompensation {
            high_pass: Some(HighPassCompensation { timeconstant: 1e-6 }),
            ..Default::default()
        };
        let result = verify_precompensation_parameters(Some(&pc), 2.0e9).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn bounce_delay_out_of_range_produces_warning() {
        let pc = Precompensation {
            bounce: Some(BounceCompensation {
                delay: 200e-9,
                amplitude: 0.1,
            }),
            ..Default::default()
        };
        let result = verify_precompensation_parameters(Some(&pc), 2.0e9).unwrap();
        assert_eq!(result, [PrecompensationWarningType::BounceDelay]);
    }

    #[test]
    fn bounce_amplitude_out_of_range_produces_warning() {
        let pc = Precompensation {
            bounce: Some(BounceCompensation {
                delay: 1e-9,
                amplitude: 1.5,
            }),
            ..Default::default()
        };
        let result = verify_precompensation_parameters(Some(&pc), 2.0e9).unwrap();
        assert_eq!(result, [PrecompensationWarningType::BounceAmplitude]);
    }

    #[test]
    fn fir_coefficients_out_of_range_produces_warning() {
        let pc = Precompensation {
            fir: Some(FirCompensation {
                coefficients: vec![5.0, 0.1],
            }),
            ..Default::default()
        };
        let result = verify_precompensation_parameters(Some(&pc), 2.0e9).unwrap();
        assert_eq!(result, [PrecompensationWarningType::FirCoefficients]);
    }

    #[test]
    fn round_to_fpga_zero() {
        assert_eq!(round_to_fpga(0.0), 0.0);
    }

    #[test]
    fn round_to_fpga_minus_one() {
        assert_eq!(round_to_fpga(-1.0), -1.0);
    }

    #[test]
    fn test_clamp_exp_filter_params() {
        const SAMPLING_FREQ: f64 = 2.4e9; // HDAWG sampling frequency with UHFQA

        let input_expected = [
            [(11.75e-9, 200000.0), (17.93e-9, 131100.0)],
            [(1e-9, 0.0), (6.456e-9, 0.0)],
            [(1e-9, 200000.0), (1.526e-9, 131100.0)],
        ];
        for [(tc_in, amp_in), (tc_exp, amp_exp)] in input_expected {
            let (tc_clamped, amp_clamped) = clamp_exp_filter_params(tc_in, amp_in, SAMPLING_FREQ);
            assert_abs_diff_eq!(tc_clamped, tc_exp, epsilon = 0.01e-9);
            assert_abs_diff_eq!(amp_clamped, amp_exp, epsilon = 100.0);
        }
    }
}

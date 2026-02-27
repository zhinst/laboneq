// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use crate::error::{Error, Result};
use laboneq_common::types::AwgKey;
use laboneq_dsl::signal_calibration::{
    BounceCompensation, ExponentialCompensation, FirCompensation, Precompensation,
};
use laboneq_dsl::types::SignalUid;

#[derive(Debug)]
pub(super) struct AssignedPrecompensation {
    pub signal_uid: SignalUid,
    pub awg: AwgKey,
    pub precompensation: Option<Precompensation>,
}

pub(super) fn precompensation_is_active(precompensation: &Precompensation) -> bool {
    precompensation.bounce.is_some()
        || !precompensation.exponential.is_empty()
        || precompensation.fir.is_some()
        || precompensation.high_pass.is_some()
}

/// Adapt precompensations for HDAWG to ensure that signals sharing the same AWG have compatible precompensation settings.
pub(super) fn adapt_precompensations(
    precompensations: &mut [AssignedPrecompensation],
) -> Result<()> {
    // Group by awg to adapt precompensations that share the same AWG together
    let mut by_awg: HashMap<AwgKey, Vec<&mut AssignedPrecompensation>> = precompensations
        .iter_mut()
        .fold(HashMap::new(), |mut acc, assigned| {
            if let Some(precompensation) = &mut assigned.precompensation {
                // If the precompensation is not active, treat it as None to avoid unnecessary adaptations
                if !precompensation_is_active(precompensation) {
                    assigned.precompensation = None;
                }
            }
            acc.entry(assigned.awg).or_default().push(assigned);
            acc
        });
    for precomps in by_awg.values_mut() {
        if precomps.len() < 2 {
            continue;
        }
        adapt_awg_precompensation(precomps)?;
    }
    Ok(())
}

fn adapt_awg_precompensation(precompensations: &mut [&mut AssignedPrecompensation]) -> Result<()> {
    // All precompensations must either have or not have high pass
    let has_high_pass: Vec<_> = precompensations
        .iter()
        .filter_map(|p| {
            if let Some(precomp) = &p.precompensation {
                precomp.high_pass.is_some().then_some(())
            } else {
                None
            }
        })
        .collect();
    if !has_high_pass.is_empty() && has_high_pass.len() != precompensations.len() {
        return Err(Error::new(format!(
            "All precompensation settings for the same AWG must have the high pass filter enabled or disabled: '{}'.",
            precompensations
                .iter()
                .map(|p| p.signal_uid.0.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        )));
    }

    let mut n_exponentials = 0;
    let mut has_fir = false;
    let mut has_bounce = false;
    for precomp in precompensations.iter() {
        if let Some(precomp) = &precomp.precompensation {
            n_exponentials = n_exponentials.max(precomp.exponential.len());
            if precomp.fir.is_some() {
                has_fir = true;
            }
            if precomp.bounce.is_some() {
                has_bounce = true;
            }
        }
    }

    for precomp in precompensations.iter_mut() {
        if precomp.precompensation.is_none() {
            precomp.precompensation = Some(Precompensation::default());
        }
        // Adapt exponentials: Extend to n_exponentials if needed
        let current_len = precomp.precompensation.as_ref().unwrap().exponential.len();
        if current_len < n_exponentials {
            let to_add = n_exponentials - current_len;
            precomp
                .precompensation
                .as_mut()
                .unwrap()
                .exponential
                .extend(default_exponential_settings(to_add));
        }

        // Adapt FIR
        if has_fir && precomp.precompensation.as_ref().unwrap().fir.is_none() {
            precomp.precompensation.as_mut().unwrap().fir = default_fir_settings().into();
        }

        // Adapt bounce
        if has_bounce && precomp.precompensation.as_ref().unwrap().bounce.is_none() {
            precomp.precompensation.as_mut().unwrap().bounce = default_bounce_settings().into();
        }
    }
    Ok(())
}

/// Default FIR compensation settings for additional signals sharing the same AWG.
fn default_fir_settings() -> FirCompensation {
    FirCompensation {
        coefficients: vec![1.0],
    }
}

/// Default bounce compensation settings for additional signals sharing the same AWG.
fn default_bounce_settings() -> BounceCompensation {
    BounceCompensation {
        delay: 10e-9,
        amplitude: 0.0,
    }
}

/// Default exponential compensation settings for additional signals sharing the same AWG.
fn default_exponential_settings(n: usize) -> Vec<ExponentialCompensation> {
    vec![
        ExponentialCompensation {
            timeconstant: 10e-9,
            amplitude: 0.0,
        };
        n
    ]
}

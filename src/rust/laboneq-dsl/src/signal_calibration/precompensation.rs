// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

/// Precompensation settings
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Precompensation {
    pub bounce: Option<BounceCompensation>,
    pub exponential: Vec<ExponentialCompensation>,
    pub fir: Option<FirCompensation>,
    pub high_pass: Option<HighPassCompensation>,
}

/// High-pass precompensation filter
#[derive(Debug, Clone, PartialEq)]
pub struct HighPassCompensation {
    pub timeconstant: f64,
}

/// Exponential precompensation filter
#[derive(Debug, Clone, PartialEq)]
pub struct ExponentialCompensation {
    pub timeconstant: f64,
    pub amplitude: f64,
}

/// FIR precompensation filter
#[derive(Debug, Clone, PartialEq)]
pub struct FirCompensation {
    pub coefficients: Vec<f64>,
}

/// Bounce precompensation
#[derive(Debug, Clone, PartialEq)]
pub struct BounceCompensation {
    pub delay: f64,
    pub amplitude: f64,
}

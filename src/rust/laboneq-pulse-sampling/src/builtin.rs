// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

//! LabOne Q's standard pulses and their parameters.
//!
//! Each pulse definition follows its LabOne Q pulse library counterpart,
//! operation by operation, so that both evaluate to the same samples.

use std::ops::Range;

use num_complex::Complex64;

use crate::envelope;
use crate::samples::{Samples, SamplesRef};

/// Reused buffers for standard pulse envelopes.
#[derive(Default)]
pub(crate) struct Scratch {
    real: Vec<f64>,
    complex: Vec<Complex64>,
    ones: Vec<f64>,
}

/// LabOne Q's standard pulses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuiltinPulse {
    Const,
    Gaussian,
    Drag,
    GaussianSquare,
}

/// The pulse parameters standard pulses read.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Parameters {
    pub sigma: Option<ParameterValue>,
    pub beta: Option<ParameterValue>,
    pub width: Option<ParameterValue>,
    pub order: Option<ParameterValue>,
    pub zero_boundaries: Option<ParameterValue>,
    /// Whether a parameter reuses a name reserved for functional-pulse arguments
    /// (`x`, `length`, `sampling_rate`).
    pub reuses_argument_name: bool,
}

impl Parameters {
    pub(crate) const NONE: Self = Parameters {
        sigma: None,
        beta: None,
        width: None,
        order: None,
        zero_boundaries: None,
        reuses_argument_name: false,
    };
}

/// A pulse parameter's value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ParameterValue {
    Int(i64),
    Float(f64),
    Bool(bool),
    /// Any other value, to be interpreted by an
    /// [`ExternalSampler`](crate::external::ExternalSampler)
    Opaque,
}

const DEFAULT_SIGMA: f64 = 1.0 / 3.0;
const DEFAULT_BETA: f64 = 0.2;
const DEFAULT_WIDTH_FRACTION: f64 = 0.9;

/// Returns the `slice` of a `pulse` envelope of `length` samples,
/// or `None` for the following cases:
/// - a parameter reusing a functional pulse argument name
/// - a non-float `sigma`, `beta` or `width`
/// - a Gaussian `order` that is not a positive even number
/// - a non-boolean `zero_boundaries`
/// - `zero_boundaries` on fewer than two samples
/// - a `gaussian_square` pulse that LabOne Q would reject
pub(crate) fn builtin_envelope<'s>(
    pulse: BuiltinPulse,
    length: usize,
    slice: Range<usize>,
    parameters: &Parameters,
    sampling_rate: f64,
    scratch: &'s mut Scratch,
) -> Option<SamplesRef<'s>> {
    if parameters.reuses_argument_name {
        return None;
    }
    let zero_boundaries = match parameters.zero_boundaries {
        None | Some(ParameterValue::Bool(false)) => false,
        Some(ParameterValue::Bool(true)) => true,
        Some(_) => return None,
    };
    let n = slice.len();
    // LabOne Q extrapolates the zeroed boundary from two points.
    if zero_boundaries && n < 2 && pulse != BuiltinPulse::Const {
        return None;
    }
    let x = || envelope::evaluation_points(length, sampling_rate, slice.clone());
    Some(match pulse {
        // `const` ignores every other parameter.
        BuiltinPulse::Const => constant(n, &mut scratch.ones),
        BuiltinPulse::Gaussian => {
            let sigma = float(parameters.sigma, DEFAULT_SIGMA)?;
            let edge = zero_boundaries.then(|| edge(x()));
            match order(parameters.order)? {
                2.0 => gaussian(x(), sigma, edge, &mut scratch.real),
                order => super_gaussian(x(), sigma, order, edge, &mut scratch.real),
            }
        }
        BuiltinPulse::Drag => {
            let sigma = float(parameters.sigma, DEFAULT_SIGMA)?;
            let beta = float(parameters.beta, DEFAULT_BETA)?;
            let edge = zero_boundaries.then(|| edge(x()));
            drag(x(), sigma, beta, edge, &mut scratch.complex)
        }
        BuiltinPulse::GaussianSquare => {
            let length = length as f64 / sampling_rate;
            let width = float(parameters.width, DEFAULT_WIDTH_FRACTION * length)?;
            // a rise without points leaves nothing to extrapolate the zeroed boundary from
            let risefall =
                risefall(n, width, length).filter(|&risefall| risefall > 0 || !zero_boundaries)?;
            let sigma = float(parameters.sigma, DEFAULT_SIGMA)?;
            gaussian_square(n, sigma, risefall, zero_boundaries, &mut scratch.real)
        }
    })
}

/// Returns `parameter` as a Gaussian `order` if it is positive and even,
/// 2.0 if absent, and `None` otherwise.
fn order(parameter: Option<ParameterValue>) -> Option<f64> {
    let order = match parameter {
        None => return Some(2.0),
        Some(ParameterValue::Int(order)) => order as f64,
        Some(ParameterValue::Float(order)) => order,
        Some(_) => return None,
    };
    (order >= 2.0 && order % 2.0 == 0.0).then_some(order)
}

/// Returns `parameter` as a float, `default` if absent, and `None` for other values.
fn float(parameter: Option<ParameterValue>, default: f64) -> Option<f64> {
    match parameter {
        None => Some(default),
        Some(ParameterValue::Float(value)) => Some(value),
        Some(_) => None,
    }
}

fn constant(n: usize, ones: &mut Vec<f64>) -> SamplesRef<'_> {
    if ones.len() < n {
        ones.resize(n, 1.0);
    }
    Samples::Real(&ones[..n])
}

/// A Gaussian zeroed at `edge`, if given.
fn gaussian(
    x: impl Iterator<Item = f64>,
    sigma: f64,
    edge: Option<f64>,
    out: &mut Vec<f64>,
) -> SamplesRef<'_> {
    let denominator = 2.0 * sigma.powf(2.0);
    let gaussian = |x: f64| (-(x * x / denominator)).exp();
    Samples::Real(zeroed(x.map(gaussian), edge.map(gaussian), out))
}

/// A Gaussian of positive and even `order` > 2, with `sigma` adjusted
/// to keep the edge values of a standard (`order` = 2) Gaussian, zeroed at `edge`, if given.
fn super_gaussian(
    x: impl Iterator<Item = f64>,
    sigma: f64,
    order: f64,
    edge: Option<f64>,
    out: &mut Vec<f64>,
) -> SamplesRef<'_> {
    let sigma = sigma.powf(2.0 / order) / 2.0_f64.powf((order - 2.0) / (2.0 * order));
    let denominator = 2.0 * sigma.powf(2.0);
    let exponent = order / 2.0;
    let power = |base: f64| {
        if exponent == 2.0 {
            base * base
        } else {
            base.powf(exponent)
        }
    };
    let delta = edge.map(|edge| (-(edge * edge / denominator).powf(exponent)).exp());
    let samples = x.map(|x| (-power(x * x / denominator)).exp());
    Samples::Real(zeroed(samples, delta, out))
}

/// A DRAG pulse whose Gaussian is zeroed at `edge`, if given.
fn drag(
    x: impl Iterator<Item = f64>,
    sigma: f64,
    beta: f64,
    edge: Option<f64>,
    out: &mut Vec<Complex64>,
) -> SamplesRef<'_> {
    let variance = sigma.powf(2.0);
    let denominator = 2.0 * variance;
    let gaussian = |x: f64| (-(x * x) / denominator).exp();
    // numpy divides a complex number by `1 - delta` through its reciprocal.
    let zeroing = edge.map(gaussian).map(|delta| (delta, 1.0 / (1.0 - delta)));
    out.clear();
    out.extend(x.map(|x| {
        let gauss = gaussian(x);
        let derivative = beta * (-x / variance * gauss);
        match zeroing {
            None => Complex64::new(gauss, derivative),
            Some((delta, scale)) => Complex64::new((gauss - delta) * scale, derivative * scale),
        }
    }));
    Samples::Complex(out)
}

/// Returns how many of the `n` samples rise and fall around a flat top of `width`
/// seconds in a pulse of `length` seconds. `None` if LabOne Q rejects them.
fn risefall(n: usize, width: f64, length: f64) -> Option<usize> {
    if width >= length {
        return None;
    }
    // `round` in Python rounds half to even.
    let risefall = (n as f64 * (1.0 - width / length) / 2.0).round_ties_even();
    // reject a non-finite or negative count.
    (0.0..=(n / 2) as f64)
        .contains(&risefall)
        .then_some(risefall as usize)
}

/// A Gaussian square pulse, zeroed at the points before its rise and after its fall
/// when `zero_boundaries` is `true`.
fn gaussian_square(
    n: usize,
    sigma: f64,
    risefall: usize,
    zero_boundaries: bool,
    out: &mut Vec<f64>,
) -> SamplesRef<'_> {
    let denominator = 2.0 * sigma.powf(2.0);
    let gaussian = |x: f64| (-(x * x) / denominator).exp();
    let rise = linspace(-1.0, 1.0, 2 * risefall);
    let delta = zero_boundaries.then(|| gaussian(edge(rise.clone())));
    out.clear();
    out.extend(rise.clone().take(risefall).map(gaussian));
    out.resize(n - risefall, 1.0);
    out.extend(rise.skip(risefall).map(gaussian));
    if let Some(delta) = delta {
        for sample in out.iter_mut() {
            *sample = (*sample - delta) / (1.0 - delta);
        }
    }
    Samples::Real(out)
}

/// Returns the point preceding `points`, extrapolated from their first two.
fn edge(mut points: impl Iterator<Item = f64>) -> f64 {
    let mut next = || points.next().expect("at least two points");
    let (first, second) = (next(), next());
    first - (second - first)
}

/// Writes `samples` into `out`, shifted down by `delta` and normalized, if `delta` is given.
fn zeroed(samples: impl Iterator<Item = f64>, delta: Option<f64>, out: &mut Vec<f64>) -> &[f64] {
    out.clear();
    match delta {
        None => out.extend(samples),
        Some(delta) => out.extend(samples.map(|sample| (sample - delta) / (1.0 - delta))),
    }
    out
}

/// Returns `num` evenly spaced points from `start` to `stop` (as `numpy.linspace`)
fn linspace(start: f64, stop: f64, num: usize) -> impl Iterator<Item = f64> + Clone {
    let step = (stop - start) / (num as f64 - 1.0);
    (0..num).map(move |i| {
        if i + 1 == num {
            stop
        } else {
            i as f64 * step + start
        }
    })
}

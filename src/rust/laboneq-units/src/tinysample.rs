// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use crate::duration::{Duration, Second, seconds};

/// The smallest time unit used in the compiler.
///
/// Selected as the LCM of the supported instrument sampling rates (1.8, 2.0, 2.4 GHz),
/// times a "scheduler fidelity" factor of 100. Using i64 to count TinySamples, this
/// yields a maximum experiment length of ~29.6 days.
///
/// Example conversion: at 2.0 GS/s, 1 sample = 0.5 ns = 1800 × TINYSAMPLE_DURATION.
pub const TINYSAMPLE_DURATION: f64 = 1.0 / 3600000e6;

#[derive(Debug, Clone, Copy, Default, PartialOrd, PartialEq, Ord, Eq)]
pub struct TinySample;

pub fn tiny_samples<T>(value: T) -> Duration<TinySample, T> {
    Duration {
        value,
        unit: TinySample,
    }
}

pub fn seconds_to_tinysamples<T: Into<f64>>(seconds: Duration<Second, T>) -> TinySamples {
    ((seconds.value().into() / TINYSAMPLE_DURATION).round() as i64).into()
}

pub fn tinysamples_to_seconds(ts: TinySamples) -> Duration<Second> {
    seconds(ts.value() as f64 * TINYSAMPLE_DURATION)
}

pub fn samples_to_tinysamples(samples: i64, sampling_rate: f64) -> TinySamples {
    tiny_samples(((samples as f64) / (TINYSAMPLE_DURATION * sampling_rate)).round() as i64)
}

pub fn tinysamples_to_samples(ts: TinySamples, sampling_rate: f64) -> i64 {
    (ts.value() as f64 * TINYSAMPLE_DURATION * sampling_rate).round() as i64
}

/// A duration expressed in TinySamples.
pub type TinySamples<T = i64> = Duration<TinySample, T>;

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_seconds_to_tinysamples() {
        assert_eq!(
            seconds_to_tinysamples(seconds(1)),
            tiny_samples(3600000000000)
        );
        assert_eq!(seconds_to_tinysamples(seconds(1e-9)), tiny_samples(3600));
    }
    #[test]
    fn test_roundtrip() {
        assert_eq!(
            tiny_samples(1),
            seconds_to_tinysamples(tinysamples_to_seconds(tiny_samples(1)))
        );
    }
}

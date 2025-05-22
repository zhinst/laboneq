// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

pub static TINYSAMPLE: f64 = 1.0 / 3600000e6;

pub fn floor_to_grid(value: i64, grid: i64) -> i64 {
    value - value % grid
}

pub fn ceil_to_grid(value: i64, grid: i64) -> i64 {
    value + (grid - (value % grid)) % grid
}

pub fn samples_to_tinysample(t: i64) -> i64 {
    (t as f64 / TINYSAMPLE).round() as i64
}

pub fn tinysample_to_samples(t: i64, sampling_rate: f64) -> i64 {
    (t as f64 * TINYSAMPLE * sampling_rate).round() as i64
}

pub fn length_to_samples(t: f64, sampling_rate: f64) -> i64 {
    (t * sampling_rate).round() as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_seconds_to_samples() {
        assert_eq!(length_to_samples(0.0, 2e9), 0);
        assert_eq!(length_to_samples(100e-7, 2e9), 20000);
        assert_eq!(length_to_samples(100e-7, 2.4e9), 24000);

        assert_eq!(length_to_samples(12e-9, 2.4e9), 29);
        assert_eq!(length_to_samples(12e-9, 1.8e9), 22);
    }
}

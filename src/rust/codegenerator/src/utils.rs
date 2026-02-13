// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use core::f64;
// re-export for convenience
pub(crate) use codegenerator_utils::normalize_f64;

pub fn floor_to_grid(value: i64, grid: i64) -> i64 {
    value - value % grid
}

pub fn ceil_to_grid(value: i64, grid: i64) -> i64 {
    value + (grid - (value % grid)) % grid
}

/// Converts samples to a grid-aligned value.
///
/// # Returns
///
/// The grid aligned value and the remainder.
pub fn samples_to_grid(samples: i64, grid: i64) -> (i64, i64) {
    let on_grid = floor_to_grid(samples, grid);
    (on_grid, samples % grid)
}

pub fn length_to_samples(t: f64, sampling_rate: f64) -> i64 {
    (t * sampling_rate).round() as i64
}

pub fn samples_to_length(t: i64, sampling_rate: f64) -> f64 {
    t as f64 / sampling_rate
}

fn normalize_zero(value: f64) -> f64 {
    if value == 0.0 { 0.0 } else { value }
}

pub fn normalize_phase(value: f64) -> f64 {
    if value.is_nan() || value == 0.0 {
        return 0.0;
    }
    let out = match value < 0.0 {
        true => {
            value + ((-value / 2.0 / f64::consts::PI) as i64 + 1) as f64 * 2.0 * f64::consts::PI
        }
        false => value,
    };
    let value = out % (2.0 * f64::consts::PI);
    normalize_zero(value)
}

/// Sanitize a string for use as a variable name for SeqC code.
pub fn string_sanitize(input: &str) -> String {
    // Strip non-ASCII characters
    // Only allowed characters are alphanumeric and underscore
    let mut out = String::with_capacity(input.len());
    for char in input.chars() {
        if char.is_ascii() {
            if char.is_ascii_alphanumeric() || char == '_' {
                out.push(char);
            } else {
                out.push('_');
            }
        }
    }

    if out.is_empty() {
        out.push('_');
    }
    // Names must not start with a digit
    if out.chars().next().is_some_and(|c| c.is_ascii_digit()) {
        out.insert(0, '_');
    }

    if out != input {
        let hash = md5::compute(input.as_bytes());
        out.push('_');
        out.push_str(&format!("{:04x}", u16::from_be_bytes([hash[0], hash[1]])));
    }
    out
}

#[cfg(test)]
mod tests {
    use core::f64;

    use super::*;

    #[test]
    fn test_seconds_to_samples() {
        assert_eq!(length_to_samples(0.0, 2e9), 0);
        assert_eq!(length_to_samples(100e-7, 2e9), 20000);
        assert_eq!(length_to_samples(100e-7, 2.4e9), 24000);

        assert_eq!(length_to_samples(12e-9, 2.4e9), 29);
        assert_eq!(length_to_samples(12e-9, 1.8e9), 22);
    }

    #[test]
    fn test_samples_to_grid() {
        assert_eq!(samples_to_grid(24, 16), (16, 8));
        assert_eq!(samples_to_grid(-24, 16), (-16, -8));
        assert_eq!(samples_to_grid(16, 16), (16, 0));
        assert_eq!(samples_to_grid(-16, 16), (-16, 0));
        assert_eq!(samples_to_grid(0, 16), (0, 0));
    }

    #[test]
    fn test_samples_to_length() {
        assert_eq!(samples_to_length(24, 2e9), 1.2e-8);
        assert_eq!(samples_to_length(-24, 2e9), -1.2e-8);
    }

    #[test]
    fn test_normalize_phase() {
        assert_eq!(normalize_phase(0.0), 0.0);
        assert_eq!(normalize_phase(-0.0).to_bits(), 0.0_f64.to_bits());
        assert_eq!(normalize_phase(f64::consts::PI), f64::consts::PI);
        assert_eq!(normalize_phase(f64::NAN), 0.0);
    }

    #[test]
    fn test_string_sanitize() {
        assert_eq!(
            string_sanitize("abcdefghijklmnopqrstuvwxyz_ABCDEFGHIJKLMNOPQRSTUVWXYZ_0123456789"),
            "abcdefghijklmnopqrstuvwxyz_ABCDEFGHIJKLMNOPQRSTUVWXYZ_0123456789"
        );
        assert_eq!(string_sanitize("test-123_<foo>!"), "test_123__foo___6302");
        assert_eq!(string_sanitize("abc\r\n\tdef"), "abc___def_04df");
        assert_eq!(string_sanitize("123"), "_123_202c");
        assert_eq!(string_sanitize("漢字"), "__3817");
        assert_ne!(string_sanitize("---"), string_sanitize("___"));
    }
}

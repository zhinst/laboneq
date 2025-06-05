// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use core::f64;

/// Utility function to normalize f64 value to bits, normalizing NaN and -0.0 to 0.0
pub fn normalize_f64(value: f64) -> u64 {
    if value.is_nan() {
        f64::NAN.to_bits()
    } else if value == 0.0 {
        0.0_f64.to_bits()
    } else {
        value.to_bits()
    }
}

pub fn normalize_phase(value: f64) -> f64 {
    if value.is_nan() {
        return 0.0;
    }
    let out = match value < 0.0 {
        true => {
            value + ((-value / 2.0 / f64::consts::PI) as i64 + 1) as f64 * 2.0 * f64::consts::PI
        }
        false => value,
    };
    out % (2.0 * f64::consts::PI)
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
    fn test_normalize_phase() {
        assert_eq!(normalize_phase(0.0), 0.0);
        assert_eq!(normalize_phase(f64::consts::PI), f64::consts::PI);
        assert_eq!(normalize_phase(f64::NAN), 0.0);
    }

    #[test]
    fn test_hash_f64() {
        assert_eq!(normalize_f64(0.0), normalize_f64(0.0));
        assert_eq!(normalize_f64(-0.0), normalize_f64(0.0));
        assert_eq!(normalize_f64(f64::NAN), normalize_f64(f64::NAN));
        assert_eq!(normalize_f64(-1.0), normalize_f64(-1.0));

        assert_ne!(normalize_f64(-1.0), normalize_f64(1.0));
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

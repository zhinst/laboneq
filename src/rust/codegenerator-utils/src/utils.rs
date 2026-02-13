// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_f64() {
        assert_eq!(normalize_f64(0.0), normalize_f64(0.0));
        assert_eq!(normalize_f64(-0.0), normalize_f64(0.0));
        assert_eq!(normalize_f64(f64::NAN), normalize_f64(f64::NAN));
        assert_eq!(normalize_f64(-1.0), normalize_f64(-1.0));

        assert_ne!(normalize_f64(-1.0), normalize_f64(1.0));
    }
}

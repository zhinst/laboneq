// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use core::f64;

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
}

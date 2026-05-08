// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::error::Error;
use std::fmt;

const SEED_RANGE: std::ops::Range<u32> = 1..2_u32.pow(16);
const BOUND_RANGE: std::ops::Range<u32> = 0..2_u32.pow(16) - 1;

/// Software model of the PRNG on HDAWG and SHFSG.
pub struct PrngGeneratorQccs {
    lfsr: u32,
    lower: u32,
    upper: u32,
}

impl PrngGeneratorQccs {
    /// Creates a new PRNG generator with the given seed and range.
    ///
    /// The seed must be in the range [1, 65535], and the lower and upper bounds must be in [0, 65534].
    /// Additionally, the lower bound must be less than the upper bound.
    pub fn new(seed: u32, lower: u32, upper: u32) -> Result<Self, PrngError> {
        if lower > upper {
            return Err(PrngError::InvalidRange { lower, upper });
        }
        if !SEED_RANGE.contains(&seed) {
            return Err(PrngError::InvalidSeed(seed));
        }
        if !BOUND_RANGE.contains(&lower) {
            return Err(PrngError::InvalidLowerBound(lower));
        }
        if !BOUND_RANGE.contains(&upper) {
            return Err(PrngError::InvalidUpperBound(upper));
        }

        Ok(Self {
            lfsr: seed,
            lower,
            upper,
        })
    }

    /// Generates the next random number in the sequence.
    pub fn generate(&mut self) -> u32 {
        let lsb = self.lfsr & 1;
        self.lfsr >>= 1;
        if lsb != 0 {
            self.lfsr ^= 0xB400;
        }
        let next = ((self.lfsr * (self.upper - self.lower + 1)) >> 16) + self.lower;
        next & 0xFFFF
    }
}

impl Iterator for PrngGeneratorQccs {
    type Item = u32;

    fn next(&mut self) -> Option<Self::Item> {
        Some(self.generate())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum PrngError {
    InvalidSeed(u32),
    InvalidLowerBound(u32),
    InvalidUpperBound(u32),
    InvalidRange { lower: u32, upper: u32 },
}

impl fmt::Display for PrngError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSeed(seed) => {
                write!(f, "Invalid PRNG seed: must be in [1, 65535], got {}", seed)
            }
            Self::InvalidLowerBound(bound) => {
                write!(
                    f,
                    "Invalid PRNG lower bound: must be in [0, 65534], got {}",
                    bound
                )
            }
            Self::InvalidUpperBound(bound) => {
                write!(
                    f,
                    "Invalid PRNG upper bound: must be in [0, 65534], got {}",
                    bound
                )
            }
            Self::InvalidRange { lower, upper } => {
                write!(
                    f,
                    "Invalid PRNG range: lower ({}) must be <= upper ({})",
                    lower, upper
                )
            }
        }
    }
}

impl Error for PrngError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prng() {
        let mut prng = PrngGeneratorQccs::new(0xACE1, 0, 0xFFFE).unwrap();
        let expected = [
            0xe26f_u32, 0x7137, 0x389b, 0x1c4d, 0x0e26, 0xb312, 0xed88, 0xc2c3, 0x6161, 0x30b0,
        ];
        for &exp in &expected {
            assert_eq!(prng.generate(), exp);
        }
    }

    #[test]
    fn test_prng_non_zero_lower() {
        let mut prng = PrngGeneratorQccs::new(0xACE1, 10, 20).unwrap();
        let expected = [19_u32, 14, 12, 11, 10, 17, 20, 18];
        for &exp in &expected {
            assert_eq!(prng.generate(), exp);
        }
    }

    #[test]
    fn test_prng_output_in_range() {
        let (lower, upper) = (5_u32, 100_u32);
        let mut prng = PrngGeneratorQccs::new(0x1234, lower, upper).unwrap();
        for _ in 0..1000 {
            let v = prng.generate();
            assert!(
                v >= lower && v <= upper,
                "output {v} outside [{lower}, {upper}]"
            );
        }
    }

    #[test]
    fn test_prng_iterator_matches_generate() {
        let mut prng_gen = PrngGeneratorQccs::new(0xACE1, 0, 0xFFFE).unwrap();
        let mut prng_iter = PrngGeneratorQccs::new(0xACE1, 0, 0xFFFE).unwrap();
        for _ in 0..100 {
            assert_eq!(prng_gen.generate(), prng_iter.next().unwrap());
        }
    }

    #[test]
    fn test_prng_full_period() {
        // A maximal-length 16-bit LFSR cycles through all 65535 non-zero states,
        // so output[65535] should equal output[0].
        let mut prng = PrngGeneratorQccs::new(0xACE1, 0, 0xFFFE).unwrap();
        let first = prng.generate();
        for _ in 0..65534 {
            prng.generate();
        }
        assert_eq!(prng.generate(), first);
    }

    #[test]
    fn test_prng_invalid_state() {
        assert!(PrngGeneratorQccs::new(0, 0, 0xFFFE).is_err());
        assert!(PrngGeneratorQccs::new(0xACE1, 0, 0x10000).is_err());
        assert!(PrngGeneratorQccs::new(0xACE1, 0x10000, 0xFFFE).is_err());
        assert!(PrngGeneratorQccs::new(0xACE1, 20, 10).is_err());
    }
}

// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use crate::experiment::types::SignalUid;
use laboneq_units::tinysample::{TINYSAMPLE_DURATION, TinySamples, tiny_samples};
use num_integer::{Integer, div_ceil};
// Re-export for convenience
pub(crate) use num_integer::lcm;

fn is_valid_sampling_rate(rate: f64) -> bool {
    rate.is_finite() && rate > 0.0 && !rate.is_subnormal()
}

pub(crate) trait SignalGridInfo {
    fn uid(&self) -> SignalUid;
    fn sampling_rate(&self) -> f64;
    fn sample_multiple(&self) -> u16;
}

/// Compute the signal and sequencer grids for a set of signals.
///
/// The signal grid is the least common multiple (LCM) of the signal sampling rates.
/// The sequencer grid is the LCM of the sequencer rates (sampling rate / sample multiple).
///
/// This function will panic if any signal has an invalid sampling rate (non-finite or non-positive).
pub(crate) fn compute_grid<T: SignalGridInfo + Sized>(
    signals: &[&T],
) -> (TinySamples, TinySamples) {
    let mut signals_grid = 1;
    let mut sequencer_grid = 1;

    for signal in signals {
        if !is_valid_sampling_rate(signal.sampling_rate()) {
            panic!(
                "Invalid sampling rate: {} for signal {:?}",
                signal.sampling_rate(),
                signal.uid()
            );
        }
        signals_grid = lcm(signals_grid, signal_grid(*signal).value());
        let sequencer_rate = signal.sampling_rate() / signal.sample_multiple() as f64;
        sequencer_grid = lcm(
            sequencer_grid,
            (1.0 / (TINYSAMPLE_DURATION * sequencer_rate)).round() as i64,
        );
    }
    (tiny_samples(signals_grid), tiny_samples(sequencer_grid))
}

pub(crate) fn signal_grid(signal: &impl SignalGridInfo) -> TinySamples {
    let grid = (1.0 / (TINYSAMPLE_DURATION * signal.sampling_rate())).round();
    tiny_samples(grid as i64)
}

/// Ceil the given value to the nearest multiple of the grid.
///
/// This function panics if `grid` is not positive.
#[inline]
pub(crate) fn ceil_to_grid<T: Integer + Copy>(value: T, grid: T) -> T {
    assert!(grid > T::zero(), "Grid must be positive for rounding.");
    div_ceil(value, grid) * grid
}

/// Round the given value to the nearest multiple of the grid.
///
/// The rounding follows the "round half to even" strategy.
///
/// - When the value is exactly halfway between two grid multiples, it rounds to the
///   nearest even multiple
/// - For all other cases, it rounds to the nearest multiple
///
/// This function panics if `grid` is not positive.
pub(crate) fn round_to_grid<T>(value: T, grid: T) -> T
where
    T: Integer + Copy + std::ops::Neg<Output = T>,
{
    assert!(grid > T::zero(), "Grid must be positive for rounding.");
    let abs_value = if value >= T::zero() {
        value
    } else {
        value * T::one().neg()
    };
    let remainder = abs_value % grid;
    if remainder == T::zero() {
        // Already on grid
        return value;
    }
    let quotient = value / grid;
    let double_remainder = remainder + remainder;
    if double_remainder < grid {
        // Less than half - round toward zero
        quotient * grid
    } else if double_remainder > grid {
        // More than half - round away from zero
        if value >= T::zero() {
            (quotient + T::one()) * grid
        } else {
            (quotient - T::one()) * grid
        }
    } else {
        // Exactly half - round to even quotient
        if quotient % (T::one() + T::one()) == T::zero() {
            quotient * grid // Even quotient - keep it
        } else {
            // Odd quotient - make it even
            if value >= T::zero() {
                (quotient + T::one()) * grid
            } else {
                (quotient - T::one()) * grid
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::experiment::types::SignalUid;
    use laboneq_common::{
        device_traits::{HDAWG_TRAITS, SHFSG_TRAITS, UHFQA_TRAITS},
        named_id::NamedId,
    };

    #[derive(Debug, Clone)]
    struct MySignal {
        sampling_rate: f64,
        sample_multiple: u16,
    }

    impl MySignal {
        pub(crate) fn new(sampling_rate: f64, sample_multiple: u16) -> Self {
            Self {
                sampling_rate,
                sample_multiple,
            }
        }
    }

    impl SignalGridInfo for MySignal {
        fn uid(&self) -> SignalUid {
            SignalUid(NamedId::debug_id(0))
        }

        fn sampling_rate(&self) -> f64 {
            self.sampling_rate
        }

        fn sample_multiple(&self) -> u16 {
            self.sample_multiple
        }
    }

    #[test]
    fn test_compute_grid_no_signals() {
        let (signal_grid, sequencer_grid) = compute_grid::<MySignal>(&[]);
        assert_eq!(signal_grid, tiny_samples(1));
        assert_eq!(sequencer_grid, tiny_samples(1));
    }

    #[test]
    fn test_compute_grid_single() {
        let signals = [MySignal::new(2.0e9, 16)];
        let (signal_grid, sequencer_grid) = compute_grid(&signals.iter().collect::<Vec<_>>());
        assert_eq!(signal_grid, tiny_samples(1800));
        assert_eq!(sequencer_grid, tiny_samples(28800));
    }

    #[test]
    fn test_compute_grid_same_rates() {
        let signals = [
            MySignal::new(2.0e9, 16),
            MySignal::new(2.0e9, 16),
            MySignal::new(2.0e9, 16),
        ];
        let (signal_grid, sequencer_grid) = compute_grid(&signals.iter().collect::<Vec<_>>());
        // Same as single signal
        assert_eq!(signal_grid, tiny_samples(1800));
        assert_eq!(sequencer_grid, tiny_samples(28800));
    }

    #[test]
    fn test_compute_grid_common_rates() {
        let signals = [
            MySignal::new(HDAWG_TRAITS.sampling_rate, HDAWG_TRAITS.sample_multiple),
            MySignal::new(SHFSG_TRAITS.sampling_rate, SHFSG_TRAITS.sample_multiple),
            MySignal::new(UHFQA_TRAITS.sampling_rate, UHFQA_TRAITS.sample_multiple),
        ];
        let (signal_grid, sequencer_grid) = compute_grid(&signals.iter().collect::<Vec<_>>());
        // 2.4 GHz -> 1500, 2.0 GHz -> 1800, 1.8 GHz -> 2000
        // lcm(1500, 1800, 2000) = 18000
        assert_eq!(signal_grid, tiny_samples(18000));
        assert_eq!(sequencer_grid, tiny_samples(144000));

        // Test for repeating the signals for multiple times (900 signals)
        let signals = signals
            .iter()
            .cycle()
            .take(signals.len() * 300)
            .cloned()
            .collect::<Vec<_>>();
        let (signal_grid, sequencer_grid) = compute_grid(&signals.iter().collect::<Vec<_>>());
        assert_eq!(signal_grid, tiny_samples(18000));
        assert_eq!(sequencer_grid, tiny_samples(144000));
    }

    #[test]
    fn test_ceil_to_grid() {
        assert_eq!(ceil_to_grid(0, 16), 0);
        assert_eq!(ceil_to_grid(1, 16), 16);
        assert_eq!(ceil_to_grid(15, 16), 16);
        assert_eq!(ceil_to_grid(16, 16), 16);
        assert_eq!(ceil_to_grid(17, 16), 32);
        assert_eq!(ceil_to_grid(32, 16), 32);
        assert_eq!(ceil_to_grid(33, 16), 48);

        assert_eq!(ceil_to_grid(-5, 16), 0);
        assert_eq!(ceil_to_grid(-8, 16), 0);
        assert_eq!(ceil_to_grid(-16, 16), -16);
        assert_eq!(ceil_to_grid(-17, 16), -16);
    }

    #[test]
    fn test_round_to_grid() {
        // Realistic cases where the rounding did not work as expected
        assert_eq!(round_to_grid(457200, 2000), 458000);
        assert_eq!(round_to_grid(314676, 2000), 314000);
        assert_eq!(round_to_grid(45054900, 1800), 45054000);
        assert_eq!(round_to_grid(75041100, 1800), 75042000);
        assert_eq!(
            round_to_grid(71099.99999999999_f64.round() as i64, 1800),
            72000
        );
        assert_eq!(round_to_grid(71099.99999999999 as i64, 1800), 70200);
        assert_eq!(round_to_grid(71100, 1800), 72000);
        // Basic cases
        assert_eq!(round_to_grid(0, 4), 0);
        assert_eq!(round_to_grid(13, 4), 12);
        assert_eq!(round_to_grid(15, 4), 16);
        assert_eq!(round_to_grid(16, 4), 16);
        assert_eq!(round_to_grid(17, 4), 16);

        assert_eq!(round_to_grid(-13, 4), -12);
        assert_eq!(round_to_grid(-15, 4), -16);
        assert_eq!(round_to_grid(-16, 4), -16);
        assert_eq!(round_to_grid(-17, 4), -16);
        // Halfway cases
        assert_eq!(round_to_grid(2, 4), 0); // down to even
        assert_eq!(round_to_grid(6, 4), 8); // up to even
        assert_eq!(round_to_grid(-2, 4), 0); // up to even
        assert_eq!(round_to_grid(-6, 4), -8); // down to even
        // Odd multiples
        assert_eq!(round_to_grid(3, 4), 4); // up to even
        assert_eq!(round_to_grid(7, 4), 8); // up to even
        assert_eq!(round_to_grid(-3, 4), -4); // down to even
        assert_eq!(round_to_grid(-7, 4), -8); // down to even
        // Odd quotient
        assert_eq!(round_to_grid(14, 4), 16);
        assert_eq!(round_to_grid(-14, 4), -16);
        // Odd grid (should not happen often in practice)
        assert_eq!(round_to_grid(5, 3), 6);
        assert_eq!(round_to_grid(4, 3), 3);
    }
}

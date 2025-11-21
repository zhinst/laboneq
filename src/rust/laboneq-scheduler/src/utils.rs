// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use crate::experiment::types::SignalUid;
use laboneq_units::tinysample::{TINYSAMPLE_DURATION, TinySamples, tiny_samples};
use num_integer::{Integer, div_ceil};
// Re-export for convenience
pub use num_integer::lcm;

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
pub fn compute_grid<T: SignalGridInfo + Sized>(signals: &[&T]) -> (TinySamples, TinySamples) {
    let mut signal_grid = 1;
    let mut sequencer_grid = 1;

    for signal in signals {
        if !is_valid_sampling_rate(signal.sampling_rate()) {
            panic!(
                "Invalid sampling rate: {} for signal {:?}",
                signal.sampling_rate(),
                signal.uid()
            );
        }
        signal_grid = lcm(
            signal_grid,
            (1.0 / (TINYSAMPLE_DURATION * signal.sampling_rate())).round() as i64,
        );
        let sequencer_rate = signal.sampling_rate() / signal.sample_multiple() as f64;
        sequencer_grid = lcm(
            sequencer_grid,
            (1.0 / (TINYSAMPLE_DURATION * sequencer_rate)).round() as i64,
        );
    }
    (tiny_samples(signal_grid), tiny_samples(sequencer_grid))
}

/// Ceil the given value to the nearest multiple of the grid.
///
/// This function panics if `grid` is not positive.
#[inline]
pub fn ceil_to_grid<T: Integer + Copy>(value: T, grid: T) -> T {
    assert!(grid > T::zero(), "Grid must be positive for rounding.");
    div_ceil(value, grid) * grid
}

/// Round the given value to the nearest multiple of the grid.
///
/// If the value is exactly halfway between two multiples, it rounds away from zero.
/// This function panics if `grid` is not positive.
#[inline]
pub fn round_to_grid<T: Integer + Copy>(value: T, grid: T) -> T {
    assert!(grid > T::zero(), "Grid must be positive for rounding.");
    let two = T::one() + T::one();
    let half_grid = grid / two;
    if value >= T::zero() {
        ((value + half_grid) / grid) * grid
    } else {
        ((value - half_grid) / grid) * grid
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
        pub fn new(sampling_rate: f64, sample_multiple: u16) -> Self {
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
    }
}

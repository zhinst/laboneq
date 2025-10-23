// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use crate::TinySample;
use crate::experiment::types::SignalUid;
use num_integer::lcm;

/// The smallest time unit used in the compiler.
///
/// Example conversion:
///
/// 2.0 GHz 1 sample = 1800 x TINY_SAMPLE_UNIT
static TINYSAMPLE_UNIT: f64 = 1.0 / 3600000e6;

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
pub fn compute_grid<T: SignalGridInfo + Sized>(signals: &[&T]) -> (TinySample, TinySample) {
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
            (1.0 / (TINYSAMPLE_UNIT * signal.sampling_rate())).round() as TinySample,
        );
        let sequencer_rate = signal.sampling_rate() / signal.sample_multiple() as f64;
        sequencer_grid = lcm(
            sequencer_grid,
            (1.0 / (TINYSAMPLE_UNIT * sequencer_rate)).round() as TinySample,
        );
    }
    (signal_grid, sequencer_grid)
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
        assert_eq!(signal_grid, 1);
        assert_eq!(sequencer_grid, 1);
    }

    #[test]
    fn test_compute_grid_single() {
        let signals = [MySignal::new(2.0e9, 16)];
        let (signal_grid, sequencer_grid) = compute_grid(&signals.iter().collect::<Vec<_>>());
        assert_eq!(signal_grid, 1800);
        assert_eq!(sequencer_grid, 28800);
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
        assert_eq!(signal_grid, 1800);
        assert_eq!(sequencer_grid, 28800);
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
        assert_eq!(signal_grid, 18000);
        assert_eq!(sequencer_grid, 144000);

        // Test for repeating the signals for multiple times (900 signals)
        let signals = signals
            .iter()
            .cycle()
            .take(signals.len() * 300)
            .cloned()
            .collect::<Vec<_>>();
        let (signal_grid, sequencer_grid) = compute_grid(&signals.iter().collect::<Vec<_>>());
        assert_eq!(signal_grid, 18000);
        assert_eq!(sequencer_grid, 144000);
    }
}

// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

mod test_compute_delays {
    use std::num::NonZeroU16;

    use approx::assert_abs_diff_eq;

    use crate::setup_processor::delays::calculator::{
        DelayRegistry, SignalDelayProperties, compute_signal_delays,
    };

    const SAMPLING_RATE: f64 = 2e9;
    const GRID: NonZeroU16 = grid(16);

    const fn grid(samples: u16) -> NonZeroU16 {
        NonZeroU16::new(samples).unwrap()
    }

    /// A signal on the common grid, carrying `delay_signal` samples of its own
    /// delay and needing no head start.
    fn signal(uid: u32, delay_signal: i64) -> SignalDelayProperties {
        SignalDelayProperties {
            uid: uid.into(),
            sampling_rate: SAMPLING_RATE,
            sample_multiple: GRID,
            delay_signal,
            lead_delay: 0.0.into(),
        }
    }

    /// The whole wait: the sequencer part plus the sub-grid remainder.
    fn total_wait(delays: &DelayRegistry, uid: u32) -> f64 {
        (delays.signal_start_delay(uid.into()) + delays.signal_alignment_delay(uid.into())).value()
    }

    #[test]
    fn signals_without_delays_wait_for_nothing() {
        let signals = [signal(0, 0), signal(1, 0), signal(2, 0)];
        let delays = compute_signal_delays(&signals);

        for uid in 0..signals.len() as u32 {
            assert_abs_diff_eq!(
                delays.signal_start_delay(uid.into()).value(),
                0.0,
                epsilon = 1e-12
            );
            assert_abs_diff_eq!(
                delays.signal_alignment_delay(uid.into()).value(),
                0.0,
                epsilon = 1e-12
            );
        }
    }

    #[test]
    fn every_signal_ends_up_starting_together() {
        // Signals carrying unequal delays, including two that match.
        let own_delays = [0, 52, 52, 96];
        let signals: Vec<_> = own_delays
            .iter()
            .enumerate()
            .map(|(i, &delay)| signal(i as u32, delay))
            .collect();

        let delays = compute_signal_delays(&signals);

        // A signal starts once its own delay and its wait have both elapsed.
        let starts: Vec<f64> = own_delays
            .iter()
            .enumerate()
            .map(|(i, &delay)| total_wait(&delays, i as u32) + delay as f64 / SAMPLING_RATE)
            .collect();
        for start in &starts {
            assert_abs_diff_eq!(*start, starts[0], epsilon = 1e-12);
        }
    }

    #[test]
    fn a_wait_off_the_grid_leaves_a_remainder() {
        // 52 samples of delay rounds the common start up to 64, four grid steps.
        let signals = [signal(0, 52), signal(1, 0)];
        let delays = compute_signal_delays(&signals);

        // The delayed signal waits the leftover 12 samples, none of which the
        // sequencer can time.
        assert_abs_diff_eq!(
            delays.signal_start_delay(0.into()).value(),
            0.0,
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(
            delays.signal_alignment_delay(0.into()).value(),
            12.0 / SAMPLING_RATE,
            epsilon = 1e-12
        );

        // The undelayed signal waits whole grid steps, so nothing is left over.
        assert_abs_diff_eq!(
            delays.signal_start_delay(1.into()).value(),
            64.0 / SAMPLING_RATE,
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(
            delays.signal_alignment_delay(1.into()).value(),
            0.0,
            epsilon = 1e-12
        );
    }

    #[test]
    fn signals_on_different_grids_align_on_a_common_one() {
        // Sequencers at 2.4 GHz / 16 = 150 MHz and 1.8 GHz / 8 = 225 MHz. The
        // common grid is the reciprocal of their gcd, 75 MHz. A 48-sample delay
        // falls between two of its steps, so the start rounds up to 26.67 ns —
        // either sequencer grid on its own would round it somewhere the other
        // cannot reach.
        let fast = SignalDelayProperties {
            uid: 0.into(),
            sampling_rate: 2.4e9,
            sample_multiple: grid(16),
            delay_signal: 48,
            lead_delay: 0.0.into(),
        };
        let slow = SignalDelayProperties {
            uid: 1.into(),
            sampling_rate: 1.8e9,
            sample_multiple: grid(8),
            delay_signal: 0,
            lead_delay: 0.0.into(),
        };
        let delays = compute_signal_delays(&[fast, slow]);

        assert_abs_diff_eq!(
            total_wait(&delays, 0) + 48.0 / 2.4e9,
            total_wait(&delays, 1),
            epsilon = 1e-12
        );
        // The common start lands on both sequencer grids, so neither signal is
        // left with a remainder.
        assert_abs_diff_eq!(
            delays.signal_alignment_delay(0.into()).value(),
            0.0,
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(
            delays.signal_alignment_delay(1.into()).value(),
            0.0,
            epsilon = 1e-12
        );
    }

    #[test]
    fn a_head_start_adds_to_the_start_delay() {
        let lead_delay = 5e-9;
        let signals = [
            SignalDelayProperties {
                lead_delay: lead_delay.into(),
                ..signal(0, 0)
            },
            SignalDelayProperties {
                lead_delay: lead_delay.into(),
                ..signal(1, 32)
            },
        ];
        let delays = compute_signal_delays(&signals);

        // The signal that sets the start waits for its head start alone.
        assert_abs_diff_eq!(
            delays.signal_start_delay(1.into()).value(),
            lead_delay,
            epsilon = 1e-12
        );
        // The other waits 32 samples on top of it.
        assert_abs_diff_eq!(
            delays.signal_start_delay(0.into()).value(),
            lead_delay + 32.0 / SAMPLING_RATE,
            epsilon = 1e-12
        );
    }
}

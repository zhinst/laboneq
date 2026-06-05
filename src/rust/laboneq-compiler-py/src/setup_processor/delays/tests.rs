// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

#[cfg(test)]
mod test_compute_delays {
    use approx::assert_abs_diff_eq;

    use laboneq_common::types::DeviceKind;
    use laboneq_dsl::signal_calibration::{
        BounceCompensation, ExponentialCompensation, FirCompensation, HighPassCompensation,
        Precompensation,
    };
    use laboneq_units::duration::{Duration, Second};

    use crate::setup_processor::delays::{
        calculator::{SignalDelayProperties, compute_signal_delays},
        precompensation::{
            PRECOMPENSATION_BASE_DELAY_SAMPLES, PRECOMPENSATION_HIGH_PASS_DELAY_SAMPLES,
        },
    };

    #[allow(clippy::too_many_arguments)]
    fn create_signal_props<'a>(
        uid: u32,
        device_uid: u32,
        device_kind: DeviceKind,
        sampling_rate: f64,
        delay_signal: i64,
        precompensation: Option<&'a Precompensation>,
        lead_delay: Duration<Second>,
    ) -> SignalDelayProperties<'a> {
        SignalDelayProperties::new(
            uid.into(),
            sampling_rate,
            device_uid.into(),
            device_kind,
            delay_signal,
            precompensation,
            lead_delay,
        )
        .unwrap()
    }

    fn dummy_precompensation(
        n_exponentials: usize,
        high_pass: bool,
        bounce: bool,
        fir: bool,
    ) -> Precompensation {
        let exponentials = (0..n_exponentials)
            .map(|i| ExponentialCompensation {
                timeconstant: i as f64,
                amplitude: 0.0,
            })
            .collect();
        Precompensation {
            exponential: exponentials,
            high_pass: if high_pass {
                Some(HighPassCompensation { timeconstant: 0.0 })
            } else {
                None
            },
            bounce: if bounce {
                Some(BounceCompensation {
                    delay: 0.0,
                    amplitude: 0.0,
                })
            } else {
                None
            },
            fir: if fir {
                Some(FirCompensation {
                    strict: false,
                    coefficients: vec![0.0],
                })
            } else {
                None
            },
        }
    }

    #[test]
    fn test_compute_delays_no_delays() {
        let signal_a = create_signal_props(0, 0, DeviceKind::Shfsg, 2e9, 0, None, 0.0.into());
        let signal_b = create_signal_props(1, 1, DeviceKind::Hdawg, 2e9, 0, None, 0.0.into());
        let signal_c = create_signal_props(2, 0, DeviceKind::Shfsg, 2e9, 0, None, 0.0.into());

        let signals = vec![signal_a, signal_b, signal_c];
        let delays = compute_signal_delays(&signals);

        // All signals should have zero delay when no precompensation
        assert_abs_diff_eq!(
            delays.signal_port_delay(0.into()).value(),
            0.0,
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(
            delays.signal_port_delay(1.into()).value(),
            0.0,
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(
            delays.signal_port_delay(2.into()).value(),
            0.0,
            epsilon = 1e-12
        );
    }

    /// Test that signal delays are correctly applied to signals,
    /// and that HDAWG signal delays are compensated accordingly.
    #[test]
    fn test_compute_delays_with_dedicated_delay() {
        let delay_signal = 52;

        let signal_sg_0 =
            create_signal_props(0, 1, DeviceKind::Shfsg, 2e9, delay_signal, None, 0.0.into());
        let signal_sg_1 =
            create_signal_props(1, 1, DeviceKind::Shfsg, 2e9, delay_signal, None, 0.0.into());
        let signal_hdawg_0 = create_signal_props(2, 2, DeviceKind::Hdawg, 2e9, 0, None, 0.0.into());

        let signals = vec![signal_sg_0, signal_sg_1, signal_hdawg_0];
        let delays = compute_signal_delays(&signals);

        // Signals using dedicated delay routing should have the same total delay
        let delay_sg_0 = delays.signal_start_delay(0.into()) + delays.signal_port_delay(0.into());
        let delay_sg_1 = delays.signal_start_delay(1.into()) + delays.signal_port_delay(1.into());
        let delay_hdawg_0 =
            delays.signal_start_delay(2.into()) + delays.signal_port_delay(2.into());

        // Dedicated delay routing enabled signals on SG should have the same delay
        assert_abs_diff_eq!(delay_sg_0.value(), delay_sg_1.value());
        // Signal not using dedicated delay routing should have a different delay
        let delay_diff = delay_hdawg_0.value() - delay_sg_0.value();
        assert_abs_diff_eq!(delay_diff, delay_signal as f64 / 2e9);
    }

    /// Test that precompensation delays are correctly calculated and applied to signals with precompensation settings,
    /// and the other signal delays are calculated accordingly.
    #[test]
    fn test_compute_delays_with_precompensation() {
        let precomp = dummy_precompensation(0, true, false, false);
        let delay_signal = 52;
        let signal_with_delay =
            create_signal_props(0, 0, DeviceKind::Shfsg, 2e9, delay_signal, None, 0.0.into());
        let signal_precomp =
            create_signal_props(1, 1, DeviceKind::Hdawg, 2e9, 0, Some(&precomp), 0.0.into());

        let signals = vec![signal_with_delay, signal_precomp];
        let delays = compute_signal_delays(&signals);

        // Calculate total delays including precompensation effects
        let delay_with_delay =
            delays.signal_start_delay(0.into()) + delays.signal_port_delay(0.into());
        let delay_precomp =
            delays.signal_start_delay(1.into()) + delays.signal_port_delay(1.into());

        // The precompensation delay should be greater than the delay with dedicated delay routing, and the difference should match the expected compensation
        let diff = delay_with_delay.value() - delay_precomp.value();
        let expected_precompensation_delay = PRECOMPENSATION_HIGH_PASS_DELAY_SAMPLES as f64
            + PRECOMPENSATION_BASE_DELAY_SAMPLES as f64; // From precompensation_delay_samples for the given precompensation
        let expected_dedicated_delay_routing_delay = delay_signal as f64;
        let expected_compensation =
            (expected_precompensation_delay - expected_dedicated_delay_routing_delay) / 2e9; // Precompensation delay minus dedicated delay routing delay

        assert_abs_diff_eq!(diff, expected_compensation);
    }

    #[test]
    fn test_compute_delays_hdawg_uhfqa() {
        // Test with different sample multiples affecting delay calculations
        let signal_hdawg = create_signal_props(0, 0, DeviceKind::Hdawg, 2.4e9, 0, None, 5.0.into());
        let signal_uhfqa = create_signal_props(1, 1, DeviceKind::Uhfqa, 1.8e9, 0, None, 5.0.into());

        let signals = vec![signal_hdawg, signal_uhfqa];
        let delays = compute_signal_delays(&signals);

        let delay_hdawg = delays.signal_start_delay(0.into()) + delays.signal_port_delay(0.into());
        let delay_uhfqa = delays.signal_start_delay(1.into()) + delays.signal_port_delay(1.into());

        // No additional delays, just base lead times
        assert_abs_diff_eq!(delay_hdawg.value(), 5.0,);
        assert_abs_diff_eq!(delay_uhfqa.value(), 5.0);
    }
}

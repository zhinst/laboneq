// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

#[cfg(test)]
mod test_compute_delays {
    use approx::assert_abs_diff_eq;

    use laboneq_common::{
        device_traits::{DEFAULT_HDAWG_LEAD_DESKTOP_SETUP, DEFAULT_UHFQA_LEAD_PQSC},
        types::{DeviceKind, PhysicalDeviceUid},
    };
    use laboneq_dsl::signal_calibration::{
        BounceCompensation, ExponentialCompensation, FirCompensation, HighPassCompensation,
        OutputRoute, Precompensation,
    };
    use smallvec::SmallVec;

    use crate::setup_processor::delays::{
        calculator::{SignalDelayProperties, compute_signal_delays},
        output_routing::OUTPUT_ROUTE_DELAY_SAMPLES,
        precompensation::{
            PRECOMPENSATION_BASE_DELAY_SAMPLES, PRECOMPENSATION_HIGH_PASS_DELAY_SAMPLES,
        },
    };

    #[allow(clippy::too_many_arguments)]
    fn create_signal_props<'a>(
        uid: u32,
        physical_device: u16,
        device_kind: DeviceKind,
        sampling_rate: f64,
        channels: &[u16],
        output_routes: Vec<&'a OutputRoute>,
        precompensation: Option<&'a Precompensation>,
    ) -> SignalDelayProperties<'a> {
        SignalDelayProperties::new(
            uid.into(),
            SmallVec::from_slice(channels),
            PhysicalDeviceUid(physical_device),
            sampling_rate,
            device_kind,
            output_routes,
            precompensation,
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
                    coefficients: vec![0.0],
                })
            } else {
                None
            },
        }
    }

    #[test]
    fn test_compute_delays_no_delays() {
        let signal_a = create_signal_props(0, 0, DeviceKind::Shfsg, 2e9, &[0], vec![], None);
        let signal_b = create_signal_props(1, 1, DeviceKind::Hdawg, 2e9, &[0], vec![], None);
        let signal_c = create_signal_props(2, 0, DeviceKind::Shfsg, 2e9, &[0], vec![], None);

        let signals = vec![signal_a, signal_b, signal_c];
        let delays = compute_signal_delays(&signals, false);

        // All signals should have zero delay when no precompensation or output routing
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

    /// Test that output routing delays are correctly applied along SHFSG signals,
    /// and that HDAWG signal delays are compensated accordingly.
    #[test]
    fn test_compute_delays_with_output_routing() {
        let output_route_1 = OutputRoute {
            source_channel: 1,
            amplitude_scaling: None,
            phase_shift: None,
        };
        let output_route_2 = OutputRoute {
            source_channel: 3,
            amplitude_scaling: None,
            phase_shift: None,
        };

        let signal_sg_0 = create_signal_props(
            0,
            1,
            DeviceKind::Shfsg,
            2e9,
            &[0],
            vec![&output_route_1, &output_route_2],
            None,
        );
        let signal_sg_1 = create_signal_props(1, 1, DeviceKind::Shfsg, 2e9, &[1], vec![], None);
        let signal_hdawg_0 = create_signal_props(2, 2, DeviceKind::Hdawg, 2e9, &[0], vec![], None);

        let signals = vec![signal_sg_0, signal_sg_1, signal_hdawg_0];
        let delays = compute_signal_delays(&signals, false);

        // Signals using output routing should have the same total delay
        let delay_sg_0 = delays.signal_start_delay(0.into()) + delays.signal_port_delay(0.into());
        let delay_sg_1 = delays.signal_start_delay(1.into()) + delays.signal_port_delay(1.into());
        let delay_hdawg_0 =
            delays.signal_start_delay(2.into()) + delays.signal_port_delay(2.into());

        // Output router enabled signals on SG should have the same delay
        assert_abs_diff_eq!(delay_sg_0.value(), delay_sg_1.value());
        // Signal not using output router should have a different delay
        let delay_diff = delay_hdawg_0.value() - delay_sg_0.value();
        assert_abs_diff_eq!(delay_diff, OUTPUT_ROUTE_DELAY_SAMPLES as f64 / 2e9);
    }

    /// Test that precompensation delays are correctly calculated and applied to signals with precompensation settings,
    /// and the other signal delays are calculated accordingly.
    #[test]
    fn test_compute_delays_with_precompensation() {
        let precomp = dummy_precompensation(0, true, false, false);
        let output_route = OutputRoute {
            source_channel: 1,
            amplitude_scaling: None,
            phase_shift: None,
        };

        let signal_output_route = create_signal_props(
            0,
            0,
            DeviceKind::Shfsg,
            2e9,
            &[0],
            vec![&output_route],
            None,
        );
        let signal_precomp =
            create_signal_props(1, 1, DeviceKind::Hdawg, 2e9, &[0], vec![], Some(&precomp));

        let signals = vec![signal_output_route, signal_precomp];
        let delays = compute_signal_delays(&signals, false);

        // Calculate total delays including precompensation effects
        let delay_output_route =
            delays.signal_start_delay(0.into()) + delays.signal_port_delay(0.into());
        let delay_precomp =
            delays.signal_start_delay(1.into()) + delays.signal_port_delay(1.into());

        // The precompensation delay should be greater than the output route delay, and the difference should match the expected compensation
        let diff = delay_output_route.value() - delay_precomp.value();
        let expected_precompensation_delay = PRECOMPENSATION_HIGH_PASS_DELAY_SAMPLES as f64
            + PRECOMPENSATION_BASE_DELAY_SAMPLES as f64; // From precompensation_delay_samples for the given precompensation
        let expected_output_router_delay = OUTPUT_ROUTE_DELAY_SAMPLES as f64; // From output router delay for 2 output routes
        let expected_compensation =
            (expected_precompensation_delay - expected_output_router_delay) / 2e9; // Precompensation delay minus output router delay

        assert_abs_diff_eq!(diff, expected_compensation);
    }

    #[test]
    fn test_compute_delays_hdawg_uhfqa() {
        // Test with different sample multiples affecting delay calculations
        let signal_hdawg = create_signal_props(0, 0, DeviceKind::Hdawg, 2.4e9, &[0], vec![], None);
        let signal_uhfqa = create_signal_props(1, 1, DeviceKind::Uhfqa, 1.8e9, &[0], vec![], None);

        let signals = vec![signal_hdawg, signal_uhfqa];
        let delays = compute_signal_delays(&signals, true);

        let delay_hdawg = delays.signal_start_delay(0.into()) + delays.signal_port_delay(0.into());
        let delay_uhfqa = delays.signal_start_delay(1.into()) + delays.signal_port_delay(1.into());

        // No additional delays, just base lead times
        assert_abs_diff_eq!(
            delay_hdawg.value(),
            DEFAULT_HDAWG_LEAD_DESKTOP_SETUP.value(),
        );
        assert_abs_diff_eq!(delay_uhfqa.value(), DEFAULT_UHFQA_LEAD_PQSC.value(),);
    }
}

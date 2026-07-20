// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

#[cfg(test)]
mod test_compute_delays {
    use approx::assert_abs_diff_eq;

    use laboneq_common::types::DeviceKind;
    use laboneq_units::duration::{Duration, Second};

    use crate::setup_processor::delays::calculator::{
        SignalDelayProperties, compute_signal_delays,
    };

    fn create_signal_props(
        uid: u32,
        device_uid: u32,
        device_kind: DeviceKind,
        sampling_rate: f64,
        delay_signal: i64,
        lead_delay: Duration<Second>,
    ) -> SignalDelayProperties {
        SignalDelayProperties::new(
            uid.into(),
            sampling_rate,
            device_uid.into(),
            device_kind,
            delay_signal,
            lead_delay,
        )
    }

    #[test]
    fn test_compute_delays_no_delays() {
        let signal_a = create_signal_props(0, 0, DeviceKind::Shfsg, 2e9, 0, 0.0.into());
        let signal_b = create_signal_props(1, 1, DeviceKind::Hdawg, 2e9, 0, 0.0.into());
        let signal_c = create_signal_props(2, 0, DeviceKind::Shfsg, 2e9, 0, 0.0.into());

        let signals = vec![signal_a, signal_b, signal_c];
        let delays = compute_signal_delays(&signals);

        // All signals should have zero delay
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
            create_signal_props(0, 1, DeviceKind::Shfsg, 2e9, delay_signal, 0.0.into());
        let signal_sg_1 =
            create_signal_props(1, 1, DeviceKind::Shfsg, 2e9, delay_signal, 0.0.into());
        let signal_hdawg_0 = create_signal_props(2, 2, DeviceKind::Hdawg, 2e9, 0, 0.0.into());

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

    /// Verify that the HDAWG signal delay is correctly compensated relative to the SHFSG signal delay.
    #[test]
    fn test_compute_delays_with_different_signal_delays() {
        let sampling_rate = 2.0e9_f64;

        let hdawg_delay_signal: i64 = 96; // HDAWG precompensation delay with high-pass filter
        let shfsg_delay_signal: i64 = 52; // SHFSG delay with output routing enabled

        let signal_sg = create_signal_props(
            0,
            0,
            DeviceKind::Shfsg,
            sampling_rate,
            shfsg_delay_signal,
            0.0.into(),
        );
        // delay_signal already includes the precompensation delay added by the QCCS backend
        let signal_hdawg = create_signal_props(
            1,
            1,
            DeviceKind::Hdawg,
            sampling_rate,
            hdawg_delay_signal,
            0.0.into(),
        );

        let signals = vec![signal_sg, signal_hdawg];
        let delays = compute_signal_delays(&signals);

        let total_sg = delays.signal_start_delay(0.into()) + delays.signal_port_delay(0.into());
        let total_hdawg = delays.signal_start_delay(1.into()) + delays.signal_port_delay(1.into());

        // Both signals compensated for their respective delays, the difference should match the expected delay difference
        assert_abs_diff_eq!(
            total_hdawg.value() - total_sg.value(),
            (shfsg_delay_signal - hdawg_delay_signal) as f64 / sampling_rate,
            epsilon = 1e-12
        );
    }

    #[test]
    fn test_compute_delays_hdawg_uhfqa() {
        // Test with different sample multiples affecting delay calculations
        let signal_hdawg = create_signal_props(0, 0, DeviceKind::Hdawg, 2.4e9, 0, 5.0.into());
        let signal_uhfqa = create_signal_props(1, 1, DeviceKind::Uhfqa, 1.8e9, 0, 5.0.into());

        let signals = vec![signal_hdawg, signal_uhfqa];
        let delays = compute_signal_delays(&signals);

        let delay_hdawg = delays.signal_start_delay(0.into()) + delays.signal_port_delay(0.into());
        let delay_uhfqa = delays.signal_start_delay(1.into()) + delays.signal_port_delay(1.into());

        // No additional delays, just base lead times
        assert_abs_diff_eq!(delay_hdawg.value(), 5.0,);
        assert_abs_diff_eq!(delay_uhfqa.value(), 5.0);
    }
}

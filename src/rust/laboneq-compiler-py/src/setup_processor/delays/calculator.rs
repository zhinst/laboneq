// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use laboneq_common::device_traits::DeviceTraits;
use laboneq_common::types::DeviceKind;
use laboneq_dsl::types::{DeviceUid, SignalUid};
use laboneq_units::duration::{Duration, Second};

use super::on_device::{SignalDelay, compute_on_device_delays};

/// A struct to hold the on-device delays for signals, including start and port delays.
#[derive(Debug)]
pub(crate) struct DelayRegistry {
    signal_delays: HashMap<SignalUid, SignalDelayInfo>,
    lead_delays: HashMap<DeviceUid, Duration<Second>>,
}

impl DelayRegistry {
    pub(crate) fn signal_start_delay(&self, signal: SignalUid) -> Duration<Second> {
        self.signal_delays
            .get(&signal)
            .map_or(0.0.into(), |info| info.start_delay)
    }

    pub(crate) fn signal_port_delay(&self, signal: SignalUid) -> Duration<Second> {
        self.signal_delays
            .get(&signal)
            .map_or(0.0.into(), |info| info.port_delay)
    }

    pub(crate) fn device_lead_delay(&self, device: DeviceUid) -> Duration<Second> {
        self.lead_delays
            .get(&device)
            .map_or(0.0.into(), |&delay| delay)
    }

    fn new() -> Self {
        Self {
            signal_delays: HashMap::new(),
            lead_delays: HashMap::new(),
        }
    }

    fn add_lead_delay(&mut self, device: DeviceUid, delay: Duration<Second>) {
        self.lead_delays.insert(device, delay);
    }

    fn add_delay(
        &mut self,
        signal: SignalUid,
        start_delay: Duration<Second>,
        port_delay: Duration<Second>,
    ) {
        self.signal_delays.insert(
            signal,
            SignalDelayInfo {
                start_delay,
                port_delay,
            },
        );
    }
}

#[derive(Debug)]
struct SignalDelayInfo {
    start_delay: Duration<Second>,
    port_delay: Duration<Second>,
}

pub(crate) struct SignalDelayProperties {
    uid: SignalUid,
    device_uid: DeviceUid,
    sampling_rate: f64,
    sample_multiple: u16,
    delay_signal: i64,
    device_kind: DeviceKind,
    lead_delay: Duration<Second>,
}

impl SignalDelayProperties {
    pub(crate) fn new(
        uid: SignalUid,
        sampling_rate: f64,
        device_uid: DeviceUid,
        device_kind: DeviceKind,
        delay_signal: i64,
        lead_delay: Duration<Second>,
    ) -> Self {
        let traits = DeviceTraits::from_device_kind(&device_kind);
        Self {
            uid,
            device_uid,
            sampling_rate,
            sample_multiple: traits.sample_multiple,
            delay_signal,
            device_kind,
            lead_delay,
        }
    }
}

pub(crate) fn compute_signal_delays(signals: &[SignalDelayProperties]) -> DelayRegistry {
    let mut signal_delays = vec![];
    for signal in signals.iter() {
        let uid = signal.uid;
        let total_delay_samples = signal.delay_signal;
        let signal_delay = SignalDelay {
            signal_uid: uid,
            delay_samples: total_delay_samples,
            sampling_rate: signal.sampling_rate,
            sample_multiple: signal.sample_multiple,
        };
        signal_delays.push(signal_delay);
    }

    // Compute the on-device delays based on the signal delays, and then calculate the start and port delays for each signal accordingly.
    let mut delays = DelayRegistry::new();
    for on_device_delay in compute_on_device_delays(signal_delays) {
        if let Some(signal) = signals.iter().find(|s| s.uid == on_device_delay.signal_uid) {
            // Lastly add the lead delay to the start delay.
            let lead_delay = signal.lead_delay;
            delays.add_delay(
                on_device_delay.signal_uid,
                (on_device_delay.on_signal + lead_delay.value()).into(),
                on_device_delay.on_port.into(),
            );
            delays.add_lead_delay(signal.device_uid, lead_delay);
        }
    }

    // Ensure that port delays are not assigned to UHFQA devices, as they do not support port delays.
    for signal in signals {
        if signal.device_kind == DeviceKind::Uhfqa {
            assert_eq!(
                delays.signal_port_delay(signal.uid),
                0.0.into(),
                "Port delays are not supported on UHFQA devices"
            );
        }
    }
    delays
}

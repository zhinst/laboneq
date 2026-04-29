// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use laboneq_common::device_traits::DeviceTraits;
use laboneq_common::types::DeviceKind;
use laboneq_dsl::signal_calibration::Precompensation;
use laboneq_dsl::types::{DeviceUid, SignalUid};
use laboneq_units::duration::{Duration, Second};
use smallvec::SmallVec;

use super::lead_delay::get_lead_delay;
use super::on_device::{SignalDelay, compute_on_device_delays};
use super::output_routing::{RoutedOutput, calculate_output_route_delay};
use super::precompensation::precompensation_delay_samples;

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

pub(crate) struct SignalDelayProperties<'a> {
    uid: SignalUid,
    ports: &'a Vec<String>,
    device_uid: DeviceUid,
    sampling_rate: f64,
    sample_multiple: u16,
    output_route_ports: SmallVec<[&'a str; 4]>,
    precompensation: Option<&'a Precompensation>,
    device_kind: DeviceKind,
}

impl<'a> SignalDelayProperties<'a> {
    pub(crate) fn new(
        uid: SignalUid,
        ports: &'a Vec<String>,
        sampling_rate: f64,
        device_uid: DeviceUid,
        device_kind: DeviceKind,
        output_route_ports: SmallVec<[&'a str; 4]>,
        precompensation: Option<&'a Precompensation>,
    ) -> Result<Self, &'static str> {
        if device_kind != DeviceKind::Shfsg && !output_route_ports.is_empty() {
            return Err("Output routing is only supported for SHFSG devices");
        }
        let traits = DeviceTraits::from_device_kind(&device_kind);
        if !traits.supports_precompensation && precompensation.is_some() {
            return Err("Precompensation is not supported on this device");
        }
        Ok(Self {
            uid,
            ports,
            device_uid,
            sampling_rate,
            sample_multiple: traits.sample_multiple,
            output_route_ports,
            precompensation,
            device_kind,
        })
    }
}

pub(crate) fn compute_signal_delays(
    signals: &[SignalDelayProperties],
    desktop_setup: bool,
) -> DelayRegistry {
    // Calculate the output routing delays for all signals, and the precompensation delays for signals with precompensation settings.
    let output_delays = calculate_output_route_signal_delays(signals);
    let precomp_delays = signals
        .iter()
        .filter(|info| info.device_kind != DeviceKind::Zqcs)
        .filter_map(|info| {
            info.precompensation
                .as_ref()
                .map(|precomp| (info.uid, precompensation_delay_samples(precomp)))
        })
        .collect::<HashMap<_, _>>();

    let mut signal_delays = vec![];
    for signal in signals.iter() {
        let uid = signal.uid;
        let output_delay = output_delays.get(&uid).unwrap_or(&0);
        let precomp_delay = precomp_delays.get(&uid).unwrap_or(&0);
        let total_delay_samples = output_delay + precomp_delay;
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
            let lead_delay =
                get_lead_delay(&signal.device_kind, signal.sampling_rate, desktop_setup);
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

fn calculate_output_route_signal_delays(
    signals: &[SignalDelayProperties],
) -> HashMap<SignalUid, i64> {
    // Group signals by device
    let signals_by_device = signals.iter().fold(HashMap::new(), |mut acc, signal| {
        // If the device is not SHFSG, skip the output routing delay calculation as it is only relevant for SHFSG devices.
        if signal.device_kind != DeviceKind::Shfsg {
            assert!(
                signal.output_route_ports.is_empty(),
                "Output routes are only supported for SHFSG devices"
            );
            return acc;
        }
        acc.entry(signal.device_uid)
            .or_insert_with(Vec::new)
            .push(signal);
        acc
    });

    let mut delays_by_signal = HashMap::new();
    for device_signals in signals_by_device.values() {
        // Multiple signals can point to the same channel, but we only need to calculate the delay once per channel
        let signals_by_channel: HashMap<&str, Vec<SignalUid>> =
            device_signals
                .iter()
                .fold(HashMap::new(), |mut acc, signal| {
                    for port in signal.ports {
                        acc.entry(port.as_str())
                            .or_insert_with(Vec::new)
                            .push(signal.uid);
                    }
                    acc
                });
        let routed_outputs = device_signals.iter().flat_map(|signal| {
            assert_eq!(
                signal.ports.len(),
                DeviceTraits::from_device_kind(&signal.device_kind).channels_per_awg as usize,
                "Invalid number of output ports for SHFSG. Expected {}, got {}",
                DeviceTraits::from_device_kind(&signal.device_kind).channels_per_awg,
                signal.ports.len(),
            );
            signal.ports.iter().flat_map(|port| {
                signal
                    .output_route_ports
                    .iter()
                    .map(move |output_port| RoutedOutput {
                        target: port,
                        source: output_port,
                    })
            })
        });
        let output_delays = calculate_output_route_delay(routed_outputs);
        for (port, delay) in output_delays {
            if let Some(uids) = signals_by_channel.get(port) {
                for &uid in uids {
                    delays_by_signal.insert(uid, delay);
                }
            }
        }
    }
    delays_by_signal
}

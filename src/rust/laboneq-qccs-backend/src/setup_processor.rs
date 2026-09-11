// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};

use indexmap::IndexMap;

use laboneq_common::device_options::DeviceOptions;

use crate::device_kind::DeviceKind;
use laboneq_common::named_id::NamedIdStore;
use laboneq_common::shfqc::to_sg_uid;

use laboneq_common::types::PhysicalDeviceUid;
use laboneq_dsl::types::DeviceUid;
use laboneq_dsl::types::SignalUid;

use crate::Result;
use crate::experiment_view::ExperimentViewWrapper;
use crate::ports::Port;
use crate::ports::is_shfqa_port;
use crate::ports::is_shfsg_port;
use crate::setup_description::{Instrument, InstrumentKind};

/// The properties of the AWG devices of a setup, keyed by device UID.
pub(crate) struct AwgDevices {
    /// The model each device is, in setup order.
    pub kinds: IndexMap<DeviceUid, DeviceKind>,
    /// The physical device each device maps to.
    /// This UID is used to group virtual devices that share the same
    /// physical hardware, enabling proper device detection.
    pub physical_uids: HashMap<DeviceUid, PhysicalDeviceUid>,
    /// The options the instrument each device belongs to was ordered with.
    pub options: HashMap<DeviceUid, DeviceOptions>,
    pub reassigned_signals: Vec<ReassignedSignal>,
    /// Devices that form one half of an SHFQC, which appears as a virtual
    /// SHFQA and a virtual SHFSG.
    pub shfqc_devices: HashSet<DeviceUid>,
}

pub(crate) fn create_awg_devices(experiment: &mut ExperimentViewWrapper) -> Result<AwgDevices> {
    let mut physical_instrument_uid_counter = 0;
    let mut instruments = experiment.instruments.values().map(|instrument| {
        let signals = experiment
            .signals
            .iter()
            .filter(|s| s.device_uid == instrument.uid)
            .map(|s| SignalProperties {
                signal: s.uid,
                ports: s.ports.iter().collect(),
            })
            .collect::<Vec<_>>();

        let physical_instrument_uid = physical_instrument_uid_counter;
        physical_instrument_uid_counter += 1;

        InstrumentProperties {
            physical_instrument_uid,
            instrument,
            signals,
        }
    });

    instruments.try_fold(
        AwgDevices {
            kinds: IndexMap::new(),
            physical_uids: HashMap::new(),
            options: HashMap::new(),
            reassigned_signals: Vec::new(),
            shfqc_devices: HashSet::new(),
        },
        |mut acc, instrument| {
            let out = device_to_awg_devices(&instrument, experiment.id_store)?;
            if instrument.instrument.kind == InstrumentKind::Shfqc {
                acc.shfqc_devices
                    .extend(out.devices.iter().map(|(d, _)| *d));
            }
            acc.kinds.extend(out.devices.iter().copied());
            acc.physical_uids.extend(
                out.devices
                    .iter()
                    .map(|(d, _)| (*d, instrument.physical_instrument_uid.into())),
            );
            acc.options.extend(
                out.devices
                    .iter()
                    .map(|(d, _)| (*d, instrument.instrument.options.clone())),
            );
            acc.reassigned_signals.extend(out.reassigned_signals);
            Ok(acc)
        },
    )
}

/// Represents a signal that has been reassigned to a different device UID.
pub(crate) struct ReassignedSignal {
    pub signal: SignalUid,
    pub to_device_uid: DeviceUid,
}

struct InstrumentProperties<'a> {
    physical_instrument_uid: u16,
    instrument: &'a Instrument,
    signals: Vec<SignalProperties<'a>>,
}

struct SignalProperties<'a> {
    signal: SignalUid,
    ports: Vec<&'a Port>,
}

/// The AWG devices one instrument provides, and the models they are.
struct InstrumentDevices {
    devices: Vec<(DeviceUid, DeviceKind)>,
    reassigned_signals: Vec<ReassignedSignal>,
}

fn device_to_awg_devices(
    instrument: &InstrumentProperties,
    id_store: &mut NamedIdStore,
) -> Result<InstrumentDevices> {
    match instrument.instrument.kind {
        InstrumentKind::Shfqc => {
            let out = split_shfqc(instrument, id_store)?;
            Ok(InstrumentDevices {
                devices: [
                    out.shfqa_device.map(|d| (d, DeviceKind::Shfqa)),
                    out.shfsg_device.map(|d| (d, DeviceKind::Shfsg)),
                ]
                .into_iter()
                .flatten()
                .collect(),
                reassigned_signals: out.reassigned_signals,
            })
        }
        // No additional signals for non-SHFQC devices
        _ => {
            let kind: DeviceKind = instrument
                .instrument
                .kind
                .try_into()
                .expect("Unsupported device kind");
            Ok(InstrumentDevices {
                devices: vec![(instrument.instrument.uid, kind)],
                reassigned_signals: vec![],
            })
        }
    }
}

struct ShfqcSplitResult {
    shfqa_device: Option<DeviceUid>,
    shfsg_device: Option<DeviceUid>,
    reassigned_signals: Vec<ReassignedSignal>,
}

/// Split SHFQC into SHFQA and SHFSG virtual AWG devices.
///
/// # Returns:
/// A [`ShfqcSplitResult`] containing the created SHFQA and SHFSG AWG devices, along with the list of signals that were reassigned to the SHFSG device (i.e. those connected to SG channels).
fn split_shfqc(
    instrument: &InstrumentProperties,
    id_store: &mut NamedIdStore,
) -> Result<ShfqcSplitResult> {
    assert_eq!(
        instrument.instrument.kind,
        InstrumentKind::Shfqc,
        "Expected instrument to be of kind SHFQC"
    );

    // Determine if there are any SHFQA or SHFSG ports among the signals of this instrument.
    // The correctness of the ports should be validated elsewhere.
    let has_shfqa_ports = instrument
        .signals
        .iter()
        .any(|s| s.ports.iter().any(|p| is_shfqa_port(p)));

    let shfsg_ports = instrument
        .signals
        .iter()
        .filter(|s| s.ports.iter().any(|p| is_shfsg_port(p)))
        .collect::<Vec<_>>();

    // The SHFQA half keeps the instrument's UID.
    let shfqa_device = has_shfqa_ports.then_some(instrument.instrument.uid);

    // The SHFSG half gets a UID of its own.
    let shfsg_device = if !shfsg_ports.is_empty() {
        let new_uid = to_sg_uid(id_store.resolve(instrument.instrument.uid).unwrap());
        Some(id_store.get_or_insert(&new_uid).into())
    } else {
        None
    };

    // Reassign signals connected to SG ports to the SHFSG device (if it was created).
    let reassigned_signals = if let Some(shfsg) = shfsg_device {
        shfsg_ports
            .iter()
            .map(|s| ReassignedSignal {
                signal: s.signal,
                to_device_uid: shfsg,
            })
            .collect()
    } else {
        Vec::new()
    };

    let result = ShfqcSplitResult {
        shfqa_device,
        shfsg_device,
        reassigned_signals,
    };
    Ok(result)
}

// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use laboneq_common::named_id::NamedIdStore;
use laboneq_common::types::DeviceKind;

use laboneq_dsl::device_setup::Instrument;
use laboneq_dsl::device_setup::InstrumentKind;
use laboneq_dsl::types::DeviceUid;
use laboneq_dsl::types::SignalUid;
use laboneq_ir::system::AwgDevice;

use crate::Result;
use crate::experiment_view::ExperimentViewWrapper;
use crate::ports::Port;
use crate::ports::is_shfqa_port;
use crate::ports::is_shfsg_port;

pub(crate) fn create_awg_devices(
    experiment: &mut ExperimentViewWrapper,
) -> Result<(Vec<AwgDevice>, Vec<ReassignedSignal>)> {
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

        InstrumentProperties {
            instrument,
            signals,
        }
    });

    instruments.try_fold(
        (Vec::new(), Vec::new()),
        |(mut awg_devices, mut reassigned_signals), instrument| {
            let (devices, reassigned) = device_to_awg_devices(&instrument, experiment.id_store)?;
            awg_devices.extend(devices);
            reassigned_signals.extend(reassigned);
            Ok((awg_devices, reassigned_signals))
        },
    )
}

/// Represents a signal that has been reassigned to a different device UID.
pub(crate) struct ReassignedSignal {
    pub signal: SignalUid,
    pub to_device_uid: DeviceUid,
}

struct InstrumentProperties<'a> {
    instrument: &'a Instrument,
    signals: Vec<SignalProperties<'a>>,
}

struct SignalProperties<'a> {
    signal: SignalUid,
    ports: Vec<&'a Port>,
}

fn device_to_awg_devices(
    instrument: &InstrumentProperties,
    id_store: &mut NamedIdStore,
) -> Result<(Vec<AwgDevice>, Vec<ReassignedSignal>)> {
    match instrument.instrument.kind {
        InstrumentKind::Shfqc => {
            let out = split_shfqc(instrument, id_store)?;
            let qa_sg = [out.shfqa_device, out.shfsg_device]
                .into_iter()
                .flatten()
                .collect::<Vec<_>>();
            Ok((qa_sg, out.reassigned_signals))
        }
        _ => Ok((vec![instrument_to_awg_device(instrument)], vec![])), // No additional signals for non-SHFQC devices
    }
}

fn instrument_to_awg_device(instrument: &InstrumentProperties) -> AwgDevice {
    let mut builder = AwgDevice::builder(
        instrument.instrument.uid,
        instrument.instrument.physical_device_uid,
        instrument
            .instrument
            .kind
            .try_into()
            .expect("Unsupported device kind"),
    );
    builder = builder.options(instrument.instrument.options.clone());
    builder.build()
}

struct ShfqcSplitResult {
    shfqa_device: Option<AwgDevice>,
    shfsg_device: Option<AwgDevice>,
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

    // Create SHFQA device if there are any QA signals.
    let shfqa_device = if has_shfqa_ports {
        let shfqa_builder = AwgDevice::builder(
            instrument.instrument.uid,
            instrument.instrument.physical_device_uid,
            DeviceKind::Shfqa,
        )
        .options(instrument.instrument.options.clone())
        .shfqc(true);
        Some(shfqa_builder.build())
    } else {
        None
    };

    // Create SHFSG device if there are any SG signals.
    let shfsg_device = if !shfsg_ports.is_empty() {
        let shfsg_uid = {
            let new_uid =
                format_shfqc_sg_device_uid(id_store.resolve(instrument.instrument.uid).unwrap());
            id_store.get_or_insert(&new_uid).into()
        };

        let shfsg_builder = AwgDevice::builder(
            shfsg_uid,
            instrument.instrument.physical_device_uid,
            DeviceKind::Shfsg,
        )
        .options(instrument.instrument.options.clone())
        .shfqc(true);
        let shfsg = shfsg_builder.build();
        Some(shfsg)
    } else {
        None
    };

    // Reassign signals connected to SG ports to the SHFSG device (if it was created).
    let reassigned_signals = if let Some(shfsg) = &shfsg_device {
        shfsg_ports
            .iter()
            .map(|s| ReassignedSignal {
                signal: s.signal,
                to_device_uid: shfsg.uid(),
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

/// Format the UID for the virtual SHFSG device created from splitting an SHFQC device.
///
/// The UID is derived from the original SHFQC instrument UID by appending a suffix to indicate it's the SG part of the split device.
/// This is necessary to ensure the new SHFSG device has a unique UID.
fn format_shfqc_sg_device_uid(instrument_uid: &str) -> String {
    const VIRTUAL_SHFSG_UID_SUFFIX: &str = "_sg";
    format!("{}{}", instrument_uid, VIRTUAL_SHFSG_UID_SUFFIX)
}

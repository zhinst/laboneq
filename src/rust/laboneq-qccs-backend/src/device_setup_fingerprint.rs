// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use laboneq_common::device_setup_fingerprint::InstrumentEntryType;
use laboneq_common::named_id::NamedIdStore;
use laboneq_dsl::device_setup::InstrumentKind;
use laboneq_dsl::setup_description_qccs::SetupDescriptionQccs;

use crate::ports::parse_port;

/// Generates a fingerprint for the given device setup.
pub(crate) fn device_setup_fingerprint(
    experiment: &SetupDescriptionQccs,
    id_store: &NamedIdStore,
) -> String {
    use laboneq_common::device_setup_fingerprint::{InstrumentEntry, device_setup_fingerprint};

    let mut entries =
        Vec::with_capacity(experiment.instruments.len() + experiment.auxiliary_devices.len());

    for instrument in &experiment.instruments {
        let name = id_store
            .resolve(instrument.uid)
            .unwrap_or("<unknown>")
            .to_string();
        let options = instrument.options.to_vec();

        // For SHFQC, we need to add separate entries for the SHFQA and SHFSG channels since Controller behaviour depends on whether SHFQA channels are present or not.
        let entry = if instrument.kind == InstrumentKind::Shfqc {
            let has_qa_channels = experiment.signals_by_device(instrument.uid).any(|c| {
                c.ports
                    .iter()
                    .any(|p| parse_port(p, InstrumentKind::Shfqa).is_ok())
            });
            if has_qa_channels {
                InstrumentEntry::new(name, InstrumentEntryType::Shfqc { has_qa: true }, options)
            } else {
                InstrumentEntry::new(name, InstrumentEntryType::Shfqc { has_qa: false }, options)
            }
        } else {
            InstrumentEntry::new(
                name,
                InstrumentEntryType::NonShfqc(instrument.kind.to_string()),
                options,
            )
        };
        entries.push(entry);
    }

    for aux_device in &experiment.auxiliary_devices {
        let name = id_store
            .resolve(aux_device.uid())
            .unwrap_or("<unknown>")
            .to_string();
        let kind = aux_device.kind().to_string();
        entries.push(InstrumentEntry::new(
            name,
            InstrumentEntryType::NonShfqc(kind),
            aux_device.options().to_vec(),
        ));
    }

    device_setup_fingerprint(entries)
}

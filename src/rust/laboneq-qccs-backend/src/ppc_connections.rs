// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use laboneq_common::types::{AuxiliaryDeviceKind, ChannelKey};
use laboneq_dsl::device_setup::InstrumentKind;
use laboneq_dsl::types::SignalUid;
use laboneq_error::{WithContext, bail};

use crate::Result;
use crate::experiment_view::ExperimentViewWrapper;
use crate::ports::{IoDirection, parse_port};

/// Resolves the connections between SHFPPC channels and QA channels.
///
/// Ensures that each QA channel and PPC channel is only connected once, i.e., that there are no multiple connections for the same QA or PPC channel.
pub(crate) fn resolve_ppc_connections(
    exp_view: &ExperimentViewWrapper,
) -> Result<HashMap<SignalUid, ChannelKey>> {
    let ppc_connections =
        parse_ppc_connections(exp_view).with_context(|| "Invalid PPC channel connection.")?;

    validate_single_connection(&ppc_connections)?;

    let qa_to_ppc_map = ppc_connections
        .iter()
        .map(|conn| (conn.qa_channel, conn.ppc_channel))
        .collect::<HashMap<_, _>>();

    let mut ppc_channel_map = HashMap::new();
    for signal in exp_view.signals.iter() {
        for channel in signal
            .ports
            .iter()
            .map(|p| ChannelKey::new(signal.device_uid, p.channel))
        {
            if let Some(ppc_channel) = qa_to_ppc_map.get(&channel) {
                // Check that same signal is not connected to multiple PPC channels
                if let Some(existing_ppc_channel) = ppc_channel_map.get(&signal.uid) {
                    if existing_ppc_channel != ppc_channel {
                        bail!(
                            "Signal '{}' is connected to multiple PPC channels: '{}' and '{}'",
                            signal.uid.0,
                            existing_ppc_channel,
                            ppc_channel
                        );
                    }
                } else {
                    ppc_channel_map.insert(signal.uid, *ppc_channel);
                }
            }
        }
    }
    Ok(ppc_channel_map)
}

struct PpcConnection {
    pub ppc_channel: ChannelKey,
    pub qa_channel: ChannelKey,
}

fn parse_ppc_connections(experiment: &ExperimentViewWrapper) -> Result<Vec<PpcConnection>> {
    let mut ppc_connections = Vec::new();
    for connection in experiment.internal_connections.iter() {
        let from_instrument = experiment.get_auxiliary_device_by_uid(connection.from_instrument)?;
        let to_instrument = experiment.get_device_by_uid(connection.to_instrument)?;

        if from_instrument.kind() == AuxiliaryDeviceKind::Shfppc {
            let ppc_channel = parse_port(&connection.from_port, InstrumentKind::Shfppc)?.channel;
            let qa_channel = parse_port(&connection.to_port, InstrumentKind::Shfqa)
                .and_then(|p| {
                    if p.direction != IoDirection::Input {
                        bail!("PPC connection must be made to QA input port.");
                    }
                    Ok(p.channel)
                })
                .with_context(|| "PPC channels can only be connected to QA input channels.")?;
            ppc_connections.push(PpcConnection {
                ppc_channel: ChannelKey::new(from_instrument.uid(), ppc_channel),
                qa_channel: ChannelKey::new(to_instrument.uid, qa_channel),
            });
        }
    }
    Ok(ppc_connections)
}

/// Ensures that each QA channel and PPC channel is only connected once, i.e., that there are no multiple connections for the same QA or PPC channel.
fn validate_single_connection(ppc_connections: &[PpcConnection]) -> Result<()> {
    let mut qa_to_ppc_map = HashMap::new();
    let mut ppc_to_qa_map = HashMap::new();
    for connection in ppc_connections {
        if let Some(existing_ppc) = qa_to_ppc_map.get(&connection.qa_channel) {
            if existing_ppc != &connection.ppc_channel {
                bail!(
                    "QA channel '{}' is connected to multiple PPC channels: '{}' and '{}'",
                    connection.qa_channel,
                    existing_ppc,
                    connection.ppc_channel
                );
            }
        } else {
            qa_to_ppc_map.insert(connection.qa_channel, connection.ppc_channel);
        }
        if let Some(existing_qa) = ppc_to_qa_map.get(&connection.ppc_channel) {
            if existing_qa != &connection.qa_channel {
                bail!(
                    "PPC channel '{}' is connected to multiple QA channels: '{}' and '{}'",
                    connection.ppc_channel,
                    existing_qa,
                    connection.qa_channel
                );
            }
        } else {
            ppc_to_qa_map.insert(connection.ppc_channel, connection.qa_channel);
        }
    }
    Ok(())
}

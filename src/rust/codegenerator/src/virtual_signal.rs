// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use anyhow::anyhow;
use indexmap::IndexMap;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::Result;
use crate::ir::SignalUid;
use crate::ir::compilation_job::{AwgCore, AwgKind, DeviceKind, Signal, SignalKind};

#[derive(Debug)]
struct Channel {
    pub id: u16,
    pub signal: Arc<Signal>,
}

#[derive(Debug)]
pub(crate) struct VirtualSignal {
    signals: IndexMap<SignalUid, Channel>,
    subchannel: Option<u8>,
}

impl VirtualSignal {
    fn new(channels: Vec<Channel>, subchannel: Option<u8>) -> Self {
        let signals = channels.into_iter().map(|x| (x.signal.uid, x));
        VirtualSignal {
            signals: signals.collect(),
            subchannel,
        }
    }

    pub(crate) fn signals(&self) -> impl Iterator<Item = &Arc<Signal>> {
        self.signals.iter().map(|x| &x.1.signal)
    }

    pub(crate) fn subchannel(&self) -> Option<u8> {
        self.subchannel
    }

    pub(crate) fn contains_signal(&self, uid: SignalUid) -> bool {
        self.signals.contains_key(&uid)
    }

    pub(crate) fn is_multiplexed(&self) -> bool {
        self.signals.keys().len() > 1
    }

    pub(crate) fn get_channel_by_signal(&self, uid: SignalUid) -> Option<u16> {
        self.signals.get(&uid).map(|x| x.id)
    }
}

pub(crate) struct VirtualSignals(Vec<VirtualSignal>);

impl VirtualSignals {
    pub(crate) fn iter(&self) -> impl Iterator<Item = &VirtualSignal> {
        self.0.iter()
    }
}

fn validate_signal_oscillators(signal: &VirtualSignal, awg: &AwgCore) -> Result<()> {
    if !signal.is_multiplexed() {
        return Ok(());
    }
    let mut hw_modulated_signals: HashMap<u8, Vec<SignalUid>> = HashMap::new();
    let mut sw_modulated_signals: HashMap<u8, Vec<SignalUid>> = HashMap::new();
    for channel in signal.signals() {
        let subchannel = if awg.kind == AwgKind::DOUBLE {
            channel.channels.first().unwrap_or(&0)
        } else {
            &0
        };
        if channel.is_sw_modulated() {
            if hw_modulated_signals.contains_key(subchannel)
                && !hw_modulated_signals[subchannel].is_empty()
            {
                let mut signals = hw_modulated_signals[subchannel].iter().collect::<Vec<_>>();
                signals.sort();
                let err_msg = format!(
                    "Attempting to multiplex HW-modulated signal(s) ({}) with signal that is not HW modulated ({}).",
                    signals[0].0, channel.uid.0
                );
                return Err(anyhow!(err_msg).into());
            }
            sw_modulated_signals
                .entry(*subchannel)
                .or_default()
                .push(channel.uid);
        } else if channel.is_hw_modulated() {
            if sw_modulated_signals.contains_key(subchannel)
                && let Some(sw_modulated_signal) = sw_modulated_signals[subchannel].first()
            {
                let err_msg = format!(
                    "Attempting to multiplex SW-modulated signal(s) ({}) with signal that is not SW modulated ({}).",
                    channel.uid.0, sw_modulated_signal.0
                );
                return Err(anyhow!(err_msg).into());
            }
            hw_modulated_signals
                .entry(*subchannel)
                .or_default()
                .push(channel.uid);
        }
    }
    if awg.device_kind().traits().supports_oscillator_switching {
        return Ok(());
    }
    let mut signals = if awg.kind == AwgKind::DOUBLE {
        if hw_modulated_signals.values().all(|ids| ids.len() == 1) {
            return Ok(());
        }
        hw_modulated_signals
            .values()
            .find(|ids| ids.len() > 1)
            .expect("Internal error: Expected at least one HW-modulated signal")
            .clone()
    } else {
        let hw_modulated_signals: Vec<SignalUid> =
            hw_modulated_signals.values().flatten().copied().collect();
        if hw_modulated_signals.len() <= 1 {
            return Ok(());
        }
        hw_modulated_signals
    };
    signals.sort();
    let msg = format!(
        "Attempting to multiplex several hardware-modulated signals: \
                '{}' on device '{}', which does not support oscillator switching.",
        signals
            .iter()
            .map(|s| s.0.to_string())
            .collect::<Vec<_>>()
            .join(", "),
        awg.device_kind().as_str()
    );
    Err(anyhow!(msg).into())
}

fn get_signal_channels(signals: &[Arc<Signal>]) -> HashSet<Vec<u8>> {
    signals
        .iter()
        .filter(|signal| signal.kind != SignalKind::INTEGRATION)
        .map(|signal| signal.channels.clone())
        .collect()
}

fn is_multiplexing(awg: &AwgCore) -> Result<bool> {
    let is_shfqa = *awg.device_kind() == DeviceKind::SHFQA;
    let is_hdawg = *awg.device_kind() == DeviceKind::HDAWG;
    let channel_sets = get_signal_channels(&awg.signals);
    match awg.kind {
        AwgKind::DOUBLE => {
            // Signals of this type have two channels, like 0 and 1 or 4 and 5.
            if channel_sets.len() != 2 {
                return Err(
                    anyhow!("DOUBLE signals need two channels. Found: {channel_sets:?}").into(),
                );
            }
            assert!(is_hdawg, "HDAWG device required for DOUBLE signals");
            Ok(awg.signals.len() > 2)
        }
        AwgKind::SINGLE => {
            if channel_sets.len() != 1 {
                return Err(anyhow!(
                    "SINGLE signals need a single channel. Found: {channel_sets:?}"
                )
                .into());
            }
            assert!(is_hdawg, "HDAWG device required for SINGLE signals");
            Ok(awg.signals.len() > 1)
        }
        AwgKind::IQ => {
            if !is_shfqa && channel_sets.len() > 1 {
                return Err(anyhow!(
                    "IQ signals need a single channel configuration. Found: {channel_sets:?}"
                )
                .into());
            }
            Ok(!is_shfqa && awg.signals.len() > 1)
        }
    }
}

/// Creates virtual signals and allocates channels for each signal.
///
/// Virtual signals do not include integration signals.
pub(crate) fn create_virtual_signals(awg: &AwgCore) -> Result<Option<VirtualSignals>> {
    let is_multiplexing = is_multiplexing(awg)?;
    let mut virtual_signals: HashMap<Option<u8>, Vec<Channel>> = HashMap::new();
    let mut channel_to_id: HashMap<u8, u16> = HashMap::new();
    for signal_obj in awg.signals.iter() {
        if signal_obj.kind == SignalKind::INTEGRATION {
            continue;
        }
        // The channels are 0..15 for an SHFQA
        let sub_channel = if *awg.device_kind() == DeviceKind::SHFQA {
            Some(signal_obj.channels[0])
        } else {
            None
        };
        let entry = virtual_signals.entry(sub_channel).or_default();

        // The ID is an index into the list of channel numbers. In the case of
        // an RF signal, we map the actual channels to 0 (and maybe 1). In the
        // case of multiplexed IQ signals, we just count up. In any other case,
        // we anyway have only one channel.
        let id = if matches!(awg.kind, AwgKind::DOUBLE | AwgKind::SINGLE) {
            let l = channel_to_id.len();
            *channel_to_id
                .entry(signal_obj.channels[0])
                .or_insert_with(|| l as u16)
        } else if is_multiplexing {
            entry.len() as u16
        } else {
            assert!(entry.is_empty());
            0
        };
        entry.push(Channel {
            id,
            signal: Arc::clone(signal_obj),
        });
    }
    if virtual_signals.is_empty() {
        return Ok(None);
    }
    let virtual_signals: Vec<_> = virtual_signals
        .into_iter()
        .map(|(sub_channel, channels)| VirtualSignal::new(channels, sub_channel))
        .collect();
    for vsig in virtual_signals.iter() {
        validate_signal_oscillators(vsig, awg)?
    }
    let out = VirtualSignals(virtual_signals);
    Ok(Some(out))
}

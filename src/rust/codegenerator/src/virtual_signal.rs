// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use crate::Result;
use crate::ir;
use crate::ir::compilation_job::Signal;
use crate::ir::compilation_job::{self as cjob};
use anyhow::anyhow;
use indexmap::IndexMap;
use std::cmp::max;
use std::rc::Rc;

#[derive(Debug)]
struct Channel {
    pub id: Option<u16>,
    pub signal: Rc<cjob::Signal>,
}

#[derive(Debug)]
pub struct VirtualSignal {
    signals: IndexMap<String, Channel>,
    subchannel: Option<u8>,
    delay: ir::Samples,
}

impl VirtualSignal {
    fn new(channels: Vec<Channel>, subchannel: Option<u8>, delay: ir::Samples) -> Self {
        let signals = channels.into_iter().map(|x| (x.signal.uid.clone(), x));
        VirtualSignal {
            signals: signals.collect(),
            subchannel,
            delay,
        }
    }

    pub fn signals(&self) -> impl Iterator<Item = &Rc<Signal>> {
        self.signals.iter().map(|x| &x.1.signal)
    }

    pub fn subchannel(&self) -> Option<u8> {
        self.subchannel
    }

    pub fn delay(&self) -> ir::Samples {
        self.delay
    }

    pub fn contains_signal(&self, uid: &str) -> bool {
        self.signals.contains_key(uid)
    }

    pub fn is_multiplexed(&self) -> bool {
        self.signals.keys().len() > 1
    }

    pub fn get_channel_by_signal(&self, uid: &str) -> Option<u16> {
        self.signals.get(uid).and_then(|x| x.id)
    }
}

pub struct VirtualSignals(Vec<VirtualSignal>);

impl VirtualSignals {
    pub fn iter(&self) -> impl Iterator<Item = &VirtualSignal> {
        self.0.iter()
    }

    /// Common delay across the contained signals
    ///
    /// Panics if there are no signals
    pub fn delay(&self) -> &ir::Samples {
        // NOTE: All the non-integration signals on a single AWG must have the same delay
        &self.0.first().expect("Virtual signals are empty").delay
    }
}

fn validate_signal_oscillators(signal: &VirtualSignal, awg: &cjob::AwgCore) -> Result<()> {
    // Skip validation for DOUBLE
    if !signal.is_multiplexed() || awg.kind == cjob::AwgKind::DOUBLE {
        return Ok(());
    }
    let mut hw_modulated_signals: Vec<&str> = vec![];
    let mut sw_modulated_signals: Vec<&str> = vec![];
    for channel in signal.signals() {
        if channel.is_sw_modulated() {
            if !hw_modulated_signals.is_empty() {
                let mut signals = hw_modulated_signals.into_iter().collect::<Vec<_>>();
                signals.sort();
                let err_msg = format!(
                    "Attempting to multiplex HW-modulated signal(s) ({}) with signal that is not HW modulated ({}).",
                    signals[0], channel.uid
                );
                return Err(anyhow!(err_msg).into());
            }
            sw_modulated_signals.push(&channel.uid);
        } else if channel.is_hw_modulated() {
            if let Some(sw_modulated_signal) = sw_modulated_signals.first() {
                let err_msg = format!(
                    "Attempting to multiplex HW-modulated signal(s) ({}) with signal that is not HW modulated ({}).",
                    channel.uid, sw_modulated_signal
                );
                return Err(anyhow!(err_msg).into());
            }
            hw_modulated_signals.push(&channel.uid);
        }
    }
    if hw_modulated_signals.len() > 1 && !awg.device_kind.traits().supports_oscillator_switching {
        let mut signals = hw_modulated_signals.into_iter().collect::<Vec<_>>();
        signals.sort();
        let msg = format!(
            "Attempting to multiplex several hardware-modulated signals: \
            '{}' on device '{}', which does not support oscillator switching.",
            signals.join(", "),
            awg.device_kind.as_str()
        );
        return Err(anyhow!(msg).into());
    }
    Ok(())
}

/// Creates virtual signals and allocates channels for each signal.
///
/// Virtual signals do not include integration signals.
pub fn create_virtual_signals(awg: &cjob::AwgCore) -> Result<Option<VirtualSignals>> {
    // TODO: Delays on each virtual signal must be equal
    let virtual_signals = match awg.kind {
        cjob::AwgKind::SINGLE | cjob::AwgKind::IQ => {
            let mut signals = vec![];
            for signal_obj in awg.signals.iter() {
                if signal_obj.kind == cjob::SignalKind::INTEGRATION {
                    continue;
                }
                // TODO: What is the function of the subchannel?
                let sub_channel = {
                    match &awg.device_kind {
                        cjob::DeviceKind::SHFQA => Some(signal_obj.channels[0]),
                        _ => None,
                    }
                };
                let channel = Channel {
                    id: Some(0),
                    signal: signal_obj.clone(),
                };
                let v_sig = VirtualSignal::new(vec![channel], sub_channel, signal_obj.delay);
                signals.push(v_sig);
            }
            signals
        }
        cjob::AwgKind::MULTI => {
            let mut delay = 0;
            let mut channels = vec![];
            for signal_obj in awg.signals.iter() {
                if signal_obj.kind != cjob::SignalKind::INTEGRATION {
                    delay = max(delay, signal_obj.delay);
                    let channel = Channel {
                        signal: signal_obj.clone(),
                        id: Some(channels.len() as u16),
                    };
                    channels.push(channel);
                }
            }
            vec![VirtualSignal::new(channels, None, delay)]
        }
        cjob::AwgKind::DOUBLE => {
            assert_eq!(awg.signals.len(), 2);
            let channels = vec![
                Channel {
                    signal: awg.signals[0].clone(),
                    id: Some(0),
                },
                Channel {
                    signal: awg.signals[1].clone(),
                    id: Some(1),
                },
            ];
            let delay = awg.signals[0].delay;
            vec![VirtualSignal::new(channels, None, delay)]
        }
    };
    if virtual_signals.is_empty() {
        return Ok(None);
    }
    // TODO: Validate that each signal has the same delay
    for vsig in virtual_signals.iter() {
        validate_signal_oscillators(vsig, awg)?
    }
    let out = VirtualSignals(virtual_signals);
    Ok(Some(out))
}

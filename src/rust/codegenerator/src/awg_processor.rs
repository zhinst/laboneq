// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::collections::VecDeque;
use std::{collections::HashMap, sync::Arc};

use laboneq_common::device_options::DeviceOptions;
use laboneq_dsl::types::SignalUid;
use laboneq_error::laboneq_error;

use crate::Result;
use crate::ir::compilation_job::{AwgCore, DeviceKind, SignalKind};
use crate::ir::compilation_job::{Oscillator, OscillatorKind};

pub(crate) fn process_awgs(awgs: &mut [AwgCore]) -> Result<()> {
    // Sort the signals for consistent ordering.
    order_awgs(awgs);
    allocate_shfqa_generator_channels(awgs);
    for awg in awgs.iter_mut() {
        for (signal_uid, osc_index) in allocate_oscillators(awg)? {
            awg.add_oscillator_index(signal_uid, osc_index);
        }
    }
    Ok(())
}

/// Sort AWGs and their signals for consistent ordering.
fn order_awgs(awgs: &mut [AwgCore]) {
    for awg in awgs.iter_mut() {
        awg.signals.sort_by_key(|s| s.uid);
        awg.signals.sort_by(|a, b| a.channels.cmp(&b.channels));
    }
    // Sort the AWGs for consistent ordering.
    awgs.sort_by(|a, b| {
        let a_signals = a
            .signals
            .iter()
            .flat_map(|s| &s.channels)
            .collect::<Vec<_>>();
        let b_signals = b
            .signals
            .iter()
            .flat_map(|s| &s.channels)
            .collect::<Vec<_>>();
        a_signals.cmp(&b_signals)
    });
    awgs.sort_by_key(|a| a.key());
}

/// Allocate generator channels (waveform slots) for SHFQA devices.
///
/// Each signal needs to correspond to a unique waveform slot on the device.
fn allocate_shfqa_generator_channels(awgs: &mut [AwgCore]) {
    let mut shfqa_generator_allocation = HashMap::new();

    for awg in awgs.iter_mut() {
        if awg.device_kind() != &DeviceKind::SHFQA {
            continue;
        }
        let awg_key = awg.key();
        for signal in awg.signals.iter_mut() {
            if signal.kind != SignalKind::IQ {
                continue;
            }
            let generator_channel = *shfqa_generator_allocation
                .entry(awg_key.clone())
                .and_modify(|f| *f += 1)
                .or_insert_with(|| 0);

            let signal_mut =
                Arc::get_mut(signal).expect("Expected to get mutable reference to signal");
            signal_mut.channels = vec![generator_channel];
        }
    }
}

/// Allocate hardware oscillators to signals on the given AWG, based on the AWG's device kind and available oscillators determined by device options.
///
/// Returns a mapping of signal UIDs to allocated hardware oscillator indices, or an error if there are not enough oscillators available.
pub(crate) fn allocate_oscillators(awg: &AwgCore) -> Result<HashMap<SignalUid, u16>> {
    let mut available_oscs =
        available_oscillators(awg.device_kind(), awg.uid, &awg.options, awg.is_shfqc);

    let mut hw_modulated_signals = awg
        .signals
        .iter()
        .filter_map(|s| {
            if let Some(osc) = &s.oscillator
                && osc.kind == OscillatorKind::HARDWARE
            {
                Some((s, osc))
            } else {
                None
            }
        })
        .collect::<Vec<_>>();

    // Sort by channel for stable ordering.
    hw_modulated_signals.sort_by(|a, b| a.0.channels.cmp(&b.0.channels));

    // TODO(2K): This is a workaround for the case where measure and
    // acquire signals on the same QA AWG, or two RF signals on the
    // same HD AWG have different oscillators, but the same fixed
    // frequency. In principle, this shouldn't be allowed, but previous
    // code did allow it, and there are test cases relying on it.
    if hw_modulated_signals.len() == 2
        && signal_can_share_osc(hw_modulated_signals[0].1, hw_modulated_signals[1].1)
        && available_oscs.len() == 1
    {
        let ref_signal = hw_modulated_signals[0].0;
        let index = available_oscs.allocate(ref_signal.uid)?;
        let mut allocation = HashMap::with_capacity(2);
        allocation.insert(hw_modulated_signals[0].0.uid, index);
        allocation.insert(hw_modulated_signals[1].0.uid, index);
        return Ok(allocation);
    }

    let mut signal_osc_index = HashMap::with_capacity(hw_modulated_signals.len());
    let mut osc_uid_to_index = HashMap::new();

    // Exception: for SHFQA generator slot N is hardwired to
    // oscillator N, so IQ (generator) signals must claim oscillator N before integration
    // signals on the same channel, regardless of the oscillator specified.
    if awg.device_kind() == &DeviceKind::SHFQA {
        let (generator_signals, integration): (Vec<_>, Vec<_>) = hw_modulated_signals
            .into_iter()
            .partition(|(signal, _)| signal.kind == SignalKind::IQ);

        let mut channel_osc_index = HashMap::with_capacity(generator_signals.len());

        for (signal, osc) in generator_signals {
            let channel = signal.channels.first().copied().unwrap() as u16;
            if let std::collections::hash_map::Entry::Vacant(e) = channel_osc_index.entry(channel) {
                available_oscs.allocate_specific(channel, signal.uid)?;
                e.insert(osc);
            }
            signal_osc_index.insert(signal.uid, channel);
            osc_uid_to_index.insert(osc.uid.as_str(), channel);
        }

        // Update hw_modulated_signals to only contain non-IQ signals
        hw_modulated_signals = integration;
    }

    // Allocate remaining signals.
    for (signal, osc) in hw_modulated_signals {
        let osc_index = if let Some(osc_index) = osc_uid_to_index.get(osc.uid.as_str()) {
            *osc_index
        } else {
            let osc_index = available_oscs.allocate(signal.uid)?;
            osc_uid_to_index.insert(osc.uid.as_str(), osc_index);
            osc_index
        };
        signal_osc_index.insert(signal.uid, osc_index);
    }

    Ok(signal_osc_index)
}

fn signal_can_share_osc(osc0: &Oscillator, osc1: &Oscillator) -> bool {
    if osc0.kind == OscillatorKind::HARDWARE
        && osc1.kind == OscillatorKind::HARDWARE
        && osc0.frequency == osc1.frequency
    {
        return true;
    }
    false
}

/// Determine the available hardware oscillators for the given AWG.
///
/// Returns a tuple of (available oscillator indices, optional message about missing options).
fn available_oscillators(
    device: &DeviceKind,
    awg_index: u16,
    options: &DeviceOptions,
    is_shfqc: bool,
) -> AvailableOscillators {
    let (available_oscs, opt_msg) = match device {
        DeviceKind::UHFQA => (vec![0], None),
        DeviceKind::HDAWG => {
            if options.contains("MF") {
                const OSC_COUNT_HDAWG_MF_OPT: u16 = 4;

                let start = awg_index * OSC_COUNT_HDAWG_MF_OPT;
                ((start..start + OSC_COUNT_HDAWG_MF_OPT).collect(), None)
            } else {
                (vec![awg_index], Some("Missing MF option?"))
            }
        }
        DeviceKind::SHFQA => {
            if options.contains("LRT") {
                const OSC_COUNT_SHFQA_LRT: u16 = 3;
                const OSC_COUNT_SHFQA_SHFQC: u16 = 6;

                let num_oscs = if is_shfqc {
                    OSC_COUNT_SHFQA_SHFQC
                } else {
                    OSC_COUNT_SHFQA_LRT
                };
                ((0..num_oscs).collect(), None)
            } else {
                (vec![0], Some("Missing LRT option?"))
            }
        }
        DeviceKind::SHFSG => {
            // TODO: Add other SHFSG models.
            const OSC_COUNT_SHFSG8: u16 = 8;

            ((0..OSC_COUNT_SHFSG8).collect(), None)
        }
    };
    AvailableOscillators::new(available_oscs, opt_msg)
}

struct AvailableOscillators {
    opt_msg: Option<&'static str>,
    available_oscs: VecDeque<u16>,
}

impl AvailableOscillators {
    fn new(available_oscs: Vec<u16>, opt_msg: Option<&'static str>) -> Self {
        Self {
            available_oscs: VecDeque::from(available_oscs),
            opt_msg,
        }
    }

    fn len(&self) -> usize {
        self.available_oscs.len()
    }

    /// Allocate the next available oscillator.
    fn allocate(&mut self, signal_uid: SignalUid) -> Result<u16> {
        if let Some(osc_index) = self.available_oscs.pop_front() {
            Ok(osc_index)
        } else {
            Err(laboneq_error!("{}", self.create_error_message(signal_uid)))
        }
    }

    /// Allocate a specific oscillator index, if available.
    fn allocate_specific(&mut self, osc_index: u16, signal_uid: SignalUid) -> Result<()> {
        if let Some(pos) = self.available_oscs.iter().position(|&x| x == osc_index) {
            self.available_oscs.remove(pos);
            Ok(())
        } else {
            Err(laboneq_error!("{}", self.create_error_message(signal_uid)))
        }
    }

    fn create_error_message(&self, signal_uid: SignalUid) -> String {
        let mut msg = format!(
            "No free hardware oscillators available for signal '{}'",
            signal_uid.0
        );
        if let Some(opt_msg) = self.opt_msg {
            msg.push_str(&format!(" {opt_msg}"));
        }
        msg
    }
}

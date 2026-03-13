// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, VecDeque};

use laboneq_common::device_options::DeviceOptions;
use laboneq_dsl::types::SignalUid;
use laboneq_error::laboneq_error;

use crate::Result;
use crate::ir::compilation_job::{AwgCore, DeviceKind, Oscillator, OscillatorKind};

/// Allocate hardware oscillators to signals on the given AWG, based on the AWG's device kind and available oscillators determined by device options.
///
/// Returns a mapping of signal UIDs to allocated hardware oscillator indices, or an error if there are not enough oscillators available.
pub(crate) fn allocate_oscillators(awg: &AwgCore) -> Result<HashMap<SignalUid, u16>> {
    // Get list of available hardware oscillators for this AWG, and prepare for allocation.
    let (available_oscs_vec, opt_msg) =
        available_oscillators(awg.device_kind(), awg.uid, &awg.options, awg.is_shfqc);
    let mut available_oscs = VecDeque::from(available_oscs_vec);

    let mut hw_modulated_signals = awg
        .signals
        .iter()
        .filter_map(|s| {
            if let Some(osc) = &s.oscillator
                && osc.kind == OscillatorKind::HARDWARE
            {
                Some((s.uid, osc))
            } else {
                None
            }
        })
        .collect::<Vec<_>>();

    // TODO(2K): This is a workaround for the case where measure and
    // acquire signals on the same QA AWG, or two RF signals on the
    // same HD AWG have different oscillators, but the same fixed
    // frequency. In principle, this shouldn't be allowed, but previous
    // code did allow it, and there are test cases relying on it.
    if hw_modulated_signals.len() == 2
        && signal_can_share_osc(hw_modulated_signals[0].1, hw_modulated_signals[1].1)
        && available_oscs.len() == 1
    {
        let index = available_oscs.pop_front().unwrap();
        let mut allocation = HashMap::new();
        allocation.insert(hw_modulated_signals[0].0, index);
        allocation.insert(hw_modulated_signals[1].0, index);
        return Ok(allocation);
    }

    // Sort signals by oscillator UID for stable allocation order and backwards compatibility.
    hw_modulated_signals.sort_by(|a, b| a.1.uid.cmp(&b.1.uid));

    let mut oscillator_allocation: HashMap<&str, (u16, Vec<SignalUid>)> = HashMap::new();
    for (signal_uid, osc) in hw_modulated_signals {
        if let Some((_, signals)) = oscillator_allocation.get_mut(osc.uid.as_str()) {
            signals.push(signal_uid);
            continue;
        }
        if let Some(osc_index) = available_oscs.pop_front() {
            oscillator_allocation.insert(osc.uid.as_str(), (osc_index, vec![signal_uid]));
        } else {
            let mut msg = format!(
                "No free hardware oscillators available for signal '{}', on device '{}', AWG '{}'.",
                osc.uid,
                awg.device.uid(),
                awg.uid
            );
            if let Some(opt_msg) = opt_msg {
                msg.push_str(&format!(" {opt_msg}"));
            }
            return Err(laboneq_error!("{}", msg));
        }
    }

    Ok(oscillator_allocation
        .into_values()
        .flat_map(|(index, signals)| {
            signals
                .into_iter()
                .map(move |signal_uid| (signal_uid, index))
        })
        .collect())
}

/// Determine the available hardware oscillators for the given AWG.
///
/// Returns a tuple of (available oscillator indices, optional message about missing options).
fn available_oscillators(
    device: &DeviceKind,
    awg_index: u16,
    options: &DeviceOptions,
    is_shfqc: bool,
) -> (Vec<u16>, Option<&'static str>) {
    match device {
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
    }
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

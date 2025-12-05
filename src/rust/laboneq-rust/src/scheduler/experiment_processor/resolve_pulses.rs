// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use crate::error::{Error, Result};
use crate::scheduler::experiment::{Experiment, Signal};
use crate::scheduler::pulse::{PulseDef, PulseKind};
use laboneq_common::named_id::NamedIdStore;
use laboneq_scheduler::experiment::ExperimentNode;
use laboneq_scheduler::experiment::types::{Acquire, Operation, PulseUid, SignalUid};
use laboneq_units::duration::{Duration, Second, seconds};

/// Resolve missing pulses.
///
/// For acquire operations with a length, but without a defined kernel, a pulse is created with length only.
/// For acquire operations with defined kernel but missing length, the length is determined from
/// the pulses in the kernel.
pub(super) fn resolve_pulses(experiment: &mut Experiment) -> Result<()> {
    let mut ctx = Context {
        id_store: &mut experiment.id_store,
        pulses: &mut experiment.pulses,
        signals: &experiment.signals,
    };
    for section in experiment.sections.iter_mut() {
        resolve_pulses_impl(section, &mut ctx)?;
    }
    Ok(())
}

struct Context<'a> {
    id_store: &'a mut NamedIdStore,
    pulses: &'a mut HashMap<PulseUid, PulseDef>,
    signals: &'a HashMap<SignalUid, Signal>,
}

fn resolve_pulses_impl(node: &mut ExperimentNode, ctx: &mut Context) -> Result<()> {
    for child in node.children.iter_mut() {
        let child = child.make_mut();
        match &mut child.kind {
            Operation::Acquire(obj) => {
                resolve_acquire_pulses_and_length(obj, ctx.id_store, ctx.pulses, ctx.signals)?;
            }
            _ => {
                resolve_pulses_impl(child, ctx)?;
            }
        }
    }
    Ok(())
}

fn resolve_acquire_pulses_and_length(
    obj: &mut Acquire,
    id_store: &mut NamedIdStore,
    pulses: &mut HashMap<PulseUid, PulseDef>,
    signals: &HashMap<SignalUid, Signal>,
) -> Result<()> {
    if obj.kernel.is_empty() && obj.length.is_none() {
        return Err(Error::new(
            "Acquire operation must have either a kernel or a length defined.",
        ));
    }
    let signal = signals.get(&obj.signal).unwrap_or_else(|| {
        panic!(
            "Internal Error: Expected signal '{}' to be present",
            obj.signal.0
        )
    });

    if obj.kernel.is_empty()
        && let Some(length) = &obj.length
    {
        // TODO: Do we want to auto-generate pulses here?
        // For now we do the same thing as Python, create a pulse without a function so that
        // no kernels are generated for these pulses.
        let pulse = create_length_only_acquisition_pulse(*length, id_store, pulses);
        obj.kernel = vec![pulse];
    }
    if obj.length.is_none() {
        obj.length = find_maximum_pulse_length(&obj.kernel, pulses, signal.sampling_rate).into();
    }
    Ok(())
}

fn find_maximum_pulse_length(
    pulse_uids: &[PulseUid],
    pulses: &HashMap<PulseUid, PulseDef>,
    sampling_rate: f64,
) -> Duration<Second> {
    pulse_uids
        .iter()
        .map(|pulse_uid| {
            let pulse_ref = pulses.get(pulse_uid).expect("Expected pulse");
            determine_pulse_length(pulse_ref, sampling_rate)
        })
        .max()
        .unwrap_or_else(|| unreachable!("Expected at least one pulse uid"))
}

fn determine_pulse_length(pulse_def: &PulseDef, sampling_rate: f64) -> Duration<Second> {
    match &pulse_def.kind {
        PulseKind::Functional(func) => func.length,
        PulseKind::Sampled(obj) => {
            let num_samples = obj.length as f64;
            seconds(num_samples / sampling_rate)
        }
        PulseKind::LengthOnly { length } => *length,
    }
}

fn create_length_only_acquisition_pulse(
    length: Duration<Second>,
    id_store: &mut NamedIdStore,
    pulses: &mut HashMap<PulseUid, PulseDef>,
) -> PulseUid {
    // TODO: Cache these pulses to avoid duplicates? Implement hash for Duration?
    // TODO: Use a better naming scheme
    let suffix = pulses.len();
    let uid_string = format!("laboneq_acquire_pulse_{suffix}");
    let uid = PulseUid(id_store.get_or_insert(&uid_string));
    let pulse = PulseDef {
        uid,
        kind: PulseKind::LengthOnly { length },
        amplitude: 1.0.into(),
        can_compress: true,
    };
    pulses.insert(pulse.uid, pulse);
    uid
}

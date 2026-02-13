// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::sync::Arc;

use crate::error::{Error, Result};
use crate::scheduler::experiment::Experiment;
use crate::scheduler::signal_view::SignalView;
use laboneq_common::named_id::NamedIdStore;
use laboneq_dsl::ExperimentNode;
use laboneq_dsl::operation::{Acquire, Operation, PlayPulse};
use laboneq_dsl::types::{Marker, PulseUid, SignalUid, ValueOrParameter};
use laboneq_py_utils::pulse::{PulseDef, PulseKind};
use laboneq_units::duration::{Duration, Second, seconds};

/// Resolve missing pulses.
///
/// For acquire operations with a length, but without a defined kernel, a pulse is created with length only.
/// For acquire operations with defined kernel but missing length, the length is determined from
/// the pulses in the kernel.
///
/// For play pulse operations with markers but without a defined pulse, a zero-amplitude pulse is created
/// to play the markers.
/// For play pulse operations with a defined pulse but missing length, the length is determined from
/// the pulse.
pub(super) fn resolve_pulses(
    experiment: &mut Experiment,
    signals: &HashMap<SignalUid, SignalView>,
) -> Result<()> {
    let mut ctx = Context {
        id_store: Arc::get_mut(&mut experiment.id_store).unwrap(),
        pulses: &mut experiment.pulses,
        signals,
    };
    resolve_pulses_impl(&mut experiment.root, &mut ctx)?;
    Ok(())
}

struct Context<'a> {
    id_store: &'a mut NamedIdStore,
    pulses: &'a mut HashMap<PulseUid, PulseDef>,
    signals: &'a HashMap<SignalUid, SignalView<'a>>,
}

fn resolve_pulses_impl(node: &mut ExperimentNode, ctx: &mut Context) -> Result<()> {
    for child in node.children.iter_mut() {
        let child = child.make_mut();
        match &mut child.kind {
            Operation::Acquire(obj) => {
                resolve_acquire_pulses_and_length(obj, ctx.id_store, ctx.pulses, ctx.signals)?;
            }
            Operation::PlayPulse(obj) => process_play_pulses(obj, ctx)?,
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
    signals: &HashMap<SignalUid, SignalView<'_>>,
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
        obj.length = find_maximum_pulse_length(&obj.kernel, pulses, signal.sampling_rate()).into();
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
        PulseKind::MarkerPulse { length } => *length,
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

/// Process play pulse operation to resolve missing pulses and lengths.
///
/// - If a pulse is defined but length is missing, the length is determined from the pulse.
/// - If markers are defined but no pulse, a zero-amplitude pulse is created to play the markers.
/// - If no pulse is defined, check for virtual Z gate and set length to zero.
fn process_play_pulses(obj: &mut PlayPulse, ctx: &mut Context<'_>) -> Result<()> {
    let sampling_rate = ctx.signals.get(&obj.signal).unwrap().sampling_rate();
    if let Some(pulse) = obj.pulse
        && obj.length.is_none()
    {
        let length = determine_pulse_length(ctx.pulses.get(&pulse).unwrap(), sampling_rate);
        obj.length = Some(ValueOrParameter::Value(length));
        return Ok(());
    }
    if !obj.markers.is_empty() && obj.pulse.is_none() {
        // Generate a zero-amplitude pulse to play the markers
        let length = determine_marker_pulse_length(&obj.markers, ctx.pulses, sampling_rate)?;
        let uid = create_zero_amplitude_marker_pulse(length, ctx.id_store, ctx.pulses);
        obj.pulse = Some(uid);
        obj.length = Some(ValueOrParameter::Value(length));
    }
    if obj.pulse.is_none() {
        // Check for virtual Z gate
        validate_virtual_z_gate_fields(obj)?;
        obj.length = Some(ValueOrParameter::Value(seconds(0.0)));
    }
    Ok(())
}

/// Determine the maximum length of all markers.
fn determine_marker_pulse_length(
    markers: &[Marker],
    pulses: &HashMap<PulseUid, PulseDef>,
    sampling_rate: f64,
) -> Result<Duration<Second>> {
    assert!(!markers.is_empty(), "Expected at least one marker");
    let mut max_length: Option<Duration<Second>> = None;
    for marker in markers {
        let length = if let Some(marker_pulse) = marker.pulse_id {
            Ok(determine_pulse_length(
                pulses.get(&marker_pulse).unwrap(),
                sampling_rate,
            ))
        } else if let (Some(start), Some(length)) = (marker.start, marker.length) {
            Ok(seconds(start.value() + length.value()))
        } else {
            Err(Error::new(
                "Missing marker information. Please specify a start and length or a waveform for a play command without pulse and enabled marker(s).",
            ))
        }?;
        max_length = max_length
            .map(|max_length| max_length.max(length))
            .or(Some(length));
    }
    Ok(max_length.unwrap())
}

/// Create a zero-amplitude marker pulse with the given length.
///
/// The pulse uid is prefixed with `__marker__` to indicate it's a marker pulse for compatibility with Python side.
/// This pulse is inserted into the provided `pulses`.
fn create_zero_amplitude_marker_pulse(
    length: Duration<Second>,
    id_store: &mut NamedIdStore,
    pulses: &mut HashMap<PulseUid, PulseDef>,
) -> PulseUid {
    let suffix = pulses.len();
    // Prefix the uid with `__marker__` to indicate it's a marker pulse for compatibility with Python side.
    let uid_string = format!("__marker__laboneq_marker_pulse_{suffix}");
    let uid = PulseUid(id_store.get_or_insert(&uid_string));
    let pulse = PulseDef {
        uid,
        kind: PulseKind::MarkerPulse { length },
        amplitude: 0.0.into(),
        can_compress: false,
    };
    pulses.insert(pulse.uid, pulse);
    uid
}

fn validate_virtual_z_gate_fields(obj: &PlayPulse) -> Result<()> {
    // Internal consistency checks, if no pulse, no pulse_parameters should be set
    assert!(obj.pulse.is_none(), "Expected no pulse for virtual Z gate");
    assert!(
        obj.pulse_parameters.is_empty(),
        "Expected no pulse parameters for virtual Z gate"
    );
    // TODO: Amplitude defaults to 1.0, should we check that too?
    let fields_must_be_empty = [
        (obj.length.is_none(), "length"),
        (obj.phase.is_none(), "phase"),
        (obj.markers.is_empty(), "markers"),
        (obj.parameters.is_empty(), "pulse_parameters"), // The public API is `pulse_parameters`
    ];
    for (is_empty, field_name) in &fields_must_be_empty {
        if !is_empty {
            return Err(Error::new(format!(
                "Field '{}' cannot be used when 'pulse' is not set in play operations.",
                field_name
            )));
        }
    }
    Ok(())
}

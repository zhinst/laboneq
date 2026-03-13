// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

//! Deserializes Cap'n Proto binary format to Rust DSL experiment types.
//!
//! Reads Cap'n Proto messages produced by `capnp_serializer` and reconstructs
//! the internal `ExperimentNode`, `NamedIdStore`, sweep parameters, and pulse
//! definitions used by the compiler.
//!
//! Reference handling: Signals, sweep parameters, pulses, and acquisition
//! handles are referenced by zero-based indices mapped to `uid`/`name` strings.
//! PRNG data is inlined in section structs and resolved via text identifiers.
//! Sections may be named or unnamed. Named sections map to
//! `NamedIdStore` entries via their names, while unnamed sections receive
//! auto-generated `_s_<n>` names registered in the `NamedIdStore`, matching
//! the legacy compiler path.

use std::collections::{HashMap, HashSet};
use std::num::NonZeroU32;
use std::sync::Arc;

use crate::error::{Error, Result};

use laboneq_capnp::pulse::v1::{
    common_capnp, experiment_capnp, operation_capnp, pulse_capnp, section_capnp, sweep_capnp,
};
use laboneq_common::named_id::{NamedId, NamedIdStore};
use laboneq_dsl::ExperimentNode;
use laboneq_dsl::operation::{
    Acquire, AveragingLoop, Case, Chunking, Delay, Match, Operation, PlayPulse, PrngLoop,
    PrngSetup, PulseParameterValue, Reserve, ResetOscillatorPhase, Section, Sweep,
};
use laboneq_dsl::types::{
    AcquisitionType, AveragingMode, ComplexOrFloat, ExternalParameterUid, HandleUid, Marker,
    MarkerSelector, MatchTarget, NumericLiteral, ParameterUid, PrngSampleUid, PulseParameterUid,
    PulseUid, RepetitionMode, SectionAlignment, SectionUid, SignalUid, SweepParameter, Trigger,
    ValueOrParameter,
};
use laboneq_py_utils::pulse::{FunctionalPulse, PulseDef, PulseFunction, PulseKind, SampledPulse};
use laboneq_units::duration::seconds;
use num_complex::Complex64;
use numeric_array::NumericArray;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyString};

type ExternalParameterStore = HashMap<ExternalParameterUid, Py<PyAny>>;

/// Result of deserializing a Cap'n Proto experiment message.
pub(crate) struct DeserializedExperiment {
    pub root: ExperimentNode,
    pub id_store: NamedIdStore,
    pub parameters: HashMap<ParameterUid, SweepParameter>,
    pub pulses: HashMap<PulseUid, PulseDef>,
    pub available_signals: HashSet<SignalUid>,
    pub external_parameter_values: ExternalParameterStore,
}

// === Pure helper functions ===

/// Helper to convert capnp text reader to &str.
fn text_to_str(reader: capnp::text::Reader<'_>) -> Result<&str> {
    reader
        .to_str()
        .map_err(|e| Error::new(format!("Invalid UTF-8 in Cap'n Proto text: {e}")))
}

/// Derive an `ExternalParameterUid` by SHA-256 hashing raw bytes.
///
/// Uses the same algorithm as `PyObjectInterner::get_or_intern` to ensure
/// UID consistency. Works for both pickled payloads and raw binary data.
fn uid_from_bytes(py: Python<'_>, pickled: &[u8]) -> Result<ExternalParameterUid> {
    let sha256 = py
        .import(pyo3::intern!(py, "hashlib"))?
        .getattr(pyo3::intern!(py, "sha256"))?
        .call1((PyBytes::new(py, pickled),))?;
    let digest = sha256.call_method0(pyo3::intern!(py, "digest"))?;
    let digest = digest.cast::<PyBytes>().map_err(Error::new)?.as_bytes();
    let uid = u64::from_be_bytes(
        digest[..digest.len().min(8)]
            .try_into()
            .map_err(|_| Error::new("Failed to derive external parameter UID"))?,
    );
    Ok(ExternalParameterUid(uid))
}

/// Derive an `ExternalParameterUid` from a Python object by pickling it and
/// hashing the result. Use `uid_from_bytes` when you already have the
/// serialized representation to avoid redundant pickling.
fn intern_py_object_uid(value: &Bound<'_, PyAny>) -> Result<ExternalParameterUid> {
    let py = value.py();
    let pickled_bytes = py
        .import(pyo3::intern!(py, "pickle"))?
        .getattr(pyo3::intern!(py, "dumps"))?
        .call1((value,))?;
    let pickled = pickled_bytes
        .cast::<PyBytes>()
        .map_err(Error::new)?
        .as_bytes();
    uid_from_bytes(py, pickled)
}

/// Validate the schema version in the experiment metadata.
///
/// During the 0.x development phase, the exact version must match. From 1.0
/// onward the schema follows semver: the deserializer accepts any message whose
/// major version matches and whose minor version is less than or equal to the
/// deserializer's minor version (i.e. the client may be older than the server).
fn validate_schema_version(experiment: &experiment_capnp::experiment::Reader<'_>) -> Result<()> {
    use experiment_capnp::SCHEMA_VERSION as CURRENT_SCHEMA_VERSION;

    let metadata = experiment.get_metadata().map_err(Error::new)?;
    let version = text_to_str(metadata.get_schema_version().map_err(Error::new)?)?;

    if version.is_empty() {
        return Err(Error::new("Experiment metadata is missing schema_version"));
    }

    // Parse major.minor from both the message and the current version.
    let parse_version = |v: &str| -> Option<(u32, u32)> {
        let mut parts = v.split('.');
        let major = parts.next()?.parse::<u32>().ok()?;
        let minor = parts.next()?.parse::<u32>().ok()?;
        Some((major, minor))
    };

    let (msg_major, msg_minor) = parse_version(version).ok_or_else(|| {
        Error::new(format!(
            "Invalid schema_version format: '{version}' (expected 'major.minor')"
        ))
    })?;

    let (cur_major, cur_minor) = parse_version(CURRENT_SCHEMA_VERSION).unwrap();

    if msg_major == 0 && cur_major == 0 {
        // 0.x development phase: exact minor version must match.
        if msg_minor != cur_minor {
            return Err(Error::new(format!(
                "Schema version mismatch: message has version '{version}' but this \
                 compiler expects '{CURRENT_SCHEMA_VERSION}'. During the 0.x development \
                 phase, exact version match is required."
            )));
        }
    } else if msg_major != cur_major {
        return Err(Error::new(format!(
            "Schema version mismatch: message has major version {msg_major} but this \
             compiler supports major version {cur_major}. A different compiler version \
             is required."
        )));
    } else if msg_minor > cur_minor {
        return Err(Error::new(format!(
            "Schema version mismatch: message has version '{version}' but this compiler \
             only supports up to '{CURRENT_SCHEMA_VERSION}'. Please upgrade the compiler."
        )));
    }
    // msg_minor <= cur_minor: compatible (older client, newer server). OK.

    Ok(())
}

fn is_autogenerated_section_name(name: &str) -> bool {
    if let Some(rest) = name.strip_prefix("_s_") {
        !rest.is_empty() && rest.as_bytes().iter().all(|b| b.is_ascii_digit())
    } else {
        false
    }
}

fn has_user_provided_section_name(name: &str) -> bool {
    !name.is_empty() && !is_autogenerated_section_name(name)
}

fn section_label_for_errors(section_name: &str) -> String {
    if has_user_provided_section_name(section_name) {
        format!("Section '{section_name}'")
    } else {
        "Section".to_owned()
    }
}

fn deserialize_alignment(alignment: section_capnp::Alignment) -> SectionAlignment {
    match alignment {
        section_capnp::Alignment::Left => SectionAlignment::Left,
        section_capnp::Alignment::Right => SectionAlignment::Right,
        section_capnp::Alignment::Unspecified => SectionAlignment::Left,
    }
}

fn resolve_play_after_list(
    play_after_list: capnp::text_list::Reader<'_>,
    current_section_name: &str,
    sibling_refs: &SiblingSectionRefs,
) -> Result<Vec<SectionUid>> {
    let section_label = section_label_for_errors(current_section_name);
    let mut result = Vec::with_capacity(play_after_list.len() as usize);
    for name in play_after_list.iter() {
        let name = text_to_str(name.map_err(Error::new)?)?;
        let target = sibling_refs.by_name.get(name).copied().ok_or_else(|| {
            Error::new(format!(
                "{} should play after section '{}', \
                 but section '{}' is not defined in the experiment.",
                section_label, name, name
            ))
        })?;
        result.push(target);
    }
    Ok(result)
}

fn deserialize_play_after_from_regular(
    reader: &section_capnp::regular_section::Reader<'_>,
    current_section_name: &str,
    sibling_refs: &SiblingSectionRefs,
) -> Result<Vec<SectionUid>> {
    resolve_play_after_list(
        reader.get_play_after().map_err(Error::new)?,
        current_section_name,
        sibling_refs,
    )
}

fn deserialize_play_after_from_match(
    reader: &section_capnp::match_section::Reader<'_>,
    current_section_name: &str,
    sibling_refs: &SiblingSectionRefs,
) -> Result<Vec<SectionUid>> {
    resolve_play_after_list(
        reader.get_play_after().map_err(Error::new)?,
        current_section_name,
        sibling_refs,
    )
}

// === Pure sweep / constant deserialization ===

fn deserialize_sweep_parameter(
    uid: ParameterUid,
    reader: &sweep_capnp::sweep_parameter::Reader<'_>,
) -> Result<SweepParameter> {
    use sweep_capnp::sweep_parameter::Which;

    let values = match reader.which().map_err(Error::new)? {
        Which::None(()) => {
            let alias = text_to_str(reader.get_uid().map_err(Error::new)?).unwrap_or("<unknown>");
            return Err(Error::new(format!(
                "Sweep parameter '{alias}' has no values"
            )));
        }
        Which::Linear(linear) => {
            let linear = linear.map_err(Error::new)?;
            let count = linear.get_count() as usize;
            deserialize_linear_sweep(&linear, count)?
        }
        Which::ExplicitValues(explicit) => {
            let explicit = explicit.map_err(Error::new)?;
            deserialize_explicit_sweep(&explicit)?
        }
    };

    SweepParameter::new(uid, values).map_err(Error::new)
}

fn deserialize_linear_sweep(
    reader: &sweep_capnp::linear_sweep::Reader<'_>,
    count: usize,
) -> Result<NumericArray> {
    use sweep_capnp::linear_sweep::start::Which as StartWhich;
    use sweep_capnp::linear_sweep::stop::Which as StopWhich;

    let start_which = reader.get_start().which().map_err(Error::new)?;
    let stop_which = reader.get_stop().which().map_err(Error::new)?;

    match (start_which, stop_which) {
        (StartWhich::Real(start), StopWhich::Real(stop)) => {
            Ok(NumericArray::linspace(start, stop, count))
        }
        (StartWhich::Complex(start_reader), StopWhich::Complex(stop_reader)) => {
            let start_reader = start_reader.map_err(Error::new)?;
            let stop_reader = stop_reader.map_err(Error::new)?;
            let start = Complex64::new(start_reader.get_real(), start_reader.get_imag());
            let stop = Complex64::new(stop_reader.get_real(), stop_reader.get_imag());
            Ok(NumericArray::linspace_complex(start, stop, count))
        }
        _ => Err(Error::new(
            "Linear sweep start and stop must be of the same type",
        )),
    }
}

fn deserialize_explicit_sweep(
    reader: &sweep_capnp::explicit_sweep::Reader<'_>,
) -> Result<NumericArray> {
    use sweep_capnp::explicit_sweep::Which;
    match reader.which().map_err(Error::new)? {
        Which::RealValues(r) => {
            let vals: Vec<f64> = r.map_err(Error::new)?.iter().collect();
            Ok(NumericArray::Float64(vals))
        }
        Which::IntValues(r) => {
            let vals: Vec<i64> = r.map_err(Error::new)?.iter().collect();
            Ok(NumericArray::Integer64(vals))
        }
        Which::ComplexValues(r) => {
            let list = r.map_err(Error::new)?;
            let mut vals = Vec::with_capacity(list.len() as usize);
            for cv in list.iter() {
                vals.push(Complex64::new(cv.get_real(), cv.get_imag()));
            }
            Ok(NumericArray::Complex64(vals))
        }
    }
}

fn deserialize_constant(reader: &common_capnp::constant::Reader<'_>) -> Result<NumericLiteral> {
    use common_capnp::constant::Which;
    match reader.which().map_err(Error::new)? {
        Which::Real(v) => Ok(NumericLiteral::Float(v)),
        Which::Integer(v) => Ok(NumericLiteral::Int(v)),
        Which::Complex(cv) => {
            let cv = cv.map_err(Error::new)?;
            Ok(NumericLiteral::Complex(Complex64::new(
                cv.get_real(),
                cv.get_imag(),
            )))
        }
        Which::StringValue(_) => Err(Error::new("String values not supported as numeric")),
        Which::PickledValue(_) | Which::RawBytesValue(_) => {
            Err(Error::new("Bytes values not supported as numeric"))
        }
    }
}

fn deserialize_value_as_numeric(
    reader: &common_capnp::value::Reader<'_>,
) -> Result<Option<NumericLiteral>> {
    use common_capnp::value::Which;
    match reader.which().map_err(Error::new)? {
        Which::None(()) => Ok(None),
        Which::Constant(constant) => {
            let constant = constant.map_err(Error::new)?;
            Ok(Some(deserialize_constant(&constant)?))
        }
        Which::ParameterRef(_) => Err(Error::new(
            "Expected a constant value, got parameter reference",
        )),
    }
}

fn deserialize_acquisition_type(acq_type: operation_capnp::AcquisitionType) -> AcquisitionType {
    match acq_type {
        operation_capnp::AcquisitionType::Integration
        | operation_capnp::AcquisitionType::Unspecified => AcquisitionType::Integration,
        operation_capnp::AcquisitionType::Raw => AcquisitionType::Raw,
        operation_capnp::AcquisitionType::Discrimination => AcquisitionType::Discrimination,
        operation_capnp::AcquisitionType::SpectroscopyIq => AcquisitionType::Spectroscopy,
        operation_capnp::AcquisitionType::SpectroscopyPsd => AcquisitionType::SpectroscopyPsd,
    }
}

fn marker_spec_is_set(reader: &operation_capnp::marker_spec::Reader<'_>) -> Result<bool> {
    use operation_capnp::marker_spec::length::Which as LenWhich;
    use operation_capnp::marker_spec::start::Which as StartWhich;

    if reader.get_enable() {
        return Ok(true);
    }

    let has_start = !matches!(
        reader.get_start().which().map_err(Error::new)?,
        StartWhich::None(())
    );
    let has_length = !matches!(
        reader.get_length().which().map_err(Error::new)?,
        LenWhich::None(())
    );
    let has_waveform = reader.get_waveform() != common_capnp::NONE_ID;

    Ok(has_start || has_length || has_waveform)
}

// === Pure section deserialization (no Deserializer state needed) ===

fn deserialize_acquire_loop(
    reader: &section_capnp::acquire_loop_section::Reader<'_>,
    section_uid: SectionUid,
) -> Result<AveragingLoop> {
    let alignment = deserialize_alignment(reader.get_alignment().map_err(Error::new)?);
    let raw_count = reader.get_count();
    if raw_count > u32::MAX as u64 {
        return Err(Error::new(format!(
            "AcquireLoop count {} exceeds u32::MAX",
            raw_count
        )));
    }
    let count = NonZeroU32::new(raw_count as u32)
        .ok_or_else(|| Error::new("Sweep 'count' must be a positive integer"))?;

    let averaging_mode = match reader.get_averaging_mode().map_err(Error::new)? {
        section_capnp::AveragingMode::Sequential => AveragingMode::Sequential,
        section_capnp::AveragingMode::Cyclic | section_capnp::AveragingMode::Unspecified => {
            AveragingMode::Cyclic
        }
        section_capnp::AveragingMode::SingleShot => AveragingMode::SingleShot,
    };

    let acquisition_type =
        deserialize_acquisition_type(reader.get_acquisition_type().map_err(Error::new)?);

    use section_capnp::acquire_loop_section::repetition::Which as RepWhich;
    let repetition_mode = match reader.get_repetition().which().map_err(Error::new)? {
        RepWhich::Fastest(()) | RepWhich::Unspecified(()) => RepetitionMode::Fastest,
        RepWhich::Constant(t) => RepetitionMode::Constant { time: seconds(t) },
        RepWhich::Auto(()) => RepetitionMode::Auto,
    };

    Ok(AveragingLoop {
        uid: section_uid,
        count,
        acquisition_type,
        averaging_mode,
        repetition_mode,
        reset_oscillator_phase: reader.get_reset_oscillator_phase(),
        alignment,
    })
}

fn deserialize_case_section(
    reader: &section_capnp::case_section::Reader<'_>,
    section_uid: SectionUid,
) -> Result<Case> {
    let state_reader = reader.get_state().map_err(Error::new)?;
    let state = deserialize_value_as_numeric(&state_reader)?
        .ok_or_else(|| Error::new("Case state must be a constant value"))?;

    Ok(Case {
        uid: section_uid,
        state,
    })
}

// === SectionUidGenerator ===

/// Allocates `_s_<n>` section names for unnamed sections, matching the legacy
/// compiler path. Names are registered in the `NamedIdStore` so they resolve
/// correctly in error messages and event lists.
struct SectionUidGenerator {
    next_index: u32,
}

impl SectionUidGenerator {
    fn new() -> Self {
        Self { next_index: 0 }
    }

    fn next_uid(&mut self, id_store: &mut NamedIdStore) -> SectionUid {
        loop {
            let name = format!("_s_{}", self.next_index);
            self.next_index = self
                .next_index
                .checked_add(1)
                .expect("Exhausted internal section UID space");
            // Skip indices already claimed by user-provided `_s_<n>` names.
            if id_store.get(&name).is_none() {
                return SectionUid(id_store.get_or_insert(&name));
            }
        }
    }
}

// === Per-parent-level section index ===

#[derive(Default)]
struct SiblingSectionRefs {
    by_name: HashMap<String, SectionUid>,
}

struct ChildSectionIndex {
    section_uids: Vec<Option<SectionUid>>,
    sibling_refs: SiblingSectionRefs,
}

impl ChildSectionIndex {
    fn section_uid_for_index(&self, idx: usize) -> Result<SectionUid> {
        self.section_uids
            .get(idx)
            .and_then(|v| *v)
            .ok_or_else(|| Error::new("Internal error: missing preallocated section UID"))
    }
}

// === Deserializer ===

/// Bundles all shared state for a single experiment deserialization pass.
struct Deserializer<'py> {
    py: Python<'py>,
    id_store: NamedIdStore,
    signal_ids: Vec<NamedId>,
    parameter_ids: Vec<NamedId>,
    pulse_ids: Vec<NamedId>,
    handle_ids: Vec<NamedId>,
    section_uid_gen: SectionUidGenerator,
    external_parameter_values: ExternalParameterStore,
    available_signals: HashSet<SignalUid>,
    parameters: HashMap<ParameterUid, SweepParameter>,
    pulses: HashMap<PulseUid, PulseDef>,
    pulse_definition_params: HashMap<PulseUid, HashMap<PulseParameterUid, PulseParameterValue>>,
}

impl<'py> Deserializer<'py> {
    fn new(py: Python<'py>) -> Self {
        Self {
            py,
            id_store: NamedIdStore::new(),
            signal_ids: Vec::new(),
            parameter_ids: Vec::new(),
            pulse_ids: Vec::new(),
            handle_ids: Vec::new(),
            section_uid_gen: SectionUidGenerator::new(),
            external_parameter_values: ExternalParameterStore::new(),
            available_signals: HashSet::new(),
            parameters: HashMap::new(),
            pulses: HashMap::new(),
            pulse_definition_params: HashMap::new(),
        }
    }

    /// Main orchestrator — consumes `self` so fields can be moved into the
    /// result without cloning.
    fn deserialize(
        mut self,
        experiment: &experiment_capnp::experiment::Reader<'_>,
    ) -> Result<DeserializedExperiment> {
        validate_schema_version(experiment)?;
        if !experiment.get_device_setup().is_null() {
            return Err(Error::new(
                "Experiment.deviceSetup is reserved and must not be set".to_string(),
            ));
        }
        self.prepass_register_uids(experiment)?;
        self.register_signals();
        self.deserialize_sweep_params(experiment)?;
        self.deserialize_pulses(experiment)?;
        let root = self.deserialize_root_sections(experiment)?;

        Ok(DeserializedExperiment {
            root,
            id_store: self.id_store,
            parameters: self.parameters,
            pulses: self.pulses,
            available_signals: self.available_signals,
            external_parameter_values: self.external_parameter_values,
        })
    }

    fn register_required_alias(
        &mut self,
        alias: &str,
        entity: &str,
        idx: usize,
    ) -> Result<NamedId> {
        if alias.is_empty() {
            return Err(Error::new(format!(
                "Empty alias for {entity} at index {idx}"
            )));
        }
        Ok(self.id_store.get_or_insert(alias))
    }

    fn resolve_index(ids: &[NamedId], idx: u32, entity: &str) -> Result<NamedId> {
        ids.get(idx as usize)
            .copied()
            .ok_or_else(|| Error::new(format!("Unknown {entity} index reference: {idx}")))
    }

    fn resolve_signal_index(&self, idx: u32) -> Result<NamedId> {
        Self::resolve_index(&self.signal_ids, idx, "signal")
    }

    fn resolve_parameter_index(&self, idx: u32) -> Result<NamedId> {
        Self::resolve_index(&self.parameter_ids, idx, "sweep parameter")
    }

    fn resolve_pulse_index(&self, idx: u32) -> Result<NamedId> {
        Self::resolve_index(&self.pulse_ids, idx, "pulse")
    }

    fn resolve_handle_index(&self, idx: u32) -> Result<NamedId> {
        Self::resolve_index(&self.handle_ids, idx, "acquisition handle")
    }

    // === Pre-pass and top-level registration ===

    /// Pre-pass: register all index/text pairs from the Cap'n Proto message.
    ///
    /// This walks signals, sweep parameters, pulses, and acquisition handles to
    /// register every entity's uid/alias pair in the resolver before the main
    /// deserialization pass.
    fn prepass_register_uids(
        &mut self,
        experiment: &experiment_capnp::experiment::Reader<'_>,
    ) -> Result<()> {
        let signals = experiment.get_signals().map_err(Error::new)?;
        self.signal_ids = Vec::with_capacity(signals.len() as usize);
        for (i, signal) in signals.iter().enumerate() {
            let alias = text_to_str(signal.get_uid().map_err(Error::new)?)?;
            let named_id = self.register_required_alias(alias, "signal", i)?;
            self.signal_ids.push(named_id);
        }

        let params = experiment.get_sweep_parameters().map_err(Error::new)?;
        self.parameter_ids = Vec::with_capacity(params.len() as usize);
        for (i, param) in params.iter().enumerate() {
            let alias = text_to_str(param.get_uid().map_err(Error::new)?)?;
            let named_id = self.register_required_alias(alias, "sweep parameter", i)?;
            self.parameter_ids.push(named_id);
        }

        let pulses = experiment.get_pulses().map_err(Error::new)?;
        self.pulse_ids = Vec::with_capacity(pulses.len() as usize);
        for (i, pulse) in pulses.iter().enumerate() {
            let alias = text_to_str(pulse.get_uid().map_err(Error::new)?)?;
            let named_id = self.register_required_alias(alias, "pulse", i)?;
            self.pulse_ids.push(named_id);
        }

        let handles = experiment.get_acquisition_handles().map_err(Error::new)?;
        self.handle_ids = Vec::with_capacity(handles.len() as usize);
        for (i, handle) in handles.iter().enumerate() {
            let name = text_to_str(handle.get_uid().map_err(Error::new)?)?;
            let named_id = self.register_required_alias(name, "acquisition handle", i)?;
            self.handle_ids.push(named_id);
        }

        Ok(())
    }

    /// Register experiment signals by table index.
    fn register_signals(&mut self) {
        for &named_id in &self.signal_ids {
            self.available_signals.insert(SignalUid(named_id));
        }
    }

    /// Deserialize sweep parameters.
    fn deserialize_sweep_params(
        &mut self,
        experiment: &experiment_capnp::experiment::Reader<'_>,
    ) -> Result<()> {
        let sweep_params = experiment.get_sweep_parameters().map_err(Error::new)?;
        for (idx, param) in sweep_params.iter().enumerate() {
            let named_id = self.resolve_parameter_index(idx as u32)?;
            let uid = ParameterUid(named_id);
            let sweep_param = deserialize_sweep_parameter(uid, &param)?;
            self.parameters.insert(uid, sweep_param);
        }
        Ok(())
    }

    /// Deserialize pulse definitions.
    fn deserialize_pulses(
        &mut self,
        experiment: &experiment_capnp::experiment::Reader<'_>,
    ) -> Result<()> {
        let pulse_list = experiment.get_pulses().map_err(Error::new)?;
        for (idx, pulse) in pulse_list.iter().enumerate() {
            let pulse_def = self.deserialize_pulse(&pulse, idx as u32)?;
            self.pulses.insert(pulse_def.uid, pulse_def);
        }
        Ok(())
    }

    fn deserialize_pulse(
        &mut self,
        reader: &pulse_capnp::pulse::Reader<'_>,
        capnp_idx: u32,
    ) -> Result<PulseDef> {
        let named_id = self.resolve_pulse_index(capnp_idx)?;
        let uid = PulseUid(named_id);

        let can_compress = reader.get_can_compress();

        // Amplitude
        let amp_reader = reader.get_amplitude().map_err(Error::new)?;
        let amp_re = amp_reader.get_real();
        let amp_im = amp_reader.get_imag();
        let amplitude = if amp_im == 0.0 {
            NumericLiteral::Float(amp_re)
        } else {
            NumericLiteral::Complex(Complex64::new(amp_re, amp_im))
        };

        // Length
        use pulse_capnp::pulse::length::Which as LenWhich;
        let length = match reader.get_length().which().map_err(Error::new)? {
            LenWhich::None(()) => None,
            LenWhich::Value(v) => Some(v),
        };

        // Shape
        use pulse_capnp::pulse::Which;
        let kind = match reader.which().map_err(Error::new)? {
            Which::Functional(functional) => {
                let functional = functional.map_err(Error::new)?;
                let uri = text_to_str(functional.get_sampler_uri().map_err(Error::new)?)?;
                let function_name = uri
                    .strip_prefix("py://")
                    .ok_or_else(|| Error::new(format!("Unsupported sampler URI: {uri}")))?;
                let function = match function_name {
                    "const" => PulseFunction::Constant,
                    other => PulseFunction::Custom {
                        function: other.to_owned(),
                    },
                };
                let pulse_length =
                    length.ok_or_else(|| Error::new("Functional pulse must have a length"))?;
                // Read definition-level pulse parameters (FunctionalPulse.parameters).
                let params_reader = functional.get_parameters().map_err(Error::new)?;
                if !params_reader.is_empty() {
                    let params = self.deserialize_pulse_parameters(&params_reader)?;
                    if !params.is_empty() {
                        self.pulse_definition_params.insert(uid, params);
                    }
                }
                PulseKind::Functional(FunctionalPulse {
                    length: seconds(pulse_length),
                    function,
                })
            }
            Which::Sampled(sampled) => {
                let sampled = sampled.map_err(Error::new)?;
                let is_complex = sampled.get_sample_type().map_err(Error::new)?
                    == pulse_capnp::SampleType::Complex;

                let samples = sampled.get_samples().map_err(Error::new)?;
                use pulse_capnp::waveform_data::Which as WfWhich;
                match samples.which().map_err(Error::new)? {
                    WfWhich::Inline(inline) => {
                        let inline = inline.map_err(Error::new)?;
                        let data = inline.get_data().map_err(Error::new)?;

                        let samples = if is_complex {
                            let complex: Vec<Complex64> = data
                                .chunks_exact(16)
                                .map(|chunk| {
                                    let re = f64::from_le_bytes(chunk[..8].try_into().unwrap());
                                    let im = f64::from_le_bytes(chunk[8..].try_into().unwrap());
                                    Complex64::new(re, im)
                                })
                                .collect();
                            NumericArray::Complex64(complex)
                        } else {
                            let floats: Vec<f64> = data
                                .chunks_exact(8)
                                .map(|chunk| f64::from_le_bytes(chunk.try_into().unwrap()))
                                .collect();
                            NumericArray::Float64(floats)
                        };
                        PulseKind::Sampled(SampledPulse { samples })
                    }
                    WfWhich::None(()) => {
                        return Err(Error::new("Sampled pulse has no waveform data"));
                    }
                }
            }
            Which::None(()) => {
                let pulse_length =
                    length.ok_or_else(|| Error::new("Pulse with no shape must have a length"))?;
                PulseKind::LengthOnly {
                    length: seconds(pulse_length),
                }
            }
        };

        Ok(PulseDef {
            uid,
            kind,
            can_compress,
            amplitude,
        })
    }

    // === Section tree deserialization ===

    /// Deserialize the root section tree into an `ExperimentNode`.
    fn deserialize_root_sections(
        &mut self,
        experiment: &experiment_capnp::experiment::Reader<'_>,
    ) -> Result<ExperimentNode> {
        let root_section = experiment.get_root_section().map_err(Error::new)?;
        let root_child_index = self.index_child_sections(&root_section)?;
        let content_items = root_section.get_content_items().map_err(Error::new)?;
        let mut root = ExperimentNode::new(Operation::Root);
        for (item_idx, item) in content_items.iter().enumerate() {
            use section_capnp::section_item::Which;
            match item.which().map_err(Error::new)? {
                Which::Section(section) => {
                    let section = section.map_err(Error::new)?;
                    let node = self.deserialize_section(
                        &section,
                        root_child_index.section_uid_for_index(item_idx)?,
                        &root_child_index.sibling_refs,
                    )?;
                    root.children.push(node.into());
                }
                Which::Operation(_) => {
                    return Err(Error::new(
                        "Unexpected operation at root level of experiment",
                    ));
                }
            }
        }
        Ok(root)
    }

    fn section_uid_from_name(&mut self, section_name: &str) -> SectionUid {
        if !section_name.is_empty() {
            SectionUid(self.id_store.get_or_insert(section_name))
        } else {
            self.section_uid_gen.next_uid(&mut self.id_store)
        }
    }

    fn index_child_sections(
        &mut self,
        parent: &section_capnp::section::Reader<'_>,
    ) -> Result<ChildSectionIndex> {
        let content_items = parent.get_content_items().map_err(Error::new)?;
        let mut section_uids = vec![None; content_items.len() as usize];
        let mut sibling_refs = SiblingSectionRefs::default();
        for (idx, item) in content_items.iter().enumerate() {
            use section_capnp::section_item::Which;
            if let Which::Section(section) = item.which().map_err(Error::new)? {
                let section = section.map_err(Error::new)?;
                let section_name = text_to_str(section.get_name().map_err(Error::new)?)?;
                let section_uid = self.section_uid_from_name(section_name);
                section_uids[idx] = Some(section_uid);
                if section_name.is_empty() {
                    continue;
                }
                // Sections with the same name (and thus the same SectionUid) may appear
                // multiple times within one parent — this is section reuse, e.g. via
                // Experiment.add(), and is valid.
                sibling_refs
                    .by_name
                    .insert(section_name.to_owned(), section_uid);
            }
        }
        Ok(ChildSectionIndex {
            section_uids,
            sibling_refs,
        })
    }

    fn deserialize_section(
        &mut self,
        reader: &section_capnp::section::Reader<'_>,
        section_uid: SectionUid,
        sibling_refs: &SiblingSectionRefs,
    ) -> Result<ExperimentNode> {
        let name = text_to_str(reader.get_name().map_err(Error::new)?)?;

        use section_capnp::section::Which;
        let operation = match reader.which().map_err(Error::new)? {
            Which::Unspecified(()) => {
                return Err(Error::new(format!(
                    "Section '{}' has no section kind set",
                    name,
                )));
            }
            Which::Regular(regular) => {
                let regular = regular.map_err(Error::new)?;
                Operation::Section(self.deserialize_regular_section(
                    &regular,
                    section_uid,
                    name,
                    sibling_refs,
                )?)
            }
            Which::AcquireLoop(acq_loop) => {
                let acq_loop = acq_loop.map_err(Error::new)?;
                Operation::AveragingLoop(deserialize_acquire_loop(&acq_loop, section_uid)?)
            }
            Which::Sweep(sweep) => {
                let sweep = sweep.map_err(Error::new)?;
                Operation::Sweep(self.deserialize_sweep_section(&sweep, section_uid)?)
            }
            Which::Match(match_section) => {
                let match_section = match_section.map_err(Error::new)?;
                Operation::Match(self.deserialize_match_section(
                    &match_section,
                    section_uid,
                    name,
                    sibling_refs,
                )?)
            }
            Which::CaseSection(case) => {
                let case = case.map_err(Error::new)?;
                Operation::Case(deserialize_case_section(&case, section_uid)?)
            }
            Which::PrngSetup(setup) => {
                let setup = setup.map_err(Error::new)?;
                Operation::PrngSetup(self.deserialize_prng_setup(&setup, section_uid)?)
            }
            Which::PrngLoop(prng_loop) => {
                let prng_loop = prng_loop.map_err(Error::new)?;
                Operation::PrngLoop(self.deserialize_prng_loop(&prng_loop, section_uid)?)
            }
        };

        let child_index = self.index_child_sections(reader)?;

        // Deserialize children.
        let content_items = reader.get_content_items().map_err(Error::new)?;
        let mut children = Vec::with_capacity(content_items.len() as usize);
        for (item_idx, item) in content_items.iter().enumerate() {
            use section_capnp::section_item::Which as ItemWhich;
            match item.which().map_err(Error::new)? {
                ItemWhich::Section(child_section) => {
                    let child_section = child_section.map_err(Error::new)?;
                    let child = self.deserialize_section(
                        &child_section,
                        child_index.section_uid_for_index(item_idx)?,
                        &child_index.sibling_refs,
                    )?;
                    children.push(child.into());
                }
                ItemWhich::Operation(op) => {
                    let op = op.map_err(Error::new)?;
                    let child = self.deserialize_operation(&op, name)?;
                    children.push(child.into());
                }
            }
        }

        let mut node = ExperimentNode::new(operation);
        node.children = children;
        Ok(node)
    }

    // === Section-kind specific deserialization ===

    fn deserialize_regular_section(
        &self,
        reader: &section_capnp::regular_section::Reader<'_>,
        section_uid: SectionUid,
        section_name: &str,
        sibling_refs: &SiblingSectionRefs,
    ) -> Result<Section> {
        let alignment = deserialize_alignment(reader.get_alignment().map_err(Error::new)?);

        use section_capnp::regular_section::length::Which as LenWhich;
        let length = match reader.get_length().which().map_err(Error::new)? {
            LenWhich::None(()) => None,
            LenWhich::Value(v) => Some(seconds(v)),
        };

        let triggers = self.deserialize_triggers(reader)?;

        Ok(Section {
            uid: section_uid,
            alignment,
            length,
            play_after: deserialize_play_after_from_regular(reader, section_name, sibling_refs)?,
            triggers,
            on_system_grid: reader.get_on_system_grid(),
        })
    }

    fn deserialize_sweep_section(
        &self,
        reader: &section_capnp::sweep_section::Reader<'_>,
        section_uid: SectionUid,
    ) -> Result<Sweep> {
        let alignment = deserialize_alignment(reader.get_alignment().map_err(Error::new)?);
        let param_uids = reader.get_parameters().map_err(Error::new)?;
        let mut parameters = Vec::with_capacity(param_uids.len() as usize);
        for idx in param_uids.iter() {
            parameters.push(ParameterUid(self.resolve_parameter_index(idx)?));
        }

        if parameters.is_empty() {
            return Err(Error::new("Sweep must have at least one parameter"));
        }

        // Derive the sweep count from the first parameter's length.
        let count = parameters
            .first()
            .and_then(|uid| self.parameters.get(uid))
            .map(|p| p.len() as u32)
            .and_then(NonZeroU32::new)
            .ok_or_else(|| Error::new("Sweep parameter must have at least one value"))?;

        use section_capnp::sweep_section::chunking::Which as ChunkWhich;
        let chunking = match reader.get_chunking().which().map_err(Error::new)? {
            ChunkWhich::None(()) => None,
            ChunkWhich::Count(v) => NonZeroU32::new(v).map(|count| Chunking::Count { count }),
            ChunkWhich::Auto(()) => Some(Chunking::Auto),
        };

        Ok(Sweep {
            uid: section_uid,
            parameters,
            count,
            alignment,
            reset_oscillator_phase: reader.get_reset_oscillator_phase(),
            chunking,
        })
    }

    fn deserialize_match_section(
        &mut self,
        reader: &section_capnp::match_section::Reader<'_>,
        section_uid: SectionUid,
        section_name: &str,
        sibling_refs: &SiblingSectionRefs,
    ) -> Result<Match> {
        use section_capnp::match_section::Which;
        let target = match reader.which().map_err(Error::new)? {
            Which::Handle(idx) => {
                // Resolve via pre-declared AcquisitionHandle entries.
                let handle_named_id = self.resolve_handle_index(idx)?;
                MatchTarget::Handle(HandleUid(handle_named_id))
            }
            Which::UserRegister(reg) => MatchTarget::UserRegister(reg),
            Which::SweepParameter(idx) => {
                MatchTarget::SweepParameter(ParameterUid(self.resolve_parameter_index(idx)?))
            }
            Which::PrngSample(text_result) => {
                let alias = text_to_str(text_result.map_err(Error::new)?)?;
                let named_id = self.id_store.get_or_insert(alias);
                MatchTarget::PrngSample(SectionUid(named_id))
            }
            Which::None(()) => {
                return Err(Error::new("Match section has no target"));
            }
        };

        use section_capnp::match_section::local::Which as LocalWhich;
        let local = match reader.get_local().which().map_err(Error::new)? {
            LocalWhich::None(()) => None,
            LocalWhich::Value(v) => Some(v),
        };

        Ok(Match {
            uid: section_uid,
            target,
            local,
            play_after: deserialize_play_after_from_match(reader, section_name, sibling_refs)?,
        })
    }

    fn deserialize_prng_setup(
        &self,
        reader: &section_capnp::prng_setup_section::Reader<'_>,
        section_uid: SectionUid,
    ) -> Result<PrngSetup> {
        Ok(PrngSetup {
            uid: section_uid,
            range: reader.get_range(),
            seed: reader.get_seed(),
        })
    }

    fn deserialize_prng_loop(
        &mut self,
        reader: &section_capnp::prng_loop_section::Reader<'_>,
        section_uid: SectionUid,
    ) -> Result<PrngLoop> {
        let sample_alias = text_to_str(reader.get_prng_sample().map_err(Error::new)?)?;
        let sample_named_id = self.id_store.get_or_insert(sample_alias);
        let count = reader.get_count();
        Ok(PrngLoop {
            uid: section_uid,
            count: NonZeroU32::new(count)
                .ok_or_else(|| Error::new("PrngLoop count must be > 0"))?,
            sample_uid: PrngSampleUid(sample_named_id),
        })
    }

    fn deserialize_triggers(
        &self,
        reader: &section_capnp::regular_section::Reader<'_>,
    ) -> Result<Vec<Trigger>> {
        let triggers = reader.get_triggers().map_err(Error::new)?;
        let mut result = Vec::with_capacity(triggers.len() as usize);
        for trigger in triggers.iter() {
            let signal_uid = SignalUid(self.resolve_signal_index(trigger.get_signal())?);
            result.push(Trigger {
                signal: signal_uid,
                state: trigger.get_state() as u8,
            });
        }
        Ok(result)
    }

    // === Operation deserialization ===

    fn deserialize_operation(
        &mut self,
        reader: &operation_capnp::operation::Reader<'_>,
        parent_section_name: &str,
    ) -> Result<ExperimentNode> {
        use operation_capnp::operation::Which;
        let operation = match reader.which().map_err(Error::new)? {
            Which::Play(play) => {
                let play = play.map_err(Error::new)?;
                Operation::PlayPulse(self.deserialize_play_op(&play)?)
            }
            Which::Delay(delay) => {
                let delay = delay.map_err(Error::new)?;
                Operation::Delay(self.deserialize_delay_op(&delay)?)
            }
            Which::Reserve(reserve) => {
                let reserve = reserve.map_err(Error::new)?;
                Operation::Reserve(self.deserialize_reserve_op(&reserve)?)
            }
            Which::Acquire(acquire) => {
                let acquire = acquire.map_err(Error::new)?;
                Operation::Acquire(self.deserialize_acquire_op(&acquire)?)
            }
            Which::Call(_) => Operation::NearTimeCallback,
            Which::SetNode(_) => Operation::SetNode,
            Which::ResetOscillatorPhase(reset) => {
                let reset = reset.map_err(Error::new)?;
                Operation::ResetOscillatorPhase(self.deserialize_reset_oscillator_phase(&reset)?)
            }
            Which::None(()) => {
                let label = section_label_for_errors(parent_section_name);
                return Err(Error::new(format!("Operation in {label} has no kind set")));
            }
        };

        Ok(ExperimentNode::new(operation))
    }

    fn resolve_signal(&self, capnp_idx: u32) -> Result<SignalUid> {
        let named_id = self.resolve_signal_index(capnp_idx)?;
        Ok(SignalUid(named_id))
    }

    fn deserialize_play_op(
        &mut self,
        reader: &operation_capnp::play_op::Reader<'_>,
    ) -> Result<PlayPulse> {
        let signal = self.resolve_signal(reader.get_signal())?;

        let pulse_uid = reader.get_pulse();
        let pulse = if pulse_uid != common_capnp::NONE_ID {
            Some(PulseUid(self.resolve_pulse_index(pulse_uid)?))
        } else {
            None
        };

        let amplitude = self
            .deserialize_value_complex_or_float(&reader.get_amplitude().map_err(Error::new)?)?
            .unwrap_or(ValueOrParameter::Value(ComplexOrFloat::Float(1.0)));

        let phase = self.deserialize_value_f64(&reader.get_phase().map_err(Error::new)?)?;

        let increment_oscillator_phase = self.deserialize_value_f64(
            &reader
                .get_increment_oscillator_phase()
                .map_err(Error::new)?,
        )?;

        let set_oscillator_phase =
            self.deserialize_value_f64(&reader.get_set_oscillator_phase().map_err(Error::new)?)?;

        let length = self.deserialize_value_duration(&reader.get_length().map_err(Error::new)?)?;

        let parameters =
            self.deserialize_pulse_parameters(&reader.get_pulse_parameters().map_err(Error::new)?)?;

        let markers = self.deserialize_markers(&reader.get_markers().map_err(Error::new)?)?;

        let pulse_parameters = pulse
            .as_ref()
            .and_then(|uid| self.pulse_definition_params.get(uid))
            .cloned()
            .unwrap_or_default();

        Ok(PlayPulse {
            signal,
            pulse,
            amplitude,
            phase,
            increment_oscillator_phase,
            set_oscillator_phase,
            length,
            parameters,
            pulse_parameters,
            markers,
        })
    }

    fn deserialize_markers(
        &self,
        reader: &operation_capnp::markers::Reader<'_>,
    ) -> Result<Vec<Marker>> {
        let mut markers = Vec::new();

        let m1 = reader.get_marker1().map_err(Error::new)?;
        if marker_spec_is_set(&m1)? {
            markers.push(self.deserialize_marker_spec(&m1, MarkerSelector::M1)?);
        }

        let m2 = reader.get_marker2().map_err(Error::new)?;
        if marker_spec_is_set(&m2)? {
            markers.push(self.deserialize_marker_spec(&m2, MarkerSelector::M2)?);
        }

        Ok(markers)
    }

    fn deserialize_marker_spec(
        &self,
        reader: &operation_capnp::marker_spec::Reader<'_>,
        selector: MarkerSelector,
    ) -> Result<Marker> {
        use operation_capnp::marker_spec::length::Which as LenWhich;
        use operation_capnp::marker_spec::start::Which as StartWhich;

        let start = match reader.get_start().which().map_err(Error::new)? {
            StartWhich::None(()) => None,
            StartWhich::Value(v) => Some(seconds(v)),
        };

        let length = match reader.get_length().which().map_err(Error::new)? {
            LenWhich::None(()) => None,
            LenWhich::Value(v) => Some(seconds(v)),
        };

        let waveform = reader.get_waveform();
        let pulse_id = if waveform == common_capnp::NONE_ID {
            None
        } else {
            Some(PulseUid(self.resolve_pulse_index(waveform)?))
        };

        Ok(Marker {
            marker_selector: selector,
            enable: reader.get_enable(),
            start,
            length,
            pulse_id,
        })
    }

    fn deserialize_delay_op(
        &self,
        reader: &operation_capnp::delay_op::Reader<'_>,
    ) -> Result<Delay> {
        let signal = self.resolve_signal(reader.get_signal())?;
        let time = self
            .deserialize_value_duration(&reader.get_time().map_err(Error::new)?)?
            .ok_or_else(|| Error::new("Delay time must be specified"))?;

        Ok(Delay {
            signal,
            time,
            precompensation_clear: reader.get_precompensation_clear(),
        })
    }

    fn deserialize_reserve_op(
        &self,
        reader: &operation_capnp::reserve_op::Reader<'_>,
    ) -> Result<Reserve> {
        Ok(Reserve {
            signal: self.resolve_signal(reader.get_signal())?,
        })
    }

    fn deserialize_acquire_op(
        &mut self,
        reader: &operation_capnp::acquire_op::Reader<'_>,
    ) -> Result<Acquire> {
        let signal = self.resolve_signal(reader.get_signal())?;

        // Handle: resolve via pre-declared AcquisitionHandle entries.
        let handle = HandleUid(self.resolve_handle_index(reader.get_handle())?);

        // Kernels
        let kernel_uids = reader.get_kernels().map_err(Error::new)?;
        let mut kernel = Vec::with_capacity(kernel_uids.len() as usize);
        for idx in kernel_uids.iter() {
            kernel.push(PulseUid(self.resolve_pulse_index(idx)?));
        }

        // Length
        let length_reader = reader.get_length().map_err(Error::new)?;
        let length = match length_reader.which().map_err(Error::new)? {
            common_capnp::value::Which::None(()) => None,
            common_capnp::value::Which::Constant(constant) => {
                let constant = constant.map_err(Error::new)?;
                let numeric = deserialize_constant(&constant)?;
                let f: f64 = numeric.try_into().map_err(Error::new)?;
                Some(seconds(f))
            }
            common_capnp::value::Which::ParameterRef(_) => {
                return Err(Error::new("Acquire length cannot be a parameter reference"));
            }
        };

        // Per-operation kernel parameter overrides → Acquire.parameters.
        // Only pad to kernel length when no kernel parameters were provided (matching
        // py_conversion: None → pad, provided → use as-is).
        let kernel_params = reader.get_kernel_parameters().map_err(Error::new)?;
        let mut parameters = Vec::with_capacity(kernel_params.len() as usize);
        for param_map in kernel_params.iter() {
            let entries = param_map.get_parameters().map_err(Error::new)?;
            let params = self.deserialize_pulse_parameters(&entries)?;
            parameters.push(params);
        }
        if kernel_params.is_empty() {
            parameters.resize_with(kernel.len(), HashMap::new);
        }

        // Apply definition-level kernel parameters from functional pulse definitions.
        let pulse_parameters = kernel
            .iter()
            .map(|uid| {
                self.pulse_definition_params
                    .get(uid)
                    .cloned()
                    .unwrap_or_default()
            })
            .collect();

        Ok(Acquire {
            signal,
            handle,
            length,
            kernel,
            parameters,
            pulse_parameters,
        })
    }

    fn deserialize_reset_oscillator_phase(
        &self,
        reader: &operation_capnp::reset_oscillator_phase_op::Reader<'_>,
    ) -> Result<ResetOscillatorPhase> {
        let signal_uid = reader.get_signal();
        let signals = if signal_uid == common_capnp::NONE_ID {
            vec![]
        } else {
            vec![self.resolve_signal(signal_uid)?]
        };

        Ok(ResetOscillatorPhase { signals })
    }

    // === Value deserialization helpers ===

    fn deserialize_value_f64(
        &self,
        reader: &common_capnp::value::Reader<'_>,
    ) -> Result<Option<ValueOrParameter<f64>>> {
        use common_capnp::value::Which;
        match reader.which().map_err(Error::new)? {
            Which::None(()) => Ok(None),
            Which::Constant(constant) => {
                let constant = constant.map_err(Error::new)?;
                let f: f64 = deserialize_constant(&constant)?
                    .try_into()
                    .map_err(Error::new)?;
                Ok(Some(ValueOrParameter::Value(f)))
            }
            Which::ParameterRef(idx) => Ok(Some(ValueOrParameter::Parameter(ParameterUid(
                self.resolve_parameter_index(idx)?,
            )))),
        }
    }

    fn deserialize_value_duration(
        &self,
        reader: &common_capnp::value::Reader<'_>,
    ) -> Result<
        Option<
            ValueOrParameter<laboneq_units::duration::Duration<laboneq_units::duration::Second>>,
        >,
    > {
        use common_capnp::value::Which;
        match reader.which().map_err(Error::new)? {
            Which::None(()) => Ok(None),
            Which::Constant(constant) => {
                let constant = constant.map_err(Error::new)?;
                let f: f64 = deserialize_constant(&constant)?
                    .try_into()
                    .map_err(Error::new)?;
                Ok(Some(ValueOrParameter::Value(seconds(f))))
            }
            Which::ParameterRef(idx) => Ok(Some(ValueOrParameter::Parameter(ParameterUid(
                self.resolve_parameter_index(idx)?,
            )))),
        }
    }

    fn deserialize_value_complex_or_float(
        &self,
        reader: &common_capnp::value::Reader<'_>,
    ) -> Result<Option<ValueOrParameter<ComplexOrFloat>>> {
        use common_capnp::value::Which;
        match reader.which().map_err(Error::new)? {
            Which::None(()) => Ok(None),
            Which::Constant(constant) => {
                let constant = constant.map_err(Error::new)?;
                let cf: ComplexOrFloat = deserialize_constant(&constant)?
                    .try_into()
                    .map_err(Error::new)?;
                Ok(Some(ValueOrParameter::Value(cf)))
            }
            Which::ParameterRef(idx) => Ok(Some(ValueOrParameter::Parameter(ParameterUid(
                self.resolve_parameter_index(idx)?,
            )))),
        }
    }

    // === Pulse parameter deserialization ===

    fn deserialize_pulse_parameters(
        &mut self,
        reader: &capnp::struct_list::Reader<'_, common_capnp::value_entry::Owned>,
    ) -> Result<HashMap<PulseParameterUid, PulseParameterValue>> {
        let mut result = HashMap::new();
        for entry in reader.iter() {
            let key = text_to_str(entry.get_key().map_err(Error::new)?)?;
            let key_id = PulseParameterUid(self.id_store.get_or_insert(key));
            let value_reader = entry.get_value().map_err(Error::new)?;
            let value = self.deserialize_pulse_parameter_value(&value_reader)?;
            result.insert(key_id, value);
        }
        Ok(result)
    }

    fn deserialize_pulse_parameter_value(
        &mut self,
        reader: &common_capnp::value::Reader<'_>,
    ) -> Result<PulseParameterValue> {
        use common_capnp::value::Which;
        match reader.which().map_err(Error::new)? {
            Which::None(()) => Ok(PulseParameterValue::ValueOrParameter(
                ValueOrParameter::Value(NumericLiteral::Float(0.0)),
            )),
            Which::Constant(constant) => {
                let constant = constant.map_err(Error::new)?;
                use common_capnp::constant::Which as ConstantWhich;
                match constant.which().map_err(Error::new)? {
                    ConstantWhich::StringValue(value) => {
                        let value = text_to_str(value.map_err(Error::new)?)?;
                        let py_value: Bound<'_, PyAny> = PyString::new(self.py, value).into_any();
                        let external_uid = intern_py_object_uid(&py_value)?;
                        self.external_parameter_values
                            .entry(external_uid)
                            .or_insert_with(|| py_value.clone().unbind());
                        Ok(PulseParameterValue::ExternalParameter(external_uid))
                    }
                    // Pickling is strictly a fallback for arbitrary Python objects passed
                    // into custom functional pulse parameters (e.g., a SciPy interpolation
                    // object). Because the Rust compiler does not execute these (they are
                    // evaluated in Python during waveform sampling), they must be passed
                    // opaquely.
                    ConstantWhich::PickledValue(data) => {
                        let data = data.map_err(Error::new)?;
                        // Hash the raw pickled bytes directly — they came from
                        // `pickle.dumps` in the serializer, so re-pickling would
                        // be redundant.
                        let external_uid = uid_from_bytes(self.py, data)?;
                        let py_bytes = PyBytes::new(self.py, data);
                        let loads = self
                            .py
                            .import(pyo3::intern!(self.py, "pickle"))?
                            .getattr(pyo3::intern!(self.py, "loads"))?;
                        let py_value = loads.call1((py_bytes,))?;
                        self.external_parameter_values
                            .entry(external_uid)
                            .or_insert_with(|| py_value.clone().unbind());
                        Ok(PulseParameterValue::ExternalParameter(external_uid))
                    }
                    ConstantWhich::RawBytesValue(data) => {
                        let data = data.map_err(Error::new)?;
                        let py_value = PyBytes::new(self.py, data).into_any();
                        // Pickle-then-hash to match the legacy path's UID derivation.
                        let external_uid = intern_py_object_uid(&py_value)?;
                        self.external_parameter_values
                            .entry(external_uid)
                            .or_insert_with(|| py_value.clone().unbind());
                        Ok(PulseParameterValue::ExternalParameter(external_uid))
                    }
                    _ => Ok(PulseParameterValue::ValueOrParameter(
                        ValueOrParameter::Value(deserialize_constant(&constant)?),
                    )),
                }
            }
            Which::ParameterRef(idx) => Ok(PulseParameterValue::ValueOrParameter(
                ValueOrParameter::Parameter(ParameterUid(self.resolve_parameter_index(idx)?)),
            )),
        }
    }
}

// === Entry point ===

/// Deserializes a Cap'n Proto experiment message into Rust DSL types.
///
/// `py` is required because `SampledPulse` stores samples as a Python numpy array.
pub(crate) fn deserialize_experiment(
    py: Python<'_>,
    bytes: &[u8],
    packed: bool,
) -> Result<DeserializedExperiment> {
    let mut reader_options = capnp::message::ReaderOptions::new();
    // Raise the traversal limit well above the default (~64 MB). Experiments can
    // scale to 1,000,000+ sections, easily exceeding the default. The payload is
    // produced internally, so there is no DOS risk.
    reader_options.traversal_limit_in_words(Some(1024 * 1024 * 1024)); // ~8 GB
    if packed {
        let reader = capnp::serialize_packed::read_message(&mut &*bytes, reader_options)
            .map_err(|e| Error::new(format!("Failed to read packed Cap'n Proto message: {e}")))?;
        let experiment = reader
            .get_root::<experiment_capnp::experiment::Reader<'_>>()
            .map_err(Error::new)?;
        Deserializer::new(py).deserialize(&experiment)
    } else {
        let reader = capnp::serialize::read_message_from_flat_slice(&mut &*bytes, reader_options)
            .map_err(|e| Error::new(format!("Failed to read Cap'n Proto message: {e}")))?;
        let experiment = reader
            .get_root::<experiment_capnp::experiment::Reader<'_>>()
            .map_err(Error::new)?;
        Deserializer::new(py).deserialize(&experiment)
    }
}

// === Post-processing ===

/// Post-pass: add derived sweep parameters to each sweep section.
///
/// Mirrors `collect_derived_parameters` in `py_conversion.rs`. After signal
/// processing in the capnp path, `builder.driving_parameters` is populated
/// (by `py_signal_to_signal` / `extract_parameter`). This pass walks the
/// deserialized tree and adds any missing derived parameters to each sweep's
/// parameter list.
pub(crate) fn fix_derived_parameters(
    node: &mut ExperimentNode,
    driving_parameters: &HashMap<ParameterUid, HashSet<ParameterUid>>,
) {
    if let Operation::Sweep(ref mut sweep) = node.kind {
        let mut to_add = Vec::new();
        for &param_uid in &sweep.parameters {
            collect_derived_recursive(param_uid, driving_parameters, &mut to_add);
        }
        for d in to_add {
            if !sweep.parameters.contains(&d) {
                sweep.parameters.push(d);
            }
        }
    }
    for child in &mut node.children {
        let child = Arc::make_mut(child);
        fix_derived_parameters(child, driving_parameters);
    }
}

fn collect_derived_recursive(
    uid: ParameterUid,
    driving_parameters: &HashMap<ParameterUid, HashSet<ParameterUid>>,
    collected: &mut Vec<ParameterUid>,
) {
    if let Some(derived_set) = driving_parameters.get(&uid) {
        for &d in derived_set {
            if !collected.contains(&d) {
                collected.push(d);
                collect_derived_recursive(d, driving_parameters, collected);
            }
        }
    }
}

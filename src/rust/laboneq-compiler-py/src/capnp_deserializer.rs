// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

//! Deserializes Cap'n Proto binary format to Rust DSL experiment types.
//!
//! Reads Cap'n Proto `Experiment` messages and reconstructs the internal
//! `ExperimentNode`, `NamedIdStore`, sweep parameters, and pulse definitions
//! used by the compiler.
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

use crate::capnp_setup_description::SetupDescriptionPayload;
use crate::error::{Error, Result};

use laboneq_capnp::pulse::v1::{
    calibration_capnp, common_capnp, coprocessor_capnp, experiment_capnp, operation_capnp,
    pulse_capnp, section_capnp, sweep_capnp,
};
use laboneq_common::named_id::{NamedId, NamedIdStore};
use laboneq_common::types::Literal;
use laboneq_dsl::ExperimentNode;
use laboneq_dsl::experiment_signal::ExperimentSignal;
use laboneq_dsl::hqcs::{
    CmpOp, CoprocessorDecl, CoprocessorMappingDecl, FieldBinding, HqcsDeclarations, IsLiveTarget,
    Predicate, StreamDecl, StreamEndpoint, StreamFieldDecl, VarType, VariableDecl, VariableValue,
};
use laboneq_dsl::operation::{
    Acquire, AveragingLoop, Case, Delay, DoUntil, Ensure, ExternalOrValue, MarkStale,
    MarkStaleTarget, Match, NearTimeCallback, Operation, PlayPulse, PrngLoop, PrngSetup, Reserve,
    ResetOscillatorPhase, Section, Send, SendArg, SetNode, Sweep, ValueEntry,
};
use laboneq_dsl::signal_calibration::{
    BounceCompensation, CorrectionMatrix, ExponentialCompensation, FirCompensation,
    HighPassCompensation, MixerCalibration, OutputRoute, PortMode, Precompensation,
    SignalCalibration,
};
use laboneq_dsl::types::{
    AcquisitionType, AmplifierPump, AveragingMode, BuiltinPulse, ComplexOrFloat,
    ExternalParameterUid, FunctionalPulse, HandleUid, Marker, MarkerSelector, MatchTarget,
    NumericLiteral, Oscillator, OscillatorKind, ParameterUid, PrngSampleUid, PulseDef,
    PulseFunction, PulseKind, PulseParameterUid, PulseUid, PumpCancellationSource, Quantity,
    RepetitionMode, SampledPulse, SectionAlignment, SectionTimingMode, SectionUid, SignalUid,
    StreamUid, SweepParameter, Trigger, Unit, ValueOrParameter, VariableUid,
};
use laboneq_py_utils::py_object_store::{PyObjectPayload, PyObjectStore};
use laboneq_units::angle::Angle64;
use laboneq_units::duration::{hertz, seconds};
use num_complex::Complex64;
use numeric_array::NumericArray;
use pyo3::prelude::*;
use rustc_hash::FxHashMap;

use tracing::instrument;

type ExternalParameterStore = PyObjectStore<ExternalParameterUid>;

/// Result of deserializing a Cap'n Proto experiment message.
pub struct DeserializedExperiment {
    pub root: ExperimentNode,
    pub id_store: NamedIdStore,
    pub parameters: Vec<SweepParameter>,
    pub pulses: FxHashMap<PulseUid, PulseDef>,
    pub external_parameters: ExternalParameterStore,

    pub experiment_signals: Vec<ExperimentSignal>,

    pub setup_description: SetupDescriptionPayload,

    pub hqcs: HqcsDeclarations,
    pub handles: Vec<HandleUid>,
}

// === Pure helper functions ===

/// Helper to convert capnp text reader to &str.
fn text_to_str(reader: capnp::text::Reader<'_>) -> Result<&str> {
    reader
        .to_str()
        .map_err(|e| Error::new(format!("Invalid UTF-8 in Cap'n Proto text: {e}")))
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

fn deserialize_section_timing_mode(
    section_timing_mode: section_capnp::SectionTimingMode,
) -> SectionTimingMode {
    match section_timing_mode {
        section_capnp::SectionTimingMode::Relaxed => SectionTimingMode::Relaxed,
        section_capnp::SectionTimingMode::Strict => SectionTimingMode::Strict,
        section_capnp::SectionTimingMode::Unspecified => SectionTimingMode::Relaxed,
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

fn deserialize_constant_numeric(
    reader: &common_capnp::constant::Reader<'_>,
) -> Result<NumericLiteral> {
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
        Which::Boolean(_) => Err(Error::new("Boolean values not supported as numeric")),
        Which::PythonValue(_) | Which::RawBytesValue(_) => {
            Err(Error::new("Bytes values not supported as numeric"))
        }
    }
}

fn deserialize_constant(reader: &common_capnp::constant::Reader<'_>) -> Result<Literal> {
    use common_capnp::constant::Which;
    match reader.which().map_err(Error::new)? {
        Which::Real(v) => Ok(v.into()),
        Which::Integer(v) => Ok(v.into()),
        Which::Complex(cv) => {
            let cv = cv.map_err(Error::new)?;
            Ok(Complex64::new(cv.get_real(), cv.get_imag()).into())
        }
        Which::StringValue(v) => Ok(text_to_str(v.map_err(Error::new)?)?.into()),
        Which::Boolean(_) => Err(Error::new("Boolean values not supported")),
        Which::PythonValue(_) | Which::RawBytesValue(_) => {
            Err(Error::new("Bytes values not supported"))
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
            Ok(Some(deserialize_constant_numeric(&constant)?))
        }
        Which::ParameterRef(_) => Err(Error::new(
            "Expected a constant value, got parameter reference",
        )),
    }
}

fn deserialize_acquisition_type(acq_type: operation_capnp::AcquisitionType) -> AcquisitionType {
    match acq_type {
        operation_capnp::AcquisitionType::Integration => AcquisitionType::Integration,
        operation_capnp::AcquisitionType::Raw => AcquisitionType::Raw,
        operation_capnp::AcquisitionType::Discrimination => AcquisitionType::Discrimination,
        operation_capnp::AcquisitionType::SpectroscopyIq => AcquisitionType::Spectroscopy,
        operation_capnp::AcquisitionType::SpectroscopyPsd => AcquisitionType::SpectroscopyPsd,
        operation_capnp::AcquisitionType::Unspecified => AcquisitionType::default(),
    }
}

/// Returns the [`PulseFunction`] associated to a pulse's sampler URI.
/// `py://name` names a Python functional, `builtin://name` a standard pulse.
fn pulse_function(sampler_uri: &str) -> Result<PulseFunction> {
    let function = match sampler_uri.split_once("://") {
        Some(("py", name)) => Some(PulseFunction::Custom {
            function: name.to_owned(),
        }),
        Some(("builtin", name)) => BuiltinPulse::from_name(name).map(PulseFunction::Builtin),
        _ => None,
    };
    function.ok_or_else(|| Error::new(format!("Unsupported sampler URI: {sampler_uri}")))
}

fn coproc_type_from_capnp(t: coprocessor_capnp::VarType) -> Result<VarType> {
    use coprocessor_capnp::VarType as CapnpVarType;
    Ok(match t {
        CapnpVarType::Unspecified => {
            return Err(Error::new("Coprocessor type is unspecified"));
        }
        CapnpVarType::Int8 => VarType::Int8,
        CapnpVarType::Int16 => VarType::Int16,
        CapnpVarType::Int32 => VarType::Int32,
        CapnpVarType::Int64 => VarType::Int64,
        CapnpVarType::Uint8 => VarType::Uint8,
        CapnpVarType::Uint16 => VarType::Uint16,
        CapnpVarType::Uint32 => VarType::Uint32,
        CapnpVarType::Uint64 => VarType::Uint64,
        CapnpVarType::Phase => VarType::Phase,
        CapnpVarType::Frequency => VarType::Frequency,
        CapnpVarType::Amplitude => VarType::Amplitude,
        CapnpVarType::DiscriminationDataPacked => VarType::DiscriminationDataPacked,
        CapnpVarType::IqDataPacked => VarType::IqDataPacked,
        CapnpVarType::ScopeShot => VarType::ScopeShot,
        CapnpVarType::WaveformUpdate => VarType::WaveformUpdate,
    })
}

fn coproc_value_from_capnp(
    reader: &coprocessor_capnp::variable_value::Reader<'_>,
) -> Result<VariableValue> {
    use coprocessor_capnp::variable_value::Which;
    Ok(match reader.which().map_err(Error::new)? {
        Which::IntValue(v) => VariableValue::Int(v),
        Which::FloatValue(v) => VariableValue::Float(v),
        Which::PhaseRadians(v) => VariableValue::Phase(Angle64::from_radians(v)),
        Which::FrequencyHz(v) => VariableValue::Frequency(hertz(v)),
        Which::Amplitude(v) => VariableValue::Amplitude(v),
    })
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
        section_capnp::AveragingMode::Cyclic => AveragingMode::Cyclic,
        section_capnp::AveragingMode::SingleShot => AveragingMode::SingleShot,
        section_capnp::AveragingMode::Unspecified => AveragingMode::default(),
    };

    let acquisition_type =
        deserialize_acquisition_type(reader.get_acquisition_type().map_err(Error::new)?);

    use section_capnp::acquire_loop_section::repetition::Which as RepWhich;
    let repetition_mode = match reader.get_repetition().which().map_err(Error::new)? {
        RepWhich::Fastest(()) | RepWhich::Unspecified(()) => RepetitionMode::Fastest,
        RepWhich::Constant(t) => RepetitionMode::Constant { time: seconds(t) },
        RepWhich::Auto(()) => RepetitionMode::Auto,
    };
    let section_timing_mode =
        deserialize_section_timing_mode(reader.get_section_timing_mode().map_err(Error::new)?);

    Ok(AveragingLoop {
        uid: section_uid,
        count,
        acquisition_type,
        averaging_mode,
        repetition_mode,
        reset_oscillator_phase: reader.get_reset_oscillator_phase(),
        alignment,
        section_timing_mode,
    })
}

fn deserialize_case_section(
    reader: &section_capnp::case_section::Reader<'_>,
    section_uid: SectionUid,
) -> Result<Case> {
    let state_reader = reader.get_state().map_err(Error::new)?;
    let state = deserialize_value_as_numeric(&state_reader)?
        .ok_or_else(|| Error::new("Case state must be a constant value"))?;
    let section_timing_mode =
        deserialize_section_timing_mode(reader.get_section_timing_mode().map_err(Error::new)?);

    Ok(Case {
        uid: section_uid,
        state,
        section_timing_mode,
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
struct Deserializer {
    id_store: NamedIdStore,

    /// --- Experiment entities ---
    signal_ids: Vec<NamedId>,
    parameter_ids: Vec<NamedId>,
    pulse_ids: Vec<NamedId>,
    handle_ids: Vec<NamedId>,
    section_uid_gen: SectionUidGenerator,
    external_parameters: ExternalParameterStore,
    parameters: Vec<SweepParameter>,
    pulses: FxHashMap<PulseUid, PulseDef>,
    pulse_definition_params: HashMap<PulseUid, HashMap<PulseParameterUid, ExternalOrValue>>,
    // A map of driver -> derived parameters.
    driving_parameters: HashMap<ParameterUid, Vec<ParameterUid>>,

    hqcs: HqcsDeclarations,
}

impl Deserializer {
    fn new() -> Self {
        Self {
            id_store: NamedIdStore::new(),
            signal_ids: Vec::new(),
            parameter_ids: Vec::new(),
            pulse_ids: Vec::new(),
            handle_ids: Vec::new(),
            section_uid_gen: SectionUidGenerator::new(),
            external_parameters: ExternalParameterStore::new(),
            parameters: Vec::new(),
            pulses: FxHashMap::default(),
            pulse_definition_params: HashMap::new(),
            driving_parameters: HashMap::new(),
            hqcs: HqcsDeclarations::default(),
        }
    }

    /// Main orchestrator — consumes `self` so fields can be moved into the
    /// result without cloning.
    fn deserialize(
        mut self,
        experiment: &experiment_capnp::experiment::Reader<'_>,
    ) -> Result<DeserializedExperiment> {
        validate_schema_version(experiment)?;
        self.prepass_register_uids(experiment)?;
        self.deserialize_sweep_params(experiment)?;
        self.deserialize_pulses(experiment)?;
        self.deserialize_hqcs_declarations(experiment)?;

        let setup_description = self.deserialize_device_setup(experiment)?;
        let experiment_signals = self.deserialize_experiment_signals(experiment)?;
        let root = self.deserialize_root_sections(experiment)?;

        Ok(DeserializedExperiment {
            root,
            id_store: self.id_store,
            parameters: self.parameters,
            pulses: self.pulses,
            external_parameters: self.external_parameters,
            experiment_signals,
            setup_description,
            hqcs: self.hqcs,
            handles: self.handle_ids.into_iter().map(Into::into).collect(),
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

    fn resolve_variable_index(&self, idx: u32) -> Result<VariableUid> {
        if (idx as usize) < self.hqcs.variables.len() {
            Ok(VariableUid(idx))
        } else {
            Err(Error::new(format!(
                "Unknown variable index reference: {idx}"
            )))
        }
    }

    fn resolve_stream_index(&mut self, idx: u32) -> Result<StreamUid> {
        let name = self
            .hqcs
            .streams
            .get(idx as usize)
            .map(|stream| stream.uid.clone())
            .ok_or_else(|| Error::new(format!("Unknown stream index reference: {idx}")))?;
        Ok(StreamUid(self.id_store.get_or_insert(&name)))
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

    /// Deserialize sweep parameters.
    fn deserialize_sweep_params(
        &mut self,
        experiment: &experiment_capnp::experiment::Reader<'_>,
    ) -> Result<()> {
        let sweep_params = experiment.get_sweep_parameters().map_err(Error::new)?;
        for (idx, param) in sweep_params.iter().enumerate() {
            let named_id = self.resolve_parameter_index(idx as u32)?;
            let sweep_param = self.deserialize_sweep_parameter(named_id.into(), &param)?;
            self.parameters.push(sweep_param);
        }
        Ok(())
    }

    fn deserialize_sweep_parameter(
        &mut self,
        uid: ParameterUid,
        reader: &sweep_capnp::sweep_parameter::Reader<'_>,
    ) -> Result<SweepParameter> {
        use sweep_capnp::sweep_parameter::Which;

        let values = match reader.which().map_err(Error::new)? {
            Which::None(()) => {
                let alias =
                    text_to_str(reader.get_uid().map_err(Error::new)?).unwrap_or("<unknown>");
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
                for driver_idx in explicit.get_driven_by().map_err(Error::new)?.iter() {
                    let driver_id = self.resolve_parameter_index(driver_idx)?;
                    self.driving_parameters
                        .entry(driver_id.into())
                        .or_default()
                        .push(uid);
                }
                deserialize_explicit_sweep(&explicit)?
            }
        };

        let axis_name = if reader.has_axis_name() {
            let alias = text_to_str(reader.get_axis_name().map_err(Error::new)?)?;
            if alias.is_empty() {
                return Err(Error::new(format!(
                    "Sweep parameter with UID '{}' has an empty axis name",
                    text_to_str(reader.get_uid().map_err(Error::new)?)?
                )));
            }
            Some(self.id_store.get_or_insert(alias))
        } else {
            Default::default()
        };

        if let Some(axis_name) = axis_name {
            return SweepParameter::new_with_axis_name(uid, values, axis_name).map_err(Error::new);
        }
        SweepParameter::new(uid, values).map_err(Error::new)
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
                let function = pulse_function(uri)?;
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
                            let (pairs, _) = data.as_chunks::<8>().0.as_chunks::<2>();
                            let complex: Vec<Complex64> = pairs
                                .iter()
                                .map(|&[re, im]| {
                                    Complex64::new(f64::from_le_bytes(re), f64::from_le_bytes(im))
                                })
                                .collect();
                            NumericArray::Complex64(complex)
                        } else {
                            let (words, _) = data.as_chunks::<8>();
                            let floats: Vec<f64> =
                                words.iter().map(|&w| f64::from_le_bytes(w)).collect();
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

    // === HQCS declaration deserialization ===

    /// Deserialize the experiment-level HQCS declarations (coprocessors,
    /// mappings, variables, streams) into `self.hqcs`.
    ///
    /// Must run after `prepass_register_uids` (stream-field handle bindings
    /// resolve acquisition-handle indices) and before `deserialize_root_sections`
    /// (sends, matches, and predicates resolve variable/stream indices).
    fn deserialize_hqcs_declarations(
        &mut self,
        experiment: &experiment_capnp::experiment::Reader<'_>,
    ) -> Result<()> {
        let coprocessors = experiment.get_coprocessors().map_err(Error::new)?;
        self.hqcs.coprocessors = Vec::with_capacity(coprocessors.len() as usize);
        self.hqcs.mappings = Vec::new();
        for coproc in coprocessors.iter() {
            let label = text_to_str(coproc.get_label().map_err(Error::new)?)?;
            let payload = coproc.get_payload().map_err(Error::new)?;
            self.hqcs.coprocessors.push(CoprocessorDecl {
                label: label.to_owned(),
                payload: payload.to_vec(),
            });
            let inventory_key = text_to_str(coproc.get_inventory_key().map_err(Error::new)?)?;
            if !inventory_key.is_empty() {
                self.hqcs.mappings.push(CoprocessorMappingDecl {
                    label: label.to_owned(),
                    inventory_key: inventory_key.to_owned(),
                });
            }
        }

        let variables = experiment.get_variables().map_err(Error::new)?;
        self.hqcs.variables = Vec::with_capacity(variables.len() as usize);
        for variable in variables.iter() {
            let ty = coproc_type_from_capnp(variable.get_type().map_err(Error::new)?)?;
            let name = text_to_str(variable.get_name().map_err(Error::new)?)?;
            let log_handle = text_to_str(variable.get_log_handle().map_err(Error::new)?)?;
            // An absent `initial` would read back as the zero-valued first union
            // member, so presence has to be probed explicitly.
            let initial = if variable.has_initial() {
                let value = variable.get_initial().map_err(Error::new)?;
                Some(coproc_value_from_capnp(&value)?)
            } else {
                None
            };
            self.hqcs.variables.push(VariableDecl {
                ty,
                name: (!name.is_empty()).then(|| name.to_owned()),
                initial,
                log_handle: (!log_handle.is_empty()).then(|| log_handle.to_owned()),
            });
        }

        let streams = experiment.get_streams().map_err(Error::new)?;
        self.hqcs.streams = Vec::with_capacity(streams.len() as usize);
        let mut seen_stream_uids: HashSet<&str> = HashSet::new();
        for stream in streams.iter() {
            let resolve_endpoint = |idx: u32| -> Result<StreamEndpoint> {
                if (idx as usize) < self.hqcs.coprocessors.len() {
                    Ok(StreamEndpoint::Coprocessor(idx))
                } else {
                    Err(Error::new(format!(
                        "Stream references coprocessor index {idx}, but only {} \
                         coprocessors are declared",
                        self.hqcs.coprocessors.len()
                    )))
                }
            };
            use coprocessor_capnp::stream::{dst::Which as DstWhich, src::Which as SrcWhich};
            let src = match stream.get_src().which().map_err(Error::new)? {
                SrcWhich::ControlSystem(()) => StreamEndpoint::ControlSystem,
                SrcWhich::Coprocessor(idx) => resolve_endpoint(idx)?,
            };
            let dst = match stream.get_dst().which().map_err(Error::new)? {
                DstWhich::ControlSystem(()) => StreamEndpoint::ControlSystem,
                DstWhich::Coprocessor(idx) => resolve_endpoint(idx)?,
            };

            let link = text_to_str(stream.get_link().map_err(Error::new)?)?;
            let uid = text_to_str(stream.get_uid().map_err(Error::new)?)?;
            if !seen_stream_uids.insert(uid) {
                return Err(Error::new(format!(
                    "Duplicate stream uid '{uid}'; stream uids must be unique \
                     within an experiment."
                )));
            }

            let field_readers = stream.get_fields().map_err(Error::new)?;
            let mut fields = Vec::with_capacity(field_readers.len() as usize);
            for field in field_readers.iter() {
                let field_name = text_to_str(field.get_name().map_err(Error::new)?)?;
                let ty = coproc_type_from_capnp(field.get_type().map_err(Error::new)?)?;
                use coprocessor_capnp::struct_field::binding::Which as BindingWhich;
                let binding = match field.get_binding().which().map_err(Error::new)? {
                    BindingWhich::Unbound(()) => FieldBinding::Unbound,
                    BindingWhich::Handles(handles) => {
                        let handles = handles.map_err(Error::new)?;
                        let mut resolved = Vec::with_capacity(handles.len() as usize);
                        for idx in handles.iter() {
                            resolved.push(HandleUid(self.resolve_handle_index(idx)?));
                        }
                        FieldBinding::Handles(resolved)
                    }
                    BindingWhich::Variable(idx) => {
                        FieldBinding::Variable(self.resolve_variable_index(idx)?)
                    }
                    BindingWhich::Pulse(idx) => {
                        FieldBinding::Pulse(PulseUid(self.resolve_pulse_index(idx)?))
                    }
                };
                fields.push(StreamFieldDecl {
                    name: field_name.to_owned(),
                    ty,
                    binding,
                });
            }

            self.hqcs.streams.push(StreamDecl {
                uid: uid.to_owned(),
                src,
                dst,
                link: (!link.is_empty()).then(|| link.to_owned()),
                fields,
            });
        }

        Ok(())
    }

    fn deserialize_do_until(
        &mut self,
        reader: &section_capnp::do_until_section::Reader<'_>,
        section_uid: SectionUid,
        section_name: &str,
    ) -> Result<DoUntil> {
        let condition_reader = reader.get_condition().map_err(Error::new)?;
        let condition = self.deserialize_predicate(&condition_reader, &|| {
            format!("do_until section '{section_name}'")
        })?;
        Ok(DoUntil {
            uid: section_uid,
            condition,
            max_count: reader.get_max_count(),
        })
    }

    fn deserialize_predicate(
        &mut self,
        reader: &coprocessor_capnp::predicate::Reader<'_>,
        owner: &dyn Fn() -> String,
    ) -> Result<Predicate> {
        use coprocessor_capnp::predicate::Which;
        match reader.which().map_err(Error::new)? {
            Which::Unspecified(()) => Err(Error::new(format!("{} has no condition set", owner()))),
            Which::Comparison(cmp) => {
                let variable = self.resolve_variable_index(cmp.get_variable())?;
                let op = match cmp.get_op().map_err(Error::new)? {
                    coprocessor_capnp::CmpOp::Unspecified => {
                        return Err(Error::new(format!(
                            "{} has no comparison operator set",
                            owner()
                        )));
                    }
                    coprocessor_capnp::CmpOp::Eq => CmpOp::Eq,
                    coprocessor_capnp::CmpOp::Ne => CmpOp::Ne,
                    coprocessor_capnp::CmpOp::Lt => CmpOp::Lt,
                    coprocessor_capnp::CmpOp::Le => CmpOp::Le,
                    coprocessor_capnp::CmpOp::Gt => CmpOp::Gt,
                    coprocessor_capnp::CmpOp::Ge => CmpOp::Ge,
                };
                let rhs = coproc_value_from_capnp(&cmp.get_rhs().map_err(Error::new)?)?;
                Ok(Predicate::Comparison { variable, op, rhs })
            }
            Which::IsLive(is_live) => {
                use coprocessor_capnp::predicate::is_live::Which as IsLiveWhich;
                let target = match is_live.which().map_err(Error::new)? {
                    IsLiveWhich::Variable(idx) => {
                        IsLiveTarget::Variable(self.resolve_variable_index(idx)?)
                    }
                    IsLiveWhich::Pulse(idx) => {
                        IsLiveTarget::Pulse(PulseUid(self.resolve_pulse_index(idx)?))
                    }
                };
                Ok(Predicate::IsLive(target))
            }
        }
    }

    fn deserialize_send_op(
        &mut self,
        reader: &operation_capnp::send_op::Reader<'_>,
    ) -> Result<Send> {
        let stream = self.resolve_stream_index(reader.get_stream())?;
        let arg_readers = reader.get_args().map_err(Error::new)?;
        let mut args = Vec::with_capacity(arg_readers.len() as usize);
        for arg in arg_readers.iter() {
            let name = text_to_str(arg.get_name().map_err(Error::new)?)?;
            let value = coproc_value_from_capnp(&arg.get_value().map_err(Error::new)?)?;
            args.push(SendArg {
                name: name.to_owned(),
                value,
            });
        }
        Ok(Send { stream, args })
    }

    fn deserialize_mark_stale_op(
        &mut self,
        reader: &operation_capnp::mark_stale_op::Reader<'_>,
    ) -> Result<MarkStale> {
        use operation_capnp::mark_stale_op::target::Which;
        let target = match reader.get_target().which().map_err(Error::new)? {
            Which::Variable(idx) => MarkStaleTarget::Variable(self.resolve_variable_index(idx)?),
            Which::Pulse(idx) => MarkStaleTarget::Pulse(PulseUid(self.resolve_pulse_index(idx)?)),
        };
        Ok(MarkStale { target })
    }

    fn deserialize_ensure_op(
        &mut self,
        reader: &operation_capnp::ensure_op::Reader<'_>,
        parent_section_name: &str,
    ) -> Result<Ensure> {
        let condition_reader = reader.get_condition().map_err(Error::new)?;
        let owner = || {
            format!(
                "ensure in {}",
                section_label_for_errors(parent_section_name)
            )
        };
        let condition = self.deserialize_predicate(&condition_reader, &owner)?;
        Ok(Ensure { condition })
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
                    root.children.push(node);
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
            Which::DoUntil(do_until) => {
                let do_until = do_until.map_err(Error::new)?;
                Operation::DoUntil(self.deserialize_do_until(&do_until, section_uid, name)?)
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
                    children.push(child);
                }
                ItemWhich::Operation(op) => {
                    let op = op.map_err(Error::new)?;
                    let child = self.deserialize_operation(&op, name)?;
                    children.push(child);
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

        let section_timing_mode =
            deserialize_section_timing_mode(reader.get_section_timing_mode().map_err(Error::new)?);

        Ok(Section {
            uid: section_uid,
            alignment,
            length,
            play_after: deserialize_play_after_from_regular(reader, section_name, sibling_refs)?,
            triggers,
            on_system_grid: reader.get_on_system_grid(),
            section_timing_mode,
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
        let mut direct_parameters = Vec::with_capacity(param_uids.len() as usize);
        for idx in param_uids.iter() {
            let param_uid = ParameterUid(self.resolve_parameter_index(idx)?);
            if !parameters.contains(&param_uid) {
                parameters.push(param_uid);
            }
            if !direct_parameters.contains(&param_uid) {
                direct_parameters.push(param_uid);
            }
            // Also include all derived parameters driven by the explicitly listed parameters.
            if let Some(derived_params) = self.driving_parameters.get(&param_uid) {
                for &derived in derived_params {
                    if !parameters.contains(&derived) {
                        parameters.push(derived);
                    }
                }
            }
        }
        if parameters.is_empty() {
            return Err(Error::new("Sweep must have at least one parameter"));
        }

        // Derive the sweep count from the first parameter's length.
        let count = parameters
            .first()
            .and_then(|uid| self.parameters.iter().find(|p| p.uid == *uid))
            .map(|p| p.len() as u32)
            .and_then(NonZeroU32::new)
            .ok_or_else(|| Error::new("Sweep parameter must have at least one value"))?;

        let chunk_count = reader.get_chunk_count();
        let auto_chunking = reader.get_auto_chunking();

        let section_timing_mode =
            deserialize_section_timing_mode(reader.get_section_timing_mode().map_err(Error::new)?);

        Ok(Sweep {
            uid: section_uid,
            parameters,
            direct_parameters,
            count,
            alignment,
            reset_oscillator_phase: reader.get_reset_oscillator_phase(),
            chunk_count: chunk_count.try_into().map_err(|_| {
                Error::new(format!(
                    "Chunk count must be >= 1, but {} was provided.",
                    chunk_count
                ))
            })?,
            auto_chunking,
            section_timing_mode,
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
            Which::Variable(idx) => MatchTarget::Variable(self.resolve_variable_index(idx)?),
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
            Which::Call(call) => {
                let call = call.map_err(Error::new)?;
                Operation::NearTimeCallback(self.deserialize_call_op(call)?)
            }
            Which::SetNode(set_node) => {
                let set_node = set_node.map_err(Error::new)?;
                Operation::SetNode(self.deserialize_set_node_op(set_node)?)
            }
            Which::ResetOscillatorPhase(reset) => {
                let reset = reset.map_err(Error::new)?;
                Operation::ResetOscillatorPhase(self.deserialize_reset_oscillator_phase(&reset)?)
            }
            Which::Send(send) => {
                let send = send.map_err(Error::new)?;
                Operation::Send(self.deserialize_send_op(&send)?)
            }
            Which::MarkStale(mark_stale) => {
                let mark_stale = mark_stale.map_err(Error::new)?;
                Operation::MarkStale(self.deserialize_mark_stale_op(&mark_stale)?)
            }
            Which::Ensure(ensure) => {
                let ensure = ensure.map_err(Error::new)?;
                Operation::Ensure(self.deserialize_ensure_op(&ensure, parent_section_name)?)
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
                let numeric = deserialize_constant_numeric(&constant)?;
                let f: f64 = numeric.try_into().map_err(Error::new)?;
                Some(seconds(f))
            }
            common_capnp::value::Which::ParameterRef(_) => {
                return Err(Error::new("Acquire length cannot be a parameter reference"));
            }
        };

        // Per-operation kernel parameter overrides → Acquire.parameters.
        // Only pad to kernel length when no kernel parameters were provided.
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

    fn deserialize_call_op(
        &mut self,
        reader: operation_capnp::call_op::Reader<'_>,
    ) -> Result<NearTimeCallback> {
        let callback_id = text_to_str(reader.get_callback_id().map_err(Error::new)?)?;
        let callback_id = self.id_store.get_or_insert(callback_id);

        let arg_reader = reader.get_arguments().map_err(Error::new)?;
        let mut args = Vec::with_capacity(arg_reader.len() as usize);
        for arg in arg_reader.iter() {
            let key = text_to_str(arg.get_key().map_err(Error::new)?)?;
            let value = self.deserialize_value(&arg.get_value().map_err(Error::new)?)?;
            let arg = ValueEntry {
                key: self.id_store.get_or_insert(key),
                value,
            };
            args.push(arg);
        }
        let callback = NearTimeCallback { callback_id, args };
        Ok(callback)
    }

    fn deserialize_set_node_op(
        &mut self,
        reader: operation_capnp::set_node_op::Reader<'_>,
    ) -> Result<SetNode> {
        let path = text_to_str(reader.get_path().map_err(Error::new)?)?;
        let path = self.id_store.get_or_insert(path);

        let value = self.deserialize_value_value(&reader.get_value().map_err(Error::new)?)?;
        if let Some(value) = value {
            let set_node = SetNode { path, value };
            return Ok(set_node);
        }
        Err(Error::new("SetNode value must be specified"))
    }

    // === Value deserialization helpers ===

    fn deserialize_value_value(
        &self,
        reader: &common_capnp::value::Reader<'_>,
    ) -> Result<Option<ValueOrParameter<Literal>>> {
        use common_capnp::value::Which;
        match reader.which().map_err(Error::new)? {
            Which::None(()) => Ok(None),
            Which::Constant(constant) => {
                let constant = constant.map_err(Error::new)?;
                let value = deserialize_constant(&constant)?;
                Ok(Some(ValueOrParameter::Value(value)))
            }
            Which::ParameterRef(idx) => Ok(Some(ValueOrParameter::Parameter(ParameterUid(
                self.resolve_parameter_index(idx)?,
            )))),
        }
    }

    fn deserialize_value_f64(
        &self,
        reader: &common_capnp::value::Reader<'_>,
    ) -> Result<Option<ValueOrParameter<f64>>> {
        use common_capnp::value::Which;
        match reader.which().map_err(Error::new)? {
            Which::None(()) => Ok(None),
            Which::Constant(constant) => {
                let constant = constant.map_err(Error::new)?;
                let f: f64 = deserialize_constant_numeric(&constant)?
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
                let f: f64 = deserialize_constant_numeric(&constant)?
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
                let cf: ComplexOrFloat = deserialize_constant_numeric(&constant)?
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
    ) -> Result<HashMap<PulseParameterUid, ExternalOrValue>> {
        let mut result = HashMap::new();
        for entry in reader.iter() {
            let key = text_to_str(entry.get_key().map_err(Error::new)?)?;
            let key_id = PulseParameterUid(self.id_store.get_or_insert(key));
            let value_reader = entry.get_value().map_err(Error::new)?;
            let value = self.deserialize_value(&value_reader)?;
            result.insert(key_id, value);
        }
        Ok(result)
    }

    /// Hand an opaque value to the store and reference it by the UID it assigns.
    ///
    /// Attaching to the interpreter is confined to this method: it is the only
    /// point in deserialization that touches Python.
    fn store_external(&mut self, payload: PyObjectPayload<'_>) -> Result<ExternalOrValue> {
        let uid = Python::attach(|py| self.external_parameters.store(py, payload))?;
        Ok(ExternalOrValue::ExternalParameter(uid))
    }

    fn deserialize_value(
        &mut self,
        reader: &common_capnp::value::Reader<'_>,
    ) -> Result<ExternalOrValue> {
        use common_capnp::value::Which;
        match reader.which().map_err(Error::new)? {
            Which::None(()) => Ok(ExternalOrValue::None),
            Which::Constant(constant) => {
                let constant = constant.map_err(Error::new)?;
                use common_capnp::constant::Which as ConstantWhich;
                match constant.which().map_err(Error::new)? {
                    ConstantWhich::StringValue(value) => {
                        let value = text_to_str(value.map_err(Error::new)?)?;
                        self.store_external(PyObjectPayload::Str(value))
                    }
                    // `pythonValue` carries values that the Rust compiler never interprets:
                    // they are evaluated in Python during waveform sampling, or handed to the
                    // Controller as-is. Only JSON-representable values are accepted; the
                    // serializer rejects anything else.
                    ConstantWhich::PythonValue(data) => {
                        let data = data.map_err(Error::new)?;
                        self.store_external(PyObjectPayload::Json(data))
                    }
                    ConstantWhich::RawBytesValue(data) => {
                        let data = data.map_err(Error::new)?;
                        self.store_external(PyObjectPayload::RawBytes(data))
                    }
                    ConstantWhich::Boolean(value) => Ok(ExternalOrValue::Bool(value)),
                    _ => Ok(ExternalOrValue::ValueOrParameter(ValueOrParameter::Value(
                        deserialize_constant_numeric(&constant)?,
                    ))),
                }
            }
            Which::ParameterRef(idx) => Ok(ExternalOrValue::ValueOrParameter(
                ValueOrParameter::Parameter(ParameterUid(self.resolve_parameter_index(idx)?)),
            )),
        }
    }

    /// Lifts `DeviceSetup.setupDescription` out of the message without decoding it: only the
    /// backend named by `DeviceSetup.backend` knows the schema behind the pointer.
    fn deserialize_device_setup(
        &mut self,
        experiment: &experiment_capnp::experiment::Reader<'_>,
    ) -> Result<SetupDescriptionPayload> {
        let device_setup = experiment.get_device_setup().map_err(Error::new)?;
        let backend = text_to_str(device_setup.get_backend().map_err(Error::new)?)?;
        SetupDescriptionPayload::from_pointer(backend, device_setup.get_setup_description())
    }

    /// Deserialize signals from the device setup section
    fn deserialize_experiment_signals(
        &mut self,
        reader: &experiment_capnp::experiment::Reader<'_>,
    ) -> Result<Vec<ExperimentSignal>> {
        let signals_reader = reader.get_signals().map_err(Error::new)?;

        let mut experiment_signals = Vec::with_capacity(signals_reader.len() as usize);

        for signal in signals_reader.iter() {
            let signal_uid = text_to_str(signal.get_uid().map_err(Error::new)?)?;
            let maps_to = text_to_str(signal.get_maps_to().map_err(Error::new)?)?;

            let calibration = self
                .deserialize_signal_calibration(signal.get_calibration().map_err(Error::new)?)
                .map_err(|e| {
                    Error::new(format!(
                        "Failed to deserialize calibration for signal '{signal_uid}': {e}"
                    ))
                })?;
            experiment_signals.push(ExperimentSignal {
                uid: self.id_store.get_or_insert(signal_uid).into(),
                calibration,
                maps_to: maps_to.to_string(),
            });
        }
        Ok(experiment_signals)
    }

    fn deserialize_signal_calibration(
        &mut self,
        reader: calibration_capnp::signal_calibration::Reader<'_>,
    ) -> Result<SignalCalibration> {
        let oscillator = reader
            .has_oscillator()
            .then(|| self.deserialize_oscillator(reader.get_oscillator().map_err(Error::new)?))
            .transpose()?;
        let automute = reader.get_automute();
        let signal_delay = reader.get_delay_signal().into();
        let lo_frequency = self.deserialize_value_f64(
            &reader
                .get_local_oscillator_frequency()
                .map_err(Error::new)?,
        )?;
        let voltage_offset =
            self.deserialize_value_f64(&reader.get_voltage_offset().map_err(Error::new)?)?;

        let port_delay =
            self.deserialize_value_duration(&reader.get_port_delay().map_err(Error::new)?)?;
        let range = reader
            .has_range()
            .then(|| self.deserialize_range(reader.get_range().map_err(Error::new)?))
            .transpose()?;
        let amplitude = self.deserialize_value_f64(&reader.get_amplitude().map_err(Error::new)?)?;

        let added_outputs_reader = reader.get_added_outputs().map_err(Error::new)?;
        let mut added_outputs = Vec::with_capacity(added_outputs_reader.len() as usize);
        for added_output in added_outputs_reader.iter() {
            added_outputs.push(self.deserialize_added_output(added_output)?);
        }

        let port_mode = match reader.get_port_mode().map_err(Error::new)? {
            calibration_capnp::PortMode::Lf => Some(PortMode::LF),
            calibration_capnp::PortMode::Rf => Some(PortMode::RF),
            calibration_capnp::PortMode::Unspecified => None,
        };

        let precompensation = reader
            .has_precompensation()
            .then(|| {
                self.deserialize_precompensation(reader.get_precompensation().map_err(Error::new)?)
            })
            .transpose()?;

        let amplifier_pump = reader
            .has_amplifier_pump()
            .then(|| {
                self.deserialize_amplifier_pump(reader.get_amplifier_pump().map_err(Error::new)?)
            })
            .transpose()?;

        let thresholds = reader
            .has_threshold()
            .then(|| -> Result<Vec<f64>> {
                Ok(reader.get_threshold().map_err(Error::new)?.iter().collect())
            })
            .transpose()?
            .unwrap_or_default();

        let mixer_calibration = reader
            .has_mixer_calibration()
            .then(|| {
                let mc_reader = reader.get_mixer_calibration().map_err(Error::new)?;

                let voltage_offset_i = self.deserialize_value_f64(
                    &mc_reader.get_voltage_offset_i().map_err(Error::new)?,
                )?;
                let voltage_offset_q = self.deserialize_value_f64(
                    &mc_reader.get_voltage_offset_q().map_err(Error::new)?,
                )?;

                let correction_matrix = if mc_reader.has_correction_matrix() {
                    let matrix_values = mc_reader.get_correction_matrix().map_err(Error::new)?;
                    if matrix_values.len() != 4 {
                        return Err(Error::new(format!(
                            "Correction matrix must have exactly 4 elements, got {}",
                            matrix_values.len()
                        )));
                    }
                    let values = [
                        self.deserialize_value_f64(&matrix_values.get(0))?.ok_or_else(|| {
                            Error::new("Correction matrix element must be a specified value or parameter reference")
                        })?,
                        self.deserialize_value_f64(&matrix_values.get(1))?.ok_or_else(|| {
                            Error::new("Correction matrix element must be a specified value or parameter reference")
                        })?,
                        self.deserialize_value_f64(&matrix_values.get(2))?.ok_or_else(|| {
                            Error::new("Correction matrix element must be a specified value or parameter reference")
                        })?,
                        self.deserialize_value_f64(&matrix_values.get(3))?.ok_or_else(|| {
                            Error::new("Correction matrix element must be a specified value or parameter reference")
                        })?,
                    ];
                    Some(CorrectionMatrix::from_row_major(values))
                } else {
                    None
                };

                Ok(MixerCalibration {
                    voltage_offset_i,
                    voltage_offset_q,
                    correction_matrix,
                })
            })
            .transpose()?;

        let calibration = SignalCalibration {
            oscillator,
            automute,
            signal_delay,
            lo_frequency,
            voltage_offset,
            port_delay,
            range,
            added_outputs,
            port_mode,
            precompensation,
            amplifier_pump,
            amplitude,
            thresholds,
            mixer_calibration,
        };
        Ok(calibration)
    }

    fn deserialize_oscillator(
        &mut self,
        reader: calibration_capnp::oscillator::Reader<'_>,
    ) -> Result<Oscillator> {
        let uid = text_to_str(reader.get_uid().map_err(Error::new)?)?;
        let frequency = self
            .deserialize_value_f64(&reader.get_frequency().map_err(Error::new)?)?
            .unwrap_or(ValueOrParameter::Value(0.0));
        let kind: OscillatorKind = match reader.get_modulation_type().map_err(Error::new)? {
            calibration_capnp::ModulationType::Auto => OscillatorKind::Auto,
            calibration_capnp::ModulationType::Hardware => OscillatorKind::Hardware,
            calibration_capnp::ModulationType::Software => OscillatorKind::Software,
        };
        let osc = Oscillator {
            uid: self.id_store.get_or_insert(uid).into(),
            frequency,
            kind,
        };
        Ok(osc)
    }

    fn deserialize_range(
        &self,
        reader: calibration_capnp::signal_range::Reader<'_>,
    ) -> Result<Quantity> {
        let range_unit = if reader.has_unit() {
            match reader
                .get_unit()
                .map_err(Error::new)?
                .to_string()
                .map_err(Error::new)?
                .to_lowercase()
                .as_str()
            {
                "dbm" => Some(Unit::Dbm),
                "volt" => Some(Unit::Volt),
                other => {
                    return Err(Error::new(format!("Unknown range unit '{other}'",)));
                }
            }
        } else {
            None
        };
        let q = Quantity {
            value: reader.get_value(),
            unit: range_unit,
        };
        Ok(q)
    }

    fn deserialize_added_output(
        &mut self,
        reader: calibration_capnp::output_route::Reader<'_>,
    ) -> Result<OutputRoute> {
        let source_signal = text_to_str(reader.get_source_signal().map_err(Error::new)?)?;
        let amplitude_scaling =
            self.deserialize_value_f64(&reader.get_amplitude_scaling().map_err(Error::new)?)?;
        let phase_shift =
            self.deserialize_value_f64(&reader.get_phase_shift().map_err(Error::new)?)?;

        let route = OutputRoute {
            // The source signal should actually be a logical signal name (to match current DSL),
            // but the serializer currently puts the port number here, so we parse it as a port for now. This will be fixed in the serializer later.
            source_channel: self.id_store.get_or_insert(source_signal).into(),
            amplitude_scaling,
            phase_shift,
        };
        Ok(route)
    }

    fn deserialize_precompensation(
        &self,
        reader: calibration_capnp::precompensation::Reader<'_>,
    ) -> Result<Precompensation> {
        let bounce = reader
            .has_bounce()
            .then(|| reader.get_bounce())
            .transpose()
            .map_err(Error::new)?
            .map(|b| BounceCompensation {
                delay: b.get_delay(),
                amplitude: b.get_amplitude(),
            });

        let exponentials = reader
            .has_exponentials()
            .then(|| reader.get_exponentials())
            .transpose()
            .map_err(Error::new)?
            .map(|list| {
                list.iter()
                    .map(|exp| ExponentialCompensation {
                        timeconstant: exp.get_timeconstant(),
                        amplitude: exp.get_amplitude(),
                    })
                    .collect()
            })
            .unwrap_or_default();

        let fir = reader
            .has_fir()
            .then(|| {
                let fir = reader.get_fir().map_err(Error::new)?;
                let coefficients = fir
                    .get_coefficients()
                    .map_err(Error::new)?
                    .into_iter()
                    .collect::<Vec<f64>>();
                Ok::<FirCompensation, Error>(FirCompensation { coefficients })
            })
            .transpose()?;

        let high_pass = reader
            .has_high_pass()
            .then(|| reader.get_high_pass())
            .transpose()
            .map_err(Error::new)?
            .map(|hp| HighPassCompensation {
                timeconstant: hp.get_timeconstant(),
            });

        let precompensation = Precompensation {
            bounce,
            exponential: exponentials,
            fir,
            high_pass,
        };
        Ok(precompensation)
    }

    fn deserialize_amplifier_pump(
        &mut self,
        reader: calibration_capnp::amplifier_pump::Reader<'_>,
    ) -> Result<AmplifierPump> {
        let cancellation_source = match reader.get_cancellation_source().map_err(Error::new)? {
            calibration_capnp::CancellationSource::Unspecified => PumpCancellationSource::default(),
            calibration_capnp::CancellationSource::Internal => PumpCancellationSource::Internal,
            calibration_capnp::CancellationSource::External => PumpCancellationSource::External,
        };

        use calibration_capnp::amplifier_pump::cancellation_source_frequency::Which as CancellationFreqWhich;
        let cancellation_source_frequency = match reader
            .get_cancellation_source_frequency()
            .which()
            .map_err(Error::new)?
        {
            CancellationFreqWhich::None(()) => None,
            CancellationFreqWhich::Value(val) => Some(val),
        };

        let amplifier_pump = AmplifierPump {
            alc_on: reader.get_alc_on(),
            pump_on: reader.get_pump_on(),
            pump_filter_on: reader.get_pump_filter_on(),
            pump_power: self
                .deserialize_value_f64(&reader.get_pump_power().map_err(Error::new)?)?,
            pump_frequency: self
                .deserialize_value_f64(&reader.get_pump_frequency().map_err(Error::new)?)?,
            probe_on: reader.get_probe_on(),
            probe_power: self
                .deserialize_value_f64(&reader.get_probe_power().map_err(Error::new)?)?,
            probe_frequency: self
                .deserialize_value_f64(&reader.get_probe_frequency().map_err(Error::new)?)?,
            cancellation_on: reader.get_cancellation_on(),
            cancellation_phase: self
                .deserialize_value_f64(&reader.get_cancellation_phase().map_err(Error::new)?)?,
            cancellation_attenuation: self.deserialize_value_f64(
                &reader.get_cancellation_attenuation().map_err(Error::new)?,
            )?,
            cancellation_source,
            cancellation_source_frequency,
        };
        Ok(amplifier_pump)
    }
}

// === Entry point ===

/// Deserializes a Cap'n Proto experiment message into Rust DSL types.
#[instrument(
    level = "debug",
    name = "laboneq.compiler.deserialize_experiment",
    skip_all
)]
pub(crate) fn deserialize_experiment(bytes: &[u8], packed: bool) -> Result<DeserializedExperiment> {
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
        Deserializer::new().deserialize(&experiment)
    } else {
        let reader = capnp::serialize::read_message_from_flat_slice(&mut &*bytes, reader_options)
            .map_err(|e| Error::new(format!("Failed to read Cap'n Proto message: {e}")))?;
        let experiment = reader
            .get_root::<experiment_capnp::experiment::Reader<'_>>()
            .map_err(Error::new)?;
        Deserializer::new().deserialize(&experiment)
    }
}

#[cfg(test)]
mod tests;

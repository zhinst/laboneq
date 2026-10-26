// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

//! Serializes Python DSL experiment objects to Cap'n Proto binary format.
//!
//! Traverses the Python experiment tree and writes it into the Cap'n Proto
//! schema defined in `laboneq-capnp`. Sections use name strings directly (no
//! numeric UIDs). Entities (signals, parameters, pulses, acquisition handles)
//! are referenced by zero-based `u32` indices.
//!
//! Signals are indexed first (from `experiment.signals`, sorted alphabetically
//! for deterministic UID assignment). All other entities (parameters, pulses,
//! handles) are collected lazily during the write pass in post-order (children
//! before parent for sweeps). PRNG data is inlined directly into the section
//! structs (PrngSetupSection, PrngLoopSection, MatchSection).

use std::collections::{HashMap, HashSet};

use capnp::any_pointer;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyBytes, PyComplex, PyDict, PyFloat, PyList, PyString};
use tracing::instrument;

use laboneq_capnp::pulse::v1::{
    calibration_capnp, common_capnp, coprocessor_capnp, experiment_capnp, operation_capnp,
    pulse_capnp, section_capnp, sweep_capnp,
};
use laboneq_error::{LabOneQError, WithContext, bail, laboneq_error};
use laboneq_py_utils::constant_serializer;
use numeric_array::NumericArray;

use crate::Result;
use crate::dsl_types::{DslType, DslTypes};
use crate::py_helpers::is_exact_type;
use crate::py_types::{
    CancellationSourcePy, ExperimentCapnpPy, ExperimentSignalPy, FieldBindingPy, OscillatorPy,
    UnitPy,
};

// === Intermediate types for collecting parameters and pulses ===

#[derive(Debug)]
enum NumericValue {
    Real(f64),
    Complex(f64, f64),
    Int(i64),
}

#[derive(Debug)]
enum ExplicitValues {
    Real(Vec<f64>),
    Complex(Vec<(f64, f64)>),
    Int(Vec<i64>),
}

#[derive(Debug)]
enum SweepParameterKind {
    Linear {
        start: NumericValue,
        stop: NumericValue,
        count: u32,
    },
    Explicit {
        values: ExplicitValues,
    },
}

#[derive(Debug)]
struct CollectedParameter {
    alias: String,
    kind: SweepParameterKind,
    axis_name: Option<String>,
}

#[derive(Debug)]
enum PulseShape {
    Functional { function: String },
    Sampled { samples: Vec<u8>, is_complex: bool },
}

/// A single definition-level pulse parameter entry (resolved to capnp-ready values).
#[derive(Debug)]
enum PulseParamValue {
    Real(f64),
    Complex(f64, f64),
    Int(i64),
    Json(Vec<u8>),
    RawBytes(Vec<u8>),
    ParameterRef(u32),
}

#[derive(Debug)]
struct PulseParamEntry {
    key: String,
    value: PulseParamValue,
}

#[derive(Debug)]
struct CollectedPulse {
    alias: String,
    can_compress: bool,
    amplitude_re: f64,
    amplitude_im: f64,
    length: Option<f64>,
    shape: PulseShape,
    /// Definition-level pulse parameters from `PulseFunctional.pulse_parameters`.
    /// Only set for functional pulses. UIDs are resolved at collection time.
    functional_params: Vec<PulseParamEntry>,
}

/// An acquisition handle collected during section traversal.
#[derive(Debug)]
struct CollectedHandle {
    name: String,
}

/// Holds all entity collections and their index maps.
///
/// Each entity type uses its own zero-based sequential index. Indices are
/// assigned at insertion time so that lookups always return final indices —
/// no remapping postpass is needed.
struct EntityIndex {
    // --- parameters ---
    parameters: Vec<CollectedParameter>,
    /// uid string → final index
    parameter_indices: HashMap<String, u32>,
    /// Maps derived parameter index → driver parameter UID strings.
    /// Indices cannot be used since not all drivers are necessarily collected as experiment parameters (e.g. intermediate parameters in a chain).
    derived_parameters: HashMap<u32, Vec<String>>,

    // --- pulses ---
    pulses: Vec<CollectedPulse>,
    /// uid string → final index
    pulse_indices: HashMap<String, u32>,

    // --- handles ---
    handles: Vec<CollectedHandle>,
    /// name string → final index
    handle_indices: HashMap<String, u32>,
}

impl EntityIndex {
    fn new() -> Self {
        Self {
            parameters: Vec::new(),
            parameter_indices: HashMap::new(),
            derived_parameters: HashMap::new(),
            pulses: Vec::new(),
            pulse_indices: HashMap::new(),
            handles: Vec::new(),
            handle_indices: HashMap::new(),
        }
    }

    /// Get or insert an acquisition handle, returning its final index.
    fn get_or_insert_handle(&mut self, name: &str) -> u32 {
        if let Some(&idx) = self.handle_indices.get(name) {
            return idx;
        }
        let idx = self.handles.len() as u32;
        self.handle_indices.insert(name.to_owned(), idx);
        self.handles.push(CollectedHandle {
            name: name.to_owned(),
        });
        idx
    }
}

// === HQCS interning state ===

/// A resolved HQCS stream endpoint.
enum HqcsEndpoint {
    ControlSystem,
    Coprocessor(u32),
}

/// Interning state for HQCS entities. Coprocessors are keyed by their label;
/// streams and variables by Python object pointer (`as_ptr`) — the borrowed
/// experiment keeps them alive for the duration of the pass, so pointers are
/// stable.
#[derive(Default)]
struct HqcsIndex<'py> {
    /// Coprocessor label → `Experiment.coprocessors` index.
    coprocessor_ids: HashMap<String, u32>,
    /// Python object pointer → `Experiment.streams` index.
    stream_ids: HashMap<usize, u32>,
    /// Collected variables in id order; written to `Experiment.variables`
    /// after the section pass.
    variables: Vec<Bound<'py, PyAny>>,
    /// Python object pointer → `Experiment.variables` index.
    variable_ids: HashMap<usize, u32>,
}

impl<'py> HqcsIndex<'py> {
    fn new() -> Self {
        Self::default()
    }

    fn get_or_insert_variable(&mut self, obj: &Bound<'py, PyAny>) -> u32 {
        let key = obj.as_ptr() as usize;
        if let Some(&idx) = self.variable_ids.get(&key) {
            return idx;
        }
        let idx = self.variables.len() as u32;
        self.variable_ids.insert(key, idx);
        self.variables.push(obj.clone());
        idx
    }
}

// === Serialization context ===

/// Bundles all shared state for a single experiment serialization pass.
struct Serializer<'py> {
    dsl_types: DslTypes<'py>,
    np: Bound<'py, PyModule>,
    entities: EntityIndex,
    /// Signals in alphabetically-sorted definition order.
    signal_order: Vec<String>,
    /// uid string → final signal index.
    signal_indices: HashMap<String, u32>,
    /// Collected sweep parameters by UID for consistency checking across multiple references.
    collected_sweep_parameters: HashMap<String, Bound<'py, PyAny>>,
    /// HQCS interning state (coprocessors, streams, variables).
    hqcs: HqcsIndex<'py>,
}

impl<'py> Serializer<'py> {
    fn new(py: Python<'py>) -> Result<Self> {
        Ok(Self {
            dsl_types: DslTypes::new(py)?,
            np: py.import(intern!(py, "numpy"))?,
            entities: EntityIndex::new(),
            signal_order: Vec::new(),
            signal_indices: HashMap::new(),
            collected_sweep_parameters: HashMap::new(),
            hqcs: HqcsIndex::new(),
        })
    }

    // === Index lookup helpers ===
    // These always return the final (zero-based) index for the entity.

    fn get_signal_index(&self, uid: &str) -> Result<u32> {
        self.signal_indices.get(uid).copied().ok_or_else(|| {
            let mut available: Vec<&str> = self.signal_indices.keys().map(String::as_str).collect();
            available.sort();
            laboneq_error!(
                "Signal '{}' is not available in the experiment definition. \
                 Available signals are: '{}'.",
                uid,
                available.join(", ")
            )
        })
    }

    // === Top-level serialization steps ===

    fn collect_signals(
        &mut self,
        experiment: &ExperimentCapnpPy,
    ) -> Result<(HashMap<String, u32>, Vec<String>)> {
        // Collect UIDs.
        let mut uid_strings: Vec<String> = Vec::with_capacity(experiment.experiment_signals.len());
        for signal in experiment.experiment_signals.iter() {
            uid_strings.push(signal.uid.to_string());
        }

        // Sort alphabetically for deterministic ordering.
        // Adjacent-window duplicate check is O(N log N) vs the prior O(N²) linear scan.
        uid_strings.sort();
        if let Some(w) = uid_strings.windows(2).find(|w| w[0] == w[1]) {
            bail!("Duplicate signal uid '{}' in experiment.signals", w[0]);
        }

        // Assign indices in sorted order.
        let mut signal_indices = HashMap::with_capacity(uid_strings.len());
        let mut signal_order = Vec::with_capacity(uid_strings.len());

        for uid_str in uid_strings {
            let idx = signal_order.len() as u32;
            signal_indices.insert(uid_str.clone(), idx);
            signal_order.push(uid_str);
        }
        Ok((signal_indices, signal_order))
    }

    fn serialize_signals(
        &mut self,
        experiment: &ExperimentCapnpPy<'py>,
        mut exp_builder: experiment_capnp::experiment::Builder<'_>,
    ) -> Result<()> {
        let (signal_indices, signal_order) = self.collect_signals(experiment)?;

        let signal_by_uid: HashMap<&str, &ExperimentSignalPy> = experiment
            .experiment_signals
            .iter()
            .map(|s| Ok((s.uid.to_str()?, s)))
            .collect::<Result<_>>()?;

        let mut signals_builder = exp_builder
            .reborrow()
            .init_signals(signal_order.len() as u32);
        for (i, uid_str) in signal_order.iter().enumerate() {
            let mut sig_builder = signals_builder.reborrow().get(i as u32);
            sig_builder.set_uid(uid_str.as_str());
            if let Some(&signal) = signal_by_uid.get(uid_str.as_str()) {
                sig_builder.set_maps_to(signal.maps_to.to_str()?);
                self.serialize_signal_calibration(signal, sig_builder)?;
            }
        }
        self.signal_indices = signal_indices;
        self.signal_order = signal_order;
        Ok(())
    }

    fn serialize_root_sections(
        &mut self,
        experiment: &ExperimentCapnpPy<'py>,
        mut exp_builder: experiment_capnp::experiment::Builder<'_>,
    ) -> Result<()> {
        let sections_list = &experiment.sections;
        // Build a root section containing all top-level sections as children.
        let mut root_builder = exp_builder.reborrow().init_root_section();
        let mut items_builder = root_builder
            .reborrow()
            .init_content_items(sections_list.len() as u32);
        for (i, section) in sections_list.iter().enumerate() {
            let item = items_builder.reborrow().get(i as u32);
            let section_builder = item.init_section();
            self.serialize_section(section, section_builder)?;
        }
        Ok(())
    }

    fn write_parameters(
        &mut self,
        mut exp_builder: experiment_capnp::experiment::Builder<'_>,
    ) -> Result<()> {
        let mut params_builder = exp_builder
            .reborrow()
            .init_sweep_parameters(self.entities.parameters.len() as u32);
        for (i, param) in self.entities.parameters.iter().enumerate() {
            let mut pb = params_builder.reborrow().get(i as u32);
            pb.set_uid(&param.alias);
            if let Some(axis_name) = &param.axis_name {
                pb.set_axis_name(axis_name);
            }
            match &param.kind {
                SweepParameterKind::Linear { start, stop, count } => {
                    let mut lin = pb.init_linear();
                    set_linear_start_stop(&mut lin, start, stop);
                    lin.set_count(*count);
                }
                SweepParameterKind::Explicit { values } => {
                    let driven_by: Vec<u32> = self
                        .entities
                        .derived_parameters
                        .get(&(i as u32))
                        .map(|drivers| {
                            drivers
                                .iter()
                                .flat_map(|driver_alias| {
                                    // driver_alias is the alias of a root driver parameter. Look up its UID string for referencing.
                                    self.entities.parameter_indices.get(driver_alias).cloned()
                                })
                                .collect()
                        })
                        .unwrap_or_default();

                    let mut explicit = pb.init_explicit_values();
                    let mut driven_by_builder =
                        explicit.reborrow().init_driven_by(driven_by.len() as u32);
                    for (j, driver_idx) in driven_by.into_iter().enumerate() {
                        driven_by_builder.set(j as u32, driver_idx);
                    }

                    match values {
                        ExplicitValues::Real(vals) => {
                            let mut list = explicit.reborrow().init_real_values(vals.len() as u32);
                            for (j, v) in vals.iter().enumerate() {
                                list.set(j as u32, *v);
                            }
                        }
                        ExplicitValues::Int(vals) => {
                            let mut list = explicit.reborrow().init_int_values(vals.len() as u32);
                            for (j, v) in vals.iter().enumerate() {
                                list.set(j as u32, *v);
                            }
                        }
                        ExplicitValues::Complex(vals) => {
                            let mut list =
                                explicit.reborrow().init_complex_values(vals.len() as u32);
                            for (j, (re, im)) in vals.iter().enumerate() {
                                let mut cv = list.reborrow().get(j as u32);
                                cv.set_real(*re);
                                cv.set_imag(*im);
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    fn write_pulses(
        &mut self,
        mut exp_builder: experiment_capnp::experiment::Builder<'_>,
    ) -> Result<()> {
        let mut pulses_builder = exp_builder
            .reborrow()
            .init_pulses(self.entities.pulses.len() as u32);
        for (i, pulse) in self.entities.pulses.iter().enumerate() {
            let mut pb = pulses_builder.reborrow().get(i as u32);
            pb.set_uid(&pulse.alias);
            pb.set_can_compress(pulse.can_compress);

            // Amplitude
            let mut amp = pb
                .reborrow()
                .get_amplitude()
                .map_err(LabOneQError::from_err)?;
            amp.set_real(pulse.amplitude_re);
            amp.set_imag(pulse.amplitude_im);

            // Length
            if let Some(length) = pulse.length {
                pb.reborrow().init_length().set_value(length);
            }

            // Shape
            match &pulse.shape {
                PulseShape::Functional { function } => {
                    let mut func = pb.reborrow().init_functional();
                    let uri = match function.as_str() {
                        "const" => "py://const".to_owned(),
                        other => format!("py://{other}"),
                    };
                    func.reborrow().set_sampler_uri(&uri);
                    // Write definition-level pulse parameters (already resolved at collection time).
                    if !pulse.functional_params.is_empty() {
                        let mut entries =
                            func.init_parameters(pulse.functional_params.len() as u32);
                        for (j, param) in pulse.functional_params.iter().enumerate() {
                            let mut entry = entries.reborrow().get(j as u32);
                            entry.set_key(&param.key);
                            let mut val = entry.init_value();
                            match &param.value {
                                PulseParamValue::Real(v) => {
                                    val.reborrow().init_constant().set_real(*v);
                                }
                                PulseParamValue::Complex(re, im) => {
                                    let mut c = val.reborrow().init_constant().init_complex();
                                    c.set_real(*re);
                                    c.set_imag(*im);
                                }
                                PulseParamValue::Int(v) => {
                                    val.reborrow().init_constant().set_integer(*v);
                                }
                                PulseParamValue::Json(bytes) => {
                                    val.reborrow().init_constant().set_python_value(bytes);
                                }
                                PulseParamValue::RawBytes(bytes) => {
                                    val.reborrow().init_constant().set_raw_bytes_value(bytes);
                                }
                                PulseParamValue::ParameterRef(idx) => {
                                    // idx is already the final parameter index.
                                    val.set_parameter_ref(*idx);
                                }
                            }
                        }
                    }
                }
                PulseShape::Sampled {
                    samples,
                    is_complex,
                } => {
                    let mut sampled = pb.init_sampled();
                    if *is_complex {
                        sampled.set_sample_type(pulse_capnp::SampleType::Complex);
                    } else {
                        sampled.set_sample_type(pulse_capnp::SampleType::Real);
                    }
                    // We store the raw sample bytes inline.
                    let waveform_data = sampled.init_samples();
                    let mut inline_data = waveform_data.init_inline();
                    inline_data.set_data(samples);
                    inline_data.set_sample_count(if *is_complex {
                        samples.len() / 16
                    } else {
                        samples.len() / 8
                    } as u64);
                    inline_data.set_data_type(if *is_complex {
                        pulse_capnp::WaveformDataType::Complex128
                    } else {
                        pulse_capnp::WaveformDataType::Float64
                    });
                }
            }
        }
        Ok(())
    }

    fn write_handles(
        &mut self,
        mut exp_builder: experiment_capnp::experiment::Builder<'_>,
    ) -> Result<()> {
        let mut handles_builder = exp_builder
            .reborrow()
            .init_acquisition_handles(self.entities.handles.len() as u32);
        for (i, handle) in self.entities.handles.iter().enumerate() {
            let mut hb = handles_builder.reborrow().get(i as u32);
            hb.set_uid(&handle.name);
        }
        Ok(())
    }

    // === HQCS serialization ===

    fn serialize_hqcs_coprocessors(
        &mut self,
        experiment: &ExperimentCapnpPy<'py>,
        mut exp_builder: experiment_capnp::experiment::Builder<'_>,
    ) -> Result<()> {
        // In the DSL, we have independent coprocessor concepts in the experiment (a label
        // declared with `@experiment(coprocessors=[...])`) and in the device setup
        // (`CoprocessorInventoryEntry`), with a mapping between them.
        // This mapping mimics the mapping of signals.
        // Unlike for signals though, the coprocessors are always mapped in a 1:1 fashion
        // (no two experiment coprocessors may be mapped to the same physical one), so we
        // erase that mapping in the capnp representation.
        let mut coproc_labels: HashSet<String> = HashSet::new();
        if experiment.coprocessors.is_empty() {
            return Ok(());
        }
        let mut list = exp_builder
            .reborrow()
            .init_coprocessors(experiment.coprocessors.len() as u32);
        for (i, label) in experiment.coprocessors.iter().enumerate() {
            self.hqcs.coprocessor_ids.insert(label.clone(), i as u32);
            let mut builder = list.reborrow().get(i as u32);
            builder.set_label(label);
            if let Some(payload) = experiment.coprocessor_payloads.get(label) {
                builder.set_payload(payload);
            }
            if let Some(key) = Self::resolve_inventory_key(experiment, label)? {
                builder.set_inventory_key(&key);
            }
            coproc_labels.insert(label.clone());
        }
        for (label, _) in experiment.coprocessor_mappings.iter() {
            if !coproc_labels.contains(label) {
                bail!("map_coprocessor target '{label}' does not match any declared coprocessor");
            }
        }
        Ok(())
    }

    /// Resolve the inventory key a coprocessor is mapped to, from the
    /// experiment's label-keyed `map_coprocessor` entries. Returns `None` when
    /// the coprocessor is unmapped. The target may be an inventory key string
    /// or a `CoprocessorInventoryEntry` (has `.key`).
    fn resolve_inventory_key(
        experiment: &ExperimentCapnpPy<'py>,
        coproc_label: &str,
    ) -> Result<Option<String>> {
        for (label, target) in experiment.coprocessor_mappings.iter() {
            if label != coproc_label {
                continue;
            }
            let py = target.py();
            let key: String = if let Ok(s) = target.extract::<String>() {
                s
            } else if let Some(key) = target.getattr_opt(intern!(py, "key"))? {
                key.extract()?
            } else {
                bail!("Unsupported HQCS coprocessor mapping target for '{coproc_label}'");
            };
            return Ok(Some(key));
        }
        Ok(None)
    }

    fn resolve_hqcs_endpoint(&self, endpoint: Option<&String>) -> Result<HqcsEndpoint> {
        let Some(label) = endpoint else {
            return Ok(HqcsEndpoint::ControlSystem);
        };
        match self.hqcs.coprocessor_ids.get(label) {
            Some(&idx) => Ok(HqcsEndpoint::Coprocessor(idx)),
            None => Err(laboneq_error!(
                "HQCS stream references coprocessor '{label}', which is not \
                 declared on the experiment"
            )),
        }
    }

    fn serialize_hqcs_streams(
        &mut self,
        experiment: &ExperimentCapnpPy<'py>,
        mut exp_builder: experiment_capnp::experiment::Builder<'_>,
    ) -> Result<()> {
        if experiment.streams.is_empty() {
            return Ok(());
        }
        let mut list = exp_builder
            .reborrow()
            .init_streams(experiment.streams.len() as u32);
        for (i, stream) in experiment.streams.iter().enumerate() {
            self.hqcs
                .stream_ids
                .insert(stream.obj.as_ptr() as usize, i as u32);
            let mut builder = list.reborrow().get(i as u32);

            {
                let mut src = builder.reborrow().init_src();
                match self.resolve_hqcs_endpoint(stream.src.as_ref())? {
                    HqcsEndpoint::ControlSystem => src.set_control_system(()),
                    HqcsEndpoint::Coprocessor(idx) => src.set_coprocessor(idx),
                }
            }
            {
                let mut dst = builder.reborrow().init_dst();
                match self.resolve_hqcs_endpoint(stream.dst.as_ref())? {
                    HqcsEndpoint::ControlSystem => dst.set_control_system(()),
                    HqcsEndpoint::Coprocessor(idx) => dst.set_coprocessor(idx),
                }
            }

            if let Some(link) = &stream.link {
                builder.reborrow().set_link(link);
            }

            match &stream.uid {
                Some(uid) => builder.reborrow().set_uid(uid),
                // Streams' uid is optional; synthesize a unique per-experiment
                // id so the wire always carries a stable identifier.
                None => builder.reborrow().set_uid(format!("stream_{i}")),
            }

            // Fields, in schema declaration order (dict preserves insertion order).
            let mut fields_builder = builder.reborrow().init_fields(stream.fields.len() as u32);
            for (j, field) in stream.fields.iter().enumerate() {
                let mut fb = fields_builder.reborrow().get(j as u32);
                fb.set_name(&field.name);
                fb.set_type(coproc_type_from_py(&field.ty)?);
                self.serialize_hqcs_field_binding(&field.binding, fb)?;
            }
        }
        Ok(())
    }

    fn serialize_hqcs_field_binding(
        &mut self,
        binding: &FieldBindingPy<'py>,
        mut fb: coprocessor_capnp::struct_field::Builder<'_>,
    ) -> Result<()> {
        let mut b = fb.reborrow().init_binding();
        match binding {
            FieldBindingPy::OutboundHandles(handles) => {
                let mut list = b.init_handles(handles.len() as u32);
                for (k, handle) in handles.iter().enumerate() {
                    list.set(k as u32, self.entities.get_or_insert_handle(handle));
                }
            }
            FieldBindingPy::InboundScalar(target) => match target {
                Some(t) => b.set_variable(self.hqcs.get_or_insert_variable(t)),
                None => b.set_unbound(()),
            },
            FieldBindingPy::InboundPulse(target) => match target {
                Some(t) => {
                    let idx = self.collect_pulse(t)?;
                    b.set_pulse(idx);
                }
                None => b.set_unbound(()),
            },
            FieldBindingPy::Unbound => b.set_unbound(()),
        }
        Ok(())
    }

    fn write_hqcs_variables(
        &mut self,
        mut exp_builder: experiment_capnp::experiment::Builder<'_>,
    ) -> Result<()> {
        if self.hqcs.variables.is_empty() {
            return Ok(());
        }
        let variables = std::mem::take(&mut self.hqcs.variables);
        let mut list = exp_builder
            .reborrow()
            .init_variables(variables.len() as u32);
        for (i, var) in variables.iter().enumerate() {
            let py = var.py();
            let mut builder = list.reborrow().get(i as u32);
            let type_py = var.getattr(intern!(py, "type"))?;
            builder.set_type(coproc_type_from_py(&type_py)?);
            let name_py = var.getattr(intern!(py, "name"))?;
            if !name_py.is_none() {
                let name: String = name_py.extract()?;
                builder.set_name(&name);
            }
            let log_handle_py = var.getattr(intern!(py, "log_handle"))?;
            if !log_handle_py.is_none() {
                let handle: String = log_handle_py.extract()?;
                builder.set_log_handle(&handle);
            }
            let initial_py = var.getattr(intern!(py, "initial"))?;
            if !initial_py.is_none() {
                set_variable_value(builder.init_initial(), &initial_py)?;
            }
        }
        Ok(())
    }

    fn serialize_hqcs_predicate(
        &mut self,
        condition: &Bound<'py, PyAny>,
        builder: coprocessor_capnp::predicate::Builder<'_>,
    ) -> Result<()> {
        let py = condition.py();
        let ty = condition.get_type();
        if ty.is(self.dsl_types.laboneq_type(DslType::HqcsIsLive)) {
            let target_py = condition.getattr(intern!(py, "target"))?;
            let mut is_live = builder.init_is_live();
            if target_py
                .get_type()
                .is(self.dsl_types.laboneq_type(DslType::HqcsVariable))
            {
                is_live.set_variable(self.hqcs.get_or_insert_variable(&target_py));
            } else if target_py.getattr_opt(intern!(py, "uid"))?.is_some() {
                let idx = self.collect_pulse(&target_py)?;
                is_live.set_pulse(idx);
            } else {
                bail!("HQCS is_live target must be a Variable or a Pulse");
            }
        } else if ty.is(self.dsl_types.laboneq_type(DslType::HqcsPredicate)) {
            let lhs_py = condition.getattr(intern!(py, "lhs"))?;
            if !lhs_py
                .get_type()
                .is(self.dsl_types.laboneq_type(DslType::HqcsVariable))
            {
                bail!(
                    "HQCS comparison predicate must have a Variable \
                     on the left-hand side"
                );
            }
            let mut cmp = builder.init_comparison();
            cmp.set_variable(self.hqcs.get_or_insert_variable(&lhs_py));
            let op: String = condition.getattr(intern!(py, "op"))?.extract()?;
            use coprocessor_capnp::CmpOp;
            cmp.set_op(match op.as_str() {
                "==" => CmpOp::Eq,
                "!=" => CmpOp::Ne,
                "<" => CmpOp::Lt,
                "<=" => CmpOp::Le,
                ">" => CmpOp::Gt,
                ">=" => CmpOp::Ge,
                other => {
                    bail!("Unknown HQCS predicate operator: {other}");
                }
            });
            let rhs_py = condition.getattr(intern!(py, "rhs"))?;
            set_variable_value(cmp.init_rhs(), &rhs_py)?;
        } else {
            bail!(
                "HQCS condition must be a comparison predicate \
                 (e.g. `var != 0`) or `is_live(...)`"
            );
        }
        Ok(())
    }

    fn serialize_do_until_section(
        &mut self,
        obj: &Bound<'py, PyAny>,
        builder: &mut section_capnp::section::Builder<'_>,
    ) -> Result<()> {
        let py = obj.py();
        let mut do_until = builder.reborrow().init_do_until();
        let max_count: u32 = obj.getattr(intern!(py, "max_count"))?.extract()?;
        do_until.set_max_count(max_count);
        let condition_py = obj.getattr(intern!(py, "condition"))?;
        if condition_py.is_none() {
            bail!("HQCS do_until section requires a condition");
        }
        let condition_builder = do_until.init_condition();
        self.serialize_hqcs_predicate(&condition_py, condition_builder)?;
        Ok(())
    }

    fn serialize_send_op(
        &mut self,
        obj: &Bound<'py, PyAny>,
        builder: &mut operation_capnp::operation::Builder<'_>,
    ) -> Result<()> {
        let py = obj.py();
        let mut send = builder.reborrow().init_send();
        let stream_py = obj.getattr(intern!(py, "stream"))?;
        let stream_id = self
            .hqcs
            .stream_ids
            .get(&(stream_py.as_ptr() as usize))
            .copied()
            .ok_or_else(|| {
                laboneq_error!(
                    "HQCS send references a stream that is not registered \
                     on the experiment"
                )
            })?;
        send.set_stream(stream_id);
        let kwargs_py = obj.getattr(intern!(py, "literal_kwargs"))?;
        let kwargs: Vec<(String, Bound<'_, PyAny>)> = kwargs_py
            .call_method0(intern!(py, "items"))?
            .try_iter()?
            .map(|item| item.and_then(|i| i.extract()))
            .collect::<PyResult<_>>()?;
        let mut args = send.init_args(kwargs.len() as u32);
        for (i, (name, value)) in kwargs.iter().enumerate() {
            let mut arg = args.reborrow().get(i as u32);
            arg.set_name(name);
            set_variable_value(arg.init_value(), value)?;
        }
        Ok(())
    }

    fn serialize_mark_stale_op(
        &mut self,
        obj: &Bound<'py, PyAny>,
        builder: &mut operation_capnp::operation::Builder<'_>,
    ) -> Result<()> {
        let py = obj.py();
        let mark_stale = builder.reborrow().init_mark_stale();
        let mut target = mark_stale.init_target();
        let target_py = obj.getattr(intern!(py, "target"))?;
        if target_py
            .get_type()
            .is(self.dsl_types.laboneq_type(DslType::HqcsVariable))
        {
            target.set_variable(self.hqcs.get_or_insert_variable(&target_py));
        } else if target_py.getattr_opt(intern!(py, "uid"))?.is_some() {
            let idx = self.collect_pulse(&target_py)?;
            target.set_pulse(idx);
        } else {
            bail!("HQCS mark_stale target must be a Variable or a Pulse");
        }
        Ok(())
    }

    fn serialize_ensure_op(
        &mut self,
        obj: &Bound<'py, PyAny>,
        builder: &mut operation_capnp::operation::Builder<'_>,
    ) -> Result<()> {
        let condition_py = obj.getattr(intern!(obj.py(), "condition"))?;
        let ensure = builder.reborrow().init_ensure();
        self.serialize_hqcs_predicate(&condition_py, ensure.init_condition())
    }

    // === Section serialization ===

    fn serialize_section(
        &mut self,
        obj: &Bound<'py, PyAny>,
        mut builder: section_capnp::section::Builder<'_>,
    ) -> Result<()> {
        let py = obj.py();
        let uid_obj = obj.getattr(intern!(py, "uid"))?;
        let uid_str: &str = uid_obj.extract()?;
        builder.set_name(uid_str);

        let kind = obj.get_type();
        let is_sweep = kind.is(self.dsl_types.laboneq_type(DslType::Sweep));

        // For Sweep sections, we intentionally process children first so that any
        // derived sweep parameters referenced in child operations are collected
        // before writing the sweep's parameter list.
        if is_sweep {
            self.serialize_section_children(obj, &mut builder)?;
        }

        // Determine section kind and serialize appropriately.
        if is_sweep {
            warn_unsupported_section_fields(obj, false)?;
            self.serialize_sweep_section(obj, &mut builder)?;
        } else if obj
            .get_type()
            .is(self.dsl_types.laboneq_type(DslType::AcquireLoopRt))
        {
            warn_unsupported_section_fields(obj, false)?;
            self.serialize_acquire_loop_section(obj, &mut builder)?;
        } else if obj
            .get_type()
            .is(self.dsl_types.laboneq_type(DslType::Match))
        {
            warn_unsupported_section_fields(obj, true)?;
            self.serialize_match_section(obj, &mut builder)?;
        } else if obj
            .get_type()
            .is(self.dsl_types.laboneq_type(DslType::Case))
        {
            warn_unsupported_section_fields(obj, false)?;
            self.serialize_case_section(obj, &mut builder)?;
        } else if obj
            .get_type()
            .is(self.dsl_types.laboneq_type(DslType::PrngSetup))
        {
            warn_unsupported_section_fields(obj, false)?;
            self.serialize_prng_setup_section(obj, &mut builder)?;
        } else if obj
            .get_type()
            .is(self.dsl_types.laboneq_type(DslType::PrngLoop))
        {
            warn_unsupported_section_fields(obj, false)?;
            self.serialize_prng_loop_section(obj, &mut builder)?;
        } else if obj
            .get_type()
            .is(self.dsl_types.laboneq_type(DslType::HqcsDoUntilSection))
        {
            warn_unsupported_section_fields(obj, false)?;
            self.serialize_do_until_section(obj, &mut builder)?;
        } else if obj
            .get_type()
            .is(self.dsl_types.laboneq_type(DslType::Section))
        {
            self.serialize_regular_section(obj, &mut builder)?;
        } else {
            bail!("Unknown section type: {}", obj.get_type());
        }

        if !is_sweep {
            self.serialize_section_children(obj, &mut builder)?;
        }

        Ok(())
    }

    fn serialize_section_children(
        &mut self,
        obj: &Bound<'py, PyAny>,
        builder: &mut section_capnp::section::Builder<'_>,
    ) -> Result<()> {
        let py = obj.py();

        // Serialize children (sections and operations mixed).
        if let Some(children) = obj.getattr_opt(intern!(py, "children"))? {
            let children_list: Vec<Bound<'_, PyAny>> =
                children.try_iter()?.collect::<PyResult<_>>()?;
            let mut items_builder = builder
                .reborrow()
                .init_content_items(children_list.len() as u32);
            for (i, child) in children_list.iter().enumerate() {
                let item = items_builder.reborrow().get(i as u32);
                if self.is_section_type(child)? {
                    let section_builder = item.init_section();
                    self.serialize_section(child, section_builder)?;
                } else {
                    let op_builder = item.init_operation();
                    self.serialize_operation(child, op_builder)?;
                }
            }
        }

        Ok(())
    }

    fn is_section_type(&self, obj: &Bound<'_, PyAny>) -> Result<bool> {
        let ty = obj.get_type();
        Ok(ty.is(self.dsl_types.laboneq_type(DslType::Section))
            || ty.is(self.dsl_types.laboneq_type(DslType::Sweep))
            || ty.is(self.dsl_types.laboneq_type(DslType::AcquireLoopRt))
            || ty.is(self.dsl_types.laboneq_type(DslType::Match))
            || ty.is(self.dsl_types.laboneq_type(DslType::Case))
            || ty.is(self.dsl_types.laboneq_type(DslType::PrngSetup))
            || ty.is(self.dsl_types.laboneq_type(DslType::PrngLoop))
            || ty.is(self.dsl_types.laboneq_type(DslType::HqcsDoUntilSection)))
    }

    fn serialize_regular_section(
        &mut self,
        obj: &Bound<'_, PyAny>,
        builder: &mut section_capnp::section::Builder<'_>,
    ) -> Result<()> {
        let py = obj.py();

        let mut regular = builder.reborrow().init_regular();

        if let Some(alignment) = extract_alignment_capnp(obj)? {
            regular.set_alignment(alignment);
        }

        if let Some(section_timing_mode) = extract_section_timing_mode_capnp(obj)? {
            regular.set_section_timing_mode(section_timing_mode);
        }

        // Length
        let length_py = obj.getattr(intern!(py, "length"))?;
        if let Ok(Some(v)) = length_py.extract::<Option<f64>>() {
            regular.reborrow().init_length().set_value(v);
        }

        // on_system_grid
        let on_system_grid = obj
            .getattr(intern!(py, "on_system_grid"))?
            .extract::<Option<bool>>()?
            .unwrap_or(false);
        regular.set_on_system_grid(on_system_grid);

        // play_after
        let play_after_names = collect_play_after_names(obj)?;
        if !play_after_names.is_empty() {
            let mut pa_builder = regular
                .reborrow()
                .init_play_after(play_after_names.len() as u32);
            for (i, name) in play_after_names.iter().enumerate() {
                pa_builder.set(i as u32, name.as_str());
            }
        }

        // triggers
        self.serialize_triggers(obj, &mut regular)?;

        Ok(())
    }

    fn serialize_triggers(
        &mut self,
        obj: &Bound<'_, PyAny>,
        builder: &mut section_capnp::regular_section::Builder<'_>,
    ) -> Result<()> {
        let py = obj.py();
        let trigger_py = obj.getattr(intern!(py, "trigger"))?;
        let trigger_dict = trigger_py.cast::<PyDict>().map_err(PyErr::from)?;

        if trigger_dict.is_empty() {
            return Ok(());
        }

        let items: Vec<_> = trigger_dict.iter().collect();
        let mut triggers_builder = builder.reborrow().init_triggers(items.len() as u32);
        for (i, (signal_uid, trigger)) in items.iter().enumerate() {
            let signal_str: &str = signal_uid.extract()?;
            let signal_id = self.get_signal_index(signal_str)?;
            let state: u32 = trigger.get_item("state")?.extract()?;
            let mut tb = triggers_builder.reborrow().get(i as u32);
            tb.set_signal(signal_id);
            tb.set_state(state);
        }
        Ok(())
    }

    fn serialize_sweep_section(
        &mut self,
        obj: &Bound<'py, PyAny>,
        builder: &mut section_capnp::section::Builder<'_>,
    ) -> Result<()> {
        let py = obj.py();

        // Collect parameters
        let params_py = obj.getattr(intern!(py, "parameters"))?;
        let mut param_indices = Vec::new();
        for param in params_py.try_iter()? {
            let param = param?;
            let param_idx = self.collect_parameter(&param)?;
            param_indices.push(param_idx);
        }

        let reset_oscillator_phase = obj
            .getattr(intern!(py, "reset_oscillator_phase"))?
            .extract::<bool>()?;

        let chunk_count_py = obj.getattr(intern!(py, "chunk_count"))?;
        let chunk_count = extract_chunk_count(&chunk_count_py)?;

        let auto_chunking = obj
            .getattr(intern!(py, "auto_chunking"))?
            .extract::<bool>()?;

        let mut sweep = builder.reborrow().init_sweep();

        if let Some(alignment) = extract_alignment_capnp(obj)? {
            sweep.set_alignment(alignment);
        }

        if let Some(section_timing_mode) = extract_section_timing_mode_capnp(obj)? {
            sweep.set_section_timing_mode(section_timing_mode);
        }

        {
            let mut params_list = sweep.reborrow().init_parameters(param_indices.len() as u32);
            for (i, idx) in param_indices.iter().enumerate() {
                params_list.set(i as u32, *idx);
            }
        }
        sweep.set_reset_oscillator_phase(reset_oscillator_phase);

        sweep.reborrow().set_chunk_count(chunk_count);
        sweep.reborrow().set_auto_chunking(auto_chunking);

        Ok(())
    }

    fn collect_parameter(&mut self, obj: &Bound<'py, PyAny>) -> Result<u32> {
        let py = obj.py();
        // Keep the binding alive so we can borrow it as &str for the lookup,
        // avoiding a String allocation on the hot cache-hit path.
        let uid_binding = obj.getattr(intern!(py, "uid"))?;
        let uid_str: &str = uid_binding.extract()?;

        // Check for consistent parameter definitions with the same UID.
        if let Some(seen) = self.collected_sweep_parameters.get(uid_str) {
            if !seen.eq(obj).map_err(LabOneQError::from_err)? {
                bail!(
                    "Found multiple, inconsistent values for parameter '{}' with same UID.",
                    uid_str
                );
            }
        } else {
            self.collected_sweep_parameters
                .insert(uid_str.to_owned(), obj.clone());
        }

        // Return existing if already collected.
        if let Some(&idx) = self.entities.parameter_indices.get(uid_str) {
            return Ok(idx);
        }

        let idx = self.entities.parameters.len() as u32;
        // Only allocate owned strings on the miss path.
        self.entities
            .parameter_indices
            .insert(uid_str.to_owned(), idx);

        let linear_type = self.dsl_types.laboneq_type(DslType::LinearSweepParameter);
        let sweep_type = self.dsl_types.laboneq_type(DslType::SweepParameter);
        let axis_name: Option<String> = obj
            .getattr(intern!(py, "axis_name"))?
            .extract::<Option<String>>()?;

        if is_exact_type(obj, linear_type)? {
            let start = extract_py_numeric(&obj.getattr(intern!(py, "start"))?)?;
            let stop = extract_py_numeric(&obj.getattr(intern!(py, "stop"))?)?;
            let count: u32 = obj.getattr(intern!(py, "count"))?.extract()?;
            self.entities.parameters.push(CollectedParameter {
                alias: uid_str.to_owned(),
                axis_name,
                kind: SweepParameterKind::Linear { start, stop, count },
            });
        } else if is_exact_type(obj, sweep_type)? {
            let values_py = obj.getattr(intern!(py, "values"))?;
            let values = extract_explicit_values(&values_py, &self.np)?;
            // Push BEFORE calling register_driving_parameters: nested collect_parameter
            // calls inside it read entities.parameters.len() to assign their own index,
            // so we must claim our slot first or they will collide with idx.
            self.entities.parameters.push(CollectedParameter {
                alias: uid_str.to_owned(),
                axis_name,
                kind: SweepParameterKind::Explicit { values },
            });
            self.register_driving_parameters(idx, obj)?;
        } else {
            bail!("Unknown parameter type: {}", obj.get_type());
        };

        Ok(idx)
    }

    fn register_driving_parameters(&mut self, idx: u32, obj: &Bound<'_, PyAny>) -> Result<()> {
        let py = obj.py();
        let sweep_type = self.dsl_types.laboneq_type(DslType::SweepParameter);
        if !is_exact_type(obj, sweep_type)? {
            return Ok(());
        }
        let driven_by = obj.getattr(intern!(py, "driven_by"))?;
        if driven_by.is_none() {
            return Ok(());
        }
        // Collect the drivers into a Vec first so we can call &mut self methods
        // inside the loop without holding a borrow on the iterator.
        let drivers: Vec<Bound<'_, PyAny>> = driven_by.try_iter()?.collect::<PyResult<_>>()?;
        for driver in &drivers {
            // Collect the driver chain recursively.
            self.entities
                .derived_parameters
                .entry(idx)
                .or_default()
                .push(driver.getattr(intern!(py, "uid"))?.extract::<String>()?);
            self.register_driving_parameters(idx, driver)?;
        }
        Ok(())
    }

    fn serialize_acquire_loop_section(
        &self,
        obj: &Bound<'_, PyAny>,
        builder: &mut section_capnp::section::Builder<'_>,
    ) -> Result<()> {
        let py = obj.py();

        let count_py = obj.getattr(intern!(py, "count"))?;
        let count = if let Ok(c) = count_py.extract::<u64>() {
            if c == 0 || c > u32::MAX as u64 {
                bail!("Sweep 'count' must be a positive integer");
            }
            c
        } else {
            let c: f64 = count_py.extract()?;
            if c.fract() != 0.0 || c <= 0.0 || c > u32::MAX as f64 {
                bail!("Sweep 'count' must be a positive integer");
            }
            c as u64
        };

        let reset_oscillator_phase = obj
            .getattr(intern!(py, "reset_oscillator_phase"))?
            .extract::<bool>()?;

        let averaging_mode_py = obj.getattr(intern!(py, "averaging_mode"))?;
        let averaging_mode = if averaging_mode_py.is_none() {
            section_capnp::AveragingMode::Cyclic
        } else {
            match averaging_mode_py
                .getattr(intern!(py, "name"))?
                .extract::<&str>()?
            {
                "SEQUENTIAL" => section_capnp::AveragingMode::Sequential,
                "CYCLIC" => section_capnp::AveragingMode::Cyclic,
                "SINGLE_SHOT" => section_capnp::AveragingMode::SingleShot,
                name => {
                    bail!("Unknown averaging mode: {name}");
                }
            }
        };

        let acquisition_type_py = obj.getattr(intern!(py, "acquisition_type"))?;
        let acquisition_type = extract_acquisition_type_capnp(&acquisition_type_py)?;

        let repetition_mode_py = obj.getattr(intern!(py, "repetition_mode"))?;
        let repetition_time: Option<f64> =
            obj.getattr(intern!(py, "repetition_time"))?.extract()?;

        let mut acq_loop = builder.reborrow().init_acquire_loop();

        if let Some(alignment) = extract_alignment_capnp(obj)? {
            acq_loop.set_alignment(alignment);
        }

        if let Some(section_timing_mode) = extract_section_timing_mode_capnp(obj)? {
            acq_loop.set_section_timing_mode(section_timing_mode);
        }

        acq_loop.set_count(count);
        acq_loop.set_averaging_mode(averaging_mode);
        acq_loop.set_acquisition_type(acquisition_type);
        acq_loop.set_reset_oscillator_phase(reset_oscillator_phase);

        if !repetition_mode_py.is_none() {
            let mode_name_binding = repetition_mode_py.getattr(intern!(py, "name"))?;
            let mode_name: &str = mode_name_binding.extract()?;
            match mode_name {
                "FASTEST" => {
                    acq_loop.reborrow().init_repetition().set_fastest(());
                }
                "CONSTANT" => {
                    let t = repetition_time.ok_or_else(|| {
                        laboneq_error!("Repetition time required for CONSTANT mode")
                    })?;
                    acq_loop.reborrow().init_repetition().set_constant(t);
                }
                "AUTO" => {
                    acq_loop.reborrow().init_repetition().set_auto(());
                }
                name => {
                    bail!("Unknown repetition mode: {name}");
                }
            }
        }

        Ok(())
    }

    fn serialize_match_section(
        &mut self,
        obj: &Bound<'py, PyAny>,
        builder: &mut section_capnp::section::Builder<'_>,
    ) -> Result<()> {
        let py = obj.py();

        let handle_py = obj.getattr(intern!(py, "handle"))?;
        let user_register_py = obj.getattr(intern!(py, "user_register"))?;
        let sweep_parameter_py = obj.getattr(intern!(py, "sweep_parameter"))?;
        let prng_sample_py = obj.getattr(intern!(py, "prng_sample"))?;
        let variable_py = obj.getattr(intern!(py, "variable"))?;

        if [
            &handle_py,
            &user_register_py,
            &sweep_parameter_py,
            &prng_sample_py,
            &variable_py,
        ]
        .into_iter()
        .filter(|opt| !opt.is_none())
        .count()
            != 1
        {
            bail!(
                "Match must have exactly one of handle, user_register, sweep_parameter, prng_sample, or variable defined"
            );
        }

        let mut match_section = builder.reborrow().init_match();

        // play_after
        let play_after_names = collect_play_after_names(obj)?;
        if !play_after_names.is_empty() {
            let mut pa_builder = match_section
                .reborrow()
                .init_play_after(play_after_names.len() as u32);
            for (i, name) in play_after_names.iter().enumerate() {
                pa_builder.set(i as u32, name.as_str());
            }
        }

        if !handle_py.is_none() {
            let handle_str: &str = handle_py.extract()?;
            let handle_idx = self.entities.get_or_insert_handle(handle_str);
            match_section.set_handle(handle_idx);
        } else if !user_register_py.is_none() {
            let reg: u16 = user_register_py.extract()?;
            match_section.set_user_register(reg);
        } else if !sweep_parameter_py.is_none() {
            let param_idx = self.collect_parameter(&sweep_parameter_py)?;
            match_section.set_sweep_parameter(param_idx);
        } else if !prng_sample_py.is_none() {
            let uid_binding = prng_sample_py.getattr(intern!(py, "uid"))?;
            let sample_uid: &str = uid_binding.extract()?;
            match_section.set_prng_sample(sample_uid);
        } else if !variable_py.is_none() {
            match_section.set_variable(self.hqcs.get_or_insert_variable(&variable_py));
        }

        // local
        let local_py = obj.getattr(intern!(py, "local"))?;
        if let Ok(Some(local_val)) = local_py.extract::<Option<bool>>() {
            match_section.reborrow().init_local().set_value(local_val);
        }

        Ok(())
    }

    fn serialize_case_section(
        &self,
        obj: &Bound<'_, PyAny>,
        builder: &mut section_capnp::section::Builder<'_>,
    ) -> Result<()> {
        let py = obj.py();
        let state_py = obj.getattr(intern!(py, "state"))?;

        let mut case_section = builder.reborrow().init_case_section();
        if let Some(section_timing_mode) = extract_section_timing_mode_capnp(obj)? {
            case_section.set_section_timing_mode(section_timing_mode);
        }
        let mut state_builder = case_section.init_state();
        set_sweep_value_from_py(&state_py, &mut state_builder)?;

        Ok(())
    }

    fn serialize_prng_setup_section(
        &mut self,
        obj: &Bound<'_, PyAny>,
        builder: &mut section_capnp::section::Builder<'_>,
    ) -> Result<()> {
        let py = obj.py();
        let prng_py = obj.getattr(intern!(py, "prng"))?;
        let range: u32 = prng_py.getattr(intern!(py, "range"))?.extract()?;
        let seed: u32 = prng_py.getattr(intern!(py, "seed"))?.extract()?;

        let mut setup = builder.reborrow().init_prng_setup();
        setup.set_range(range);
        setup.set_seed(seed);

        Ok(())
    }

    fn serialize_prng_loop_section(
        &mut self,
        obj: &Bound<'_, PyAny>,
        builder: &mut section_capnp::section::Builder<'_>,
    ) -> Result<()> {
        let py = obj.py();
        let prng_sample_py = obj.getattr(intern!(py, "prng_sample"))?;
        let uid_binding = prng_sample_py.getattr(intern!(py, "uid"))?;
        let sample_uid: &str = uid_binding.extract()?;
        let count: u32 = prng_sample_py.getattr(intern!(py, "count"))?.extract()?;

        let mut loop_section = builder.reborrow().init_prng_loop();
        loop_section.set_prng_sample(sample_uid);
        loop_section.set_count(count);

        Ok(())
    }

    // === Operation serialization ===

    fn serialize_operation(
        &mut self,
        obj: &Bound<'py, PyAny>,
        mut builder: operation_capnp::operation::Builder<'_>,
    ) -> Result<()> {
        let ty = obj.get_type();

        if ty.is(self.dsl_types.laboneq_type(DslType::PlayPulse)) {
            self.serialize_play_op(obj, &mut builder)?;
        } else if ty.is(self.dsl_types.laboneq_type(DslType::Delay)) {
            self.serialize_delay_op(obj, &mut builder)?;
        } else if ty.is(self.dsl_types.laboneq_type(DslType::Reserve)) {
            self.serialize_reserve_op(obj, &mut builder)?;
        } else if ty.is(self.dsl_types.laboneq_type(DslType::Acquire)) {
            self.serialize_acquire_op(obj, &mut builder)?;
        } else if ty.is(self.dsl_types.laboneq_type(DslType::Call)) {
            self.serialize_call_op(obj, &mut builder)?;
        } else if ty.is(self.dsl_types.laboneq_type(DslType::SetNode)) {
            self.serialize_set_node_op(obj, &mut builder)?;
        } else if ty.is(self.dsl_types.laboneq_type(DslType::ResetOscillatorPhase)) {
            self.serialize_reset_oscillator_phase_op(obj, &mut builder)?;
        } else if ty.is(self.dsl_types.laboneq_type(DslType::HqcsSend)) {
            self.serialize_send_op(obj, &mut builder)?;
        } else if ty.is(self.dsl_types.laboneq_type(DslType::HqcsMarkStale)) {
            self.serialize_mark_stale_op(obj, &mut builder)?;
        } else if ty.is(self.dsl_types.laboneq_type(DslType::HqcsEnsure)) {
            self.serialize_ensure_op(obj, &mut builder)?;
        } else {
            bail!("Unknown operation type: {}", obj.get_type());
        }

        Ok(())
    }

    fn serialize_set_node_op(
        &mut self,
        obj: &Bound<'py, PyAny>,
        builder: &mut operation_capnp::operation::Builder<'_>,
    ) -> Result<()> {
        let py = obj.py();
        let mut set_node = builder.reborrow().init_set_node();

        // path
        let path_py = obj.getattr(intern!(py, "path"))?;
        if !path_py.is_none() {
            let path: &str = path_py.extract()?;
            set_node.set_path(path);
        }

        // value
        let value_py = obj.getattr(intern!(py, "value"))?;
        if !value_py.is_none() {
            self.set_value_from_py(
                &value_py,
                &mut set_node
                    .reborrow()
                    .get_value()
                    .map_err(LabOneQError::from_err)?,
            )?;
        }

        Ok(())
    }

    fn serialize_play_op(
        &mut self,
        obj: &Bound<'py, PyAny>,
        builder: &mut operation_capnp::operation::Builder<'_>,
    ) -> Result<()> {
        let py = obj.py();
        let mut play = builder.reborrow().init_play();

        // Signal
        let signal_obj = obj.getattr(intern!(py, "signal"))?;
        let signal_str: &str = signal_obj.extract()?;
        let signal_id = self.get_signal_index(signal_str)?;
        play.set_signal(signal_id);

        // Pulse
        let pulse_py = obj.getattr(intern!(py, "pulse"))?;
        if !pulse_py.is_none() {
            let pulse_idx = self.collect_pulse(&pulse_py)?;
            play.set_pulse(pulse_idx);
        }

        // Amplitude
        let amplitude_py = obj.getattr(intern!(py, "amplitude"))?;
        if !amplitude_py.is_none() {
            self.set_value_from_py(
                &amplitude_py,
                &mut play
                    .reborrow()
                    .get_amplitude()
                    .map_err(LabOneQError::from_err)?,
            )?;
        }

        // Phase
        let phase_py = obj.getattr(intern!(py, "phase"))?;
        if !phase_py.is_none() {
            self.set_value_from_py(
                &phase_py,
                &mut play
                    .reborrow()
                    .get_phase()
                    .map_err(LabOneQError::from_err)?,
            )?;
        }

        // Increment oscillator phase
        let inc_osc_py = obj.getattr(intern!(py, "increment_oscillator_phase"))?;
        if !inc_osc_py.is_none() {
            self.set_value_from_py(
                &inc_osc_py,
                &mut play
                    .reborrow()
                    .get_increment_oscillator_phase()
                    .map_err(LabOneQError::from_err)?,
            )?;
        }

        // Set oscillator phase
        let set_osc_py = obj.getattr(intern!(py, "set_oscillator_phase"))?;
        if !set_osc_py.is_none() {
            self.set_value_from_py(
                &set_osc_py,
                &mut play
                    .reborrow()
                    .get_set_oscillator_phase()
                    .map_err(LabOneQError::from_err)?,
            )?;
        }

        // Length
        let length_py = obj.getattr(intern!(py, "length"))?;
        if !length_py.is_none() {
            self.set_value_from_py(
                &length_py,
                &mut play
                    .reborrow()
                    .get_length()
                    .map_err(LabOneQError::from_err)?,
            )?;
        }

        // Pulse parameters
        let pulse_params_py = obj.getattr(intern!(py, "pulse_parameters"))?;
        if !pulse_params_py.is_none() {
            self.serialize_pulse_parameters(&pulse_params_py, play.reborrow())?;
        }

        // Markers
        let marker_py = obj.getattr(intern!(py, "marker"))?;
        if !marker_py.is_none() {
            self.serialize_markers(&marker_py, &mut play)?;
        }

        Ok(())
    }

    fn serialize_pulse_parameters(
        &mut self,
        obj: &Bound<'py, PyAny>,
        play: operation_capnp::play_op::Builder<'_>,
    ) -> Result<()> {
        let dict = obj.cast::<PyDict>().map_err(PyErr::from)?;
        if dict.is_empty() {
            return Ok(());
        }

        let items: Vec<_> = dict.iter().collect();
        let mut entries = play.init_pulse_parameters(items.len() as u32);
        for (i, (key, value)) in items.iter().enumerate() {
            let mut entry = entries.reborrow().get(i as u32);
            let key_str: &str = key.extract()?;
            entry.set_key(key_str);
            let mut val_builder = entry.init_value();
            self.set_value_entry(value, &mut val_builder)
                .with_context(|| format!("pulse parameter '{key_str}'"))?;
        }
        Ok(())
    }

    fn serialize_markers(
        &mut self,
        obj: &Bound<'py, PyAny>,
        play: &mut operation_capnp::play_op::Builder<'_>,
    ) -> Result<()> {
        let py = obj.py();
        let dict = obj.cast::<PyDict>().map_err(PyErr::from)?;
        let mut markers = play.reborrow().init_markers();

        for (key, value) in dict.iter() {
            let marker_name: &str = key.extract()?;
            let marker_dict = value.cast::<PyDict>().map_err(PyErr::from)?;

            let enable = marker_dict
                .get_item(intern!(py, "enable"))?
                .map(|o| o.extract::<bool>())
                .transpose()?
                .unwrap_or(false);

            let start = marker_dict
                .get_item(intern!(py, "start"))?
                .map(|o| o.extract::<f64>())
                .transpose()?;

            let length = marker_dict
                .get_item(intern!(py, "length"))?
                .map(|o| o.extract::<f64>())
                .transpose()?;

            // Resolve the waveform pulse index before the closure so the closure
            // doesn't need to borrow `self`.
            let waveform_py = marker_dict.get_item(intern!(py, "waveform"))?;
            let waveform_idx = waveform_py
                .as_ref()
                .filter(|w| !w.is_none())
                .map(|w| self.collect_pulse(w))
                .transpose()?;

            let set_marker = |mut spec: operation_capnp::marker_spec::Builder<'_>| -> Result<()> {
                spec.set_enable(enable);
                if let Some(s) = start {
                    spec.reborrow().init_start().set_value(s);
                }
                if let Some(l) = length {
                    spec.reborrow().init_length().set_value(l);
                }
                if let Some(idx) = waveform_idx {
                    spec.set_waveform(idx);
                }
                Ok(())
            };

            match marker_name {
                "marker1" => set_marker(
                    markers
                        .reborrow()
                        .get_marker1()
                        .map_err(LabOneQError::from_err)?,
                )?,
                "marker2" => set_marker(
                    markers
                        .reborrow()
                        .get_marker2()
                        .map_err(LabOneQError::from_err)?,
                )?,
                _ => return Err(laboneq_error!("Unknown marker: {marker_name}")),
            }
        }
        Ok(())
    }

    fn serialize_delay_op(
        &mut self,
        obj: &Bound<'py, PyAny>,
        builder: &mut operation_capnp::operation::Builder<'_>,
    ) -> Result<()> {
        let py = obj.py();
        let mut delay = builder.reborrow().init_delay();

        let signal_obj = obj.getattr(intern!(py, "signal"))?;
        let signal_str: &str = signal_obj.extract()?;
        let signal_id = self.get_signal_index(signal_str)?;
        delay.set_signal(signal_id);

        let time_py = obj.getattr(intern!(py, "time"))?;
        self.set_value_from_py(
            &time_py,
            &mut delay
                .reborrow()
                .get_time()
                .map_err(LabOneQError::from_err)?,
        )?;

        let precomp_clear = obj
            .getattr(intern!(py, "precompensation_clear"))?
            .extract::<Option<bool>>()?
            .unwrap_or(false);
        delay.set_precompensation_clear(precomp_clear);

        Ok(())
    }

    fn serialize_reserve_op(
        &mut self,
        obj: &Bound<'py, PyAny>,
        builder: &mut operation_capnp::operation::Builder<'_>,
    ) -> Result<()> {
        let py = obj.py();
        let mut reserve = builder.reborrow().init_reserve();
        let signal_obj = obj.getattr(intern!(py, "signal"))?;
        let signal_str: &str = signal_obj.extract()?;
        let signal_id = self.get_signal_index(signal_str)?;
        reserve.set_signal(signal_id);
        Ok(())
    }

    fn serialize_acquire_op(
        &mut self,
        obj: &Bound<'py, PyAny>,
        builder: &mut operation_capnp::operation::Builder<'_>,
    ) -> Result<()> {
        let py = obj.py();
        let mut acquire = builder.reborrow().init_acquire();

        let signal_obj = obj.getattr(intern!(py, "signal"))?;
        let signal_str: &str = signal_obj.extract()?;
        let signal_id = self.get_signal_index(signal_str)?;
        acquire.set_signal(signal_id);

        let handle_obj = obj.getattr(intern!(py, "handle"))?;
        let handle_str: &str = handle_obj
            .extract()
            .map_err(|_| laboneq_error!("Invalid type for field 'handle'"))?;
        let handle_idx = self.entities.get_or_insert_handle(handle_str);
        acquire.set_handle(handle_idx);

        // Kernels
        let kernel_py = obj.getattr(intern!(py, "kernel"))?;
        let mut kernel_indices = Vec::new();
        if !kernel_py.is_none() {
            if kernel_py.is_instance(&py.get_type::<PyList>())? {
                for k in kernel_py.try_iter()? {
                    let k = k?;
                    let idx = self.collect_pulse(&k)?;
                    kernel_indices.push(idx);
                }
            } else {
                let idx = self.collect_pulse(&kernel_py)?;
                kernel_indices.push(idx);
            }
        }
        if !kernel_indices.is_empty() {
            let mut kernels = acquire.reborrow().init_kernels(kernel_indices.len() as u32);
            for (i, idx) in kernel_indices.iter().enumerate() {
                kernels.set(i as u32, *idx);
            }
        }

        // Length
        let length_py = obj.getattr(intern!(py, "length"))?;
        if let Some(length) = length_py.extract::<Option<f64>>()? {
            acquire
                .reborrow()
                .get_length()
                .map_err(LabOneQError::from_err)?
                .init_constant()
                .set_real(length);
        }

        // Per-operation kernel parameter overrides (Acquire.pulse_parameters DSL attribute).
        let pulse_params_py = obj.getattr(intern!(py, "pulse_parameters"))?;
        if !pulse_params_py.is_none() {
            let per_kernel: Vec<Bound<'_, PyAny>> =
                if pulse_params_py.is_instance(&py.get_type::<PyList>())? {
                    pulse_params_py.try_iter()?.collect::<PyResult<_>>()?
                } else {
                    vec![pulse_params_py]
                };
            if !per_kernel.is_empty() {
                let mut kp_builder = acquire
                    .reborrow()
                    .init_kernel_parameters(per_kernel.len() as u32);
                for (i, param_dict) in per_kernel.iter().enumerate() {
                    if param_dict.is_none() {
                        continue;
                    }
                    let dict = param_dict.cast::<PyDict>().map_err(PyErr::from)?;
                    if dict.is_empty() {
                        continue;
                    }
                    let items: Vec<_> = dict.iter().collect();
                    let param_map = kp_builder.reborrow().get(i as u32);
                    let mut entries = param_map.init_parameters(items.len() as u32);
                    for (j, (key, value)) in items.iter().enumerate() {
                        let mut entry = entries.reborrow().get(j as u32);
                        let key_str: &str = key.extract()?;
                        entry.set_key(key_str);
                        let mut val_builder = entry.init_value();
                        self.set_value_entry(value, &mut val_builder)
                            .with_context(|| format!("pulse parameter '{key_str}'"))?;
                    }
                }
            }
        }

        Ok(())
    }

    fn serialize_reset_oscillator_phase_op(
        &mut self,
        obj: &Bound<'py, PyAny>,
        builder: &mut operation_capnp::operation::Builder<'_>,
    ) -> Result<()> {
        let py = obj.py();
        let mut reset = builder.reborrow().init_reset_oscillator_phase();

        let signal_py = obj.getattr(intern!(py, "signal"))?;
        if let Ok(Some(signal_str)) = signal_py.extract::<Option<&str>>() {
            let signal_id = self.get_signal_index(signal_str)?;
            reset.set_signal(signal_id);
        }
        // When `signal` is None we intentionally skip calling `set_signal`.
        // The schema declares `signal @0 :Common.Id = .Common.noneId` (default = 0xffffffff),
        // so an unset field reads back as `noneId`, which the backend interprets as "reset all".

        Ok(())
    }

    fn serialize_call_op(
        &mut self,
        obj: &Bound<'py, PyAny>,
        builder: &mut operation_capnp::operation::Builder<'_>,
    ) -> Result<()> {
        let py = obj.py();
        let mut call = builder.reborrow().init_call();

        let func_name_py = obj.getattr(intern!(py, "func_name"))?;
        let func_name: &str = func_name_py.extract()?;
        call.set_callback_id(func_name);

        let args_py = obj.getattr(intern!(py, "args"))?;
        let dict = args_py.cast::<PyDict>().map_err(PyErr::from)?;
        if dict.is_empty() {
            return Ok(());
        }

        let items: Vec<_> = dict.iter().collect();
        let mut entries = call.init_arguments(items.len() as u32);
        for (i, (key, value)) in items.iter().enumerate() {
            let mut entry = entries.reborrow().get(i as u32);
            let key_str: &str = key.extract()?;
            entry.set_key(key_str);

            let mut val_builder = entry.init_value();
            self.set_value_entry(value, &mut val_builder)
                .with_context(|| format!("near-time callback argument '{key_str}'"))?;
        }
        Ok(())
    }

    // === Pulse collection ===

    fn collect_pulse(&mut self, obj: &Bound<'py, PyAny>) -> Result<u32> {
        let py = obj.py();
        // Keep the binding alive so we can borrow it as &str for the lookup,
        // avoiding a String allocation on the hot cache-hit path.
        let uid_binding = obj.getattr(intern!(py, "uid"))?;
        let uid_str: &str = uid_binding.extract()?;

        if let Some(&idx) = self.entities.pulse_indices.get(uid_str) {
            return Ok(idx);
        }

        let idx = self.entities.pulses.len() as u32;
        // Only allocate on the miss path.
        self.entities.pulse_indices.insert(uid_str.to_owned(), idx);

        let can_compress: bool = obj.getattr(intern!(py, "can_compress"))?.extract()?;

        let is_functional =
            obj.is_instance(self.dsl_types.laboneq_type(DslType::PulseFunctional))?;

        let (amplitude_re, amplitude_im) = if is_functional {
            let amp_py = obj.getattr(intern!(py, "amplitude"))?;
            if amp_py.is_none() {
                (1.0, 0.0)
            } else {
                match extract_py_numeric(&amp_py)? {
                    NumericValue::Real(v) => (v, 0.0),
                    NumericValue::Int(v) => (v as f64, 0.0),
                    NumericValue::Complex(re, im) => (re, im),
                }
            }
        } else {
            (1.0, 0.0)
        };

        let length = if is_functional {
            Some(obj.getattr(intern!(py, "length"))?.extract::<f64>()?)
        } else {
            None
        };

        let (shape, functional_params) = if is_functional {
            let function: String = obj.getattr(intern!(py, "function"))?.extract()?;
            // Collect definition-level pulse parameters, resolving parameter refs to final indices.
            let pulse_params_py = obj.getattr(intern!(py, "pulse_parameters"))?;
            let params = if !pulse_params_py.is_none() {
                let dict = pulse_params_py.cast::<PyDict>().map_err(PyErr::from)?;
                let mut entries = Vec::with_capacity(dict.len());
                for (key, value) in dict.iter() {
                    let key_str: String = key.extract()?;
                    let pv =
                        if value.is_instance(self.dsl_types.laboneq_type(DslType::Parameter))? {
                            let param_idx = self.collect_parameter(&value)?;
                            PulseParamValue::ParameterRef(param_idx)
                        } else if value.is_instance_of::<PyBool>() {
                            // `bool` subclasses `int`; keep it out of the `Int` branch below so
                            // it round-trips as a bool rather than as `0`/`1`.
                            let bytes = constant_serializer::serialize_json(&value)
                                .map_err(LabOneQError::from)
                                .with_context(|| format!("pulse parameter '{key_str}'"))?;
                            PulseParamValue::Json(bytes)
                        } else if let Ok(v) = value.extract::<i64>() {
                            PulseParamValue::Int(v)
                        } else if value.is_instance_of::<PyComplex>() {
                            let c: num_complex::Complex64 = value.extract()?;
                            PulseParamValue::Complex(c.re, c.im)
                        } else if let Ok(v) = value.extract::<f64>() {
                            PulseParamValue::Real(v)
                        } else if let Ok(raw) = value.cast::<PyBytes>() {
                            PulseParamValue::RawBytes(raw.as_bytes().to_vec())
                        } else {
                            let bytes = constant_serializer::serialize_json(&value)
                                .map_err(LabOneQError::from)
                                .with_context(|| format!("pulse parameter '{key_str}'"))?;
                            PulseParamValue::Json(bytes)
                        };
                    entries.push(PulseParamEntry {
                        key: key_str,
                        value: pv,
                    });
                }
                entries
            } else {
                vec![]
            };
            (PulseShape::Functional { function }, params)
        } else {
            let samples_py = obj.getattr(intern!(py, "samples"))?;
            let arr = self
                .np
                .call_method1(intern!(py, "asarray"), (&samples_py,))?;
            let arr_kind_binding = arr
                .getattr(intern!(py, "dtype"))?
                .getattr(intern!(py, "kind"))?;
            let arr_kind: &str = arr_kind_binding.extract()?;
            let mut is_complex = arr_kind == "c";

            let bytes_arr = if is_complex {
                arr.call_method1(intern!(py, "astype"), (intern!(py, "complex128"),))?
                    .call_method0(intern!(py, "tobytes"))?
            } else {
                let ndim: usize = arr.getattr(intern!(py, "ndim"))?.extract()?;
                if ndim > 1 {
                    is_complex = true;
                    crate::py_helpers::iq_to_complex(&self.np, &arr)?
                        .call_method0(intern!(py, "tobytes"))?
                } else {
                    arr.call_method1(intern!(py, "astype"), (intern!(py, "float64"),))?
                        .call_method0(intern!(py, "tobytes"))?
                }
            };
            let samples: Vec<u8> = bytes_arr.extract()?;
            (
                PulseShape::Sampled {
                    samples,
                    is_complex,
                },
                vec![],
            )
        };

        self.entities.pulses.push(CollectedPulse {
            alias: uid_str.to_owned(),
            can_compress,
            amplitude_re,
            amplitude_im,
            length,
            shape,
            functional_params,
        });

        Ok(idx)
    }

    // === Sweep value helpers ===

    fn set_value_entry(
        &mut self,
        obj: &Bound<'py, PyAny>,
        builder: &mut common_capnp::value::Builder<'_>,
    ) -> Result<()> {
        // `bool` is a subclass of `int` in Python, so it would otherwise be serialized as an
        // integer constant and reach the user as `0`/`1`. Route it to the opaque path instead,
        // which round-trips it losslessly as JSON.
        if obj.is_instance_of::<PyBool>() {
            return self.set_external_opaque_constant(obj, builder);
        }
        match self.set_value_from_py(obj, builder) {
            Ok(()) => Ok(()),
            Err(_) => self.set_external_opaque_constant(obj, builder),
        }
    }

    fn set_external_opaque_constant(
        &self,
        obj: &Bound<'py, PyAny>,
        builder: &mut common_capnp::value::Builder<'_>,
    ) -> Result<()> {
        if let Ok(raw) = obj.cast::<PyBytes>() {
            // Plain bytes — pass through without serialization.
            builder
                .reborrow()
                .init_constant()
                .set_raw_bytes_value(raw.as_bytes());
            return Ok(());
        }
        let bytes = constant_serializer::serialize_json(obj)?;
        builder.reborrow().init_constant().set_python_value(&bytes);
        Ok(())
    }

    fn set_value_from_py(
        &mut self,
        obj: &Bound<'py, PyAny>,
        builder: &mut common_capnp::value::Builder<'_>,
    ) -> Result<()> {
        if obj.is_none() {
            return Ok(());
        }
        // Check if it's a Parameter reference.
        if obj.is_instance(self.dsl_types.laboneq_type(DslType::Parameter))? {
            let param_idx = self.collect_parameter(obj)?;
            builder.set_parameter_ref(param_idx);
            return Ok(());
        }
        set_sweep_value_from_py(obj, builder)
    }

    /// Tags `DeviceSetup` with the target backend and forwards the backend's own
    /// setup-description pointer into the opaque `setupDescription` field.
    fn serialize_device_setup(
        &mut self,
        backend: &str,
        setup_description: any_pointer::Reader<'_>,
        mut builder: experiment_capnp::experiment::Builder<'_>,
    ) -> Result<()> {
        let mut device_setup_builder = builder.reborrow().init_device_setup();
        device_setup_builder.set_backend(backend);
        device_setup_builder
            .init_setup_description()
            .set_as::<any_pointer::Owned>(setup_description)
            .map_err(|e| laboneq_error!("Failed to copy the setup description: {e}"))
    }

    fn serialize_oscillator(
        &mut self,
        oscillator: &OscillatorPy<'py>,
        builder: calibration_capnp::signal_calibration::Builder<'_>,
    ) -> Result<()> {
        let mut osc_builder = builder.init_oscillator();
        osc_builder.set_uid(oscillator.uid.to_str()?);
        self.set_value_from_py(
            &oscillator.frequency,
            &mut osc_builder
                .reborrow()
                .get_frequency()
                .map_err(LabOneQError::from_err)?,
        )?;
        let modulation_type = if let Some(modulation) = &oscillator.modulation {
            match modulation.extract::<&str>()? {
                "AUTO" => calibration_capnp::ModulationType::Auto,
                "HARDWARE" => calibration_capnp::ModulationType::Hardware,
                "SOFTWARE" => calibration_capnp::ModulationType::Software,
                other => {
                    bail!(
                        "Invalid modulation type: {}. Expected 'AUTO', 'HARDWARE', or 'SOFTWARE'.",
                        other
                    );
                }
            }
        } else {
            calibration_capnp::ModulationType::Auto
        };
        osc_builder.set_modulation_type(modulation_type);
        Ok(())
    }

    fn serialize_signal_calibration(
        &mut self,
        signal: &ExperimentSignalPy<'py>,
        mut builder: experiment_capnp::experiment_signal::Builder<'_>,
    ) -> Result<()> {
        let mut calibration_builder = builder.reborrow().init_calibration();
        if let Some(oscillator) = &signal.oscillator {
            self.serialize_oscillator(oscillator, calibration_builder.reborrow())?;
        }

        calibration_builder.set_delay_signal(signal.delay_signal);
        calibration_builder.set_automute(signal.automute);

        if let Some(amplitude) = &signal.amplitude {
            self.set_value_from_py(
                amplitude,
                &mut calibration_builder
                    .reborrow()
                    .get_amplitude()
                    .map_err(LabOneQError::from_err)?,
            )?;
        }

        if let Some(lo_freq) = &signal.lo_frequency {
            self.set_value_from_py(
                lo_freq,
                &mut calibration_builder
                    .reborrow()
                    .get_local_oscillator_frequency()
                    .map_err(LabOneQError::from_err)?,
            )?;
        }

        if let Some(port_delay) = &signal.port_delay {
            self.set_value_from_py(
                port_delay,
                &mut calibration_builder
                    .reborrow()
                    .get_port_delay()
                    .map_err(LabOneQError::from_err)?,
            )?;
        }

        if let Some(voltage_offset) = &signal.voltage_offset {
            self.set_value_from_py(
                voltage_offset,
                &mut calibration_builder
                    .reborrow()
                    .get_voltage_offset()
                    .map_err(LabOneQError::from_err)?,
            )?;
        }

        if let Some(port_mode) = &signal.port_mode {
            match port_mode.extract::<&str>()? {
                "RF" => calibration_builder.set_port_mode(calibration_capnp::PortMode::Rf),
                "LF" => calibration_builder.set_port_mode(calibration_capnp::PortMode::Lf),
                other => {
                    bail!("Invalid port mode: {}. Expected 'RF' or 'LF'.", other);
                }
            }
        } else {
            calibration_builder.set_port_mode(calibration_capnp::PortMode::Unspecified);
        }

        // Precompensation
        if let Some(precompensation) = &signal.precompensation {
            let mut precomp_builder = calibration_builder.reborrow().init_precompensation();

            // Exponentials
            if let Some(exponentials) = &precompensation.exponential {
                let mut exponential_builder = precomp_builder
                    .reborrow()
                    .init_exponentials(exponentials.len() as u32);

                for (i, exp) in exponentials.iter().enumerate() {
                    let mut exp_builder = exponential_builder.reborrow().get(i as u32);
                    exp_builder.set_amplitude(exp.amplitude);
                    exp_builder.set_timeconstant(exp.timeconstant);
                }
            }

            // High-pass
            if let Some(high_pass) = &precompensation.high_pass {
                precomp_builder
                    .reborrow()
                    .init_high_pass()
                    .set_timeconstant(high_pass.timeconstant);
            }

            // Bounce
            if let Some(bounce) = &precompensation.bounce {
                let mut bounce_builder = precomp_builder.reborrow().init_bounce();
                bounce_builder.set_amplitude(bounce.amplitude);
                bounce_builder.set_delay(bounce.delay);
            }

            // FIR
            if let Some(fir) = &precompensation.fir {
                let mut fir_builder = precomp_builder.reborrow().init_fir();
                let mut coeff_builder = fir_builder
                    .reborrow()
                    .init_coefficients(fir.coefficients.len() as u32);
                for (i, coeff) in fir.coefficients.iter().enumerate() {
                    coeff_builder.set(i as u32, *coeff);
                }
            }
        }

        // Amplifier pump
        if let Some(amplifier_pump) = &signal.amplifier_pump {
            let mut pump_builder = calibration_builder.reborrow().init_amplifier_pump();
            pump_builder.set_alc_on(amplifier_pump.alc_on);
            pump_builder.set_pump_on(amplifier_pump.pump_on);
            pump_builder.set_pump_filter_on(amplifier_pump.pump_filter_on);
            pump_builder.set_probe_on(amplifier_pump.probe_on);
            pump_builder.set_cancellation_on(amplifier_pump.cancellation_on);
            pump_builder.set_cancellation_source(match amplifier_pump.cancellation_source {
                CancellationSourcePy::Internal => calibration_capnp::CancellationSource::Internal,
                CancellationSourcePy::External => calibration_capnp::CancellationSource::External,
            });

            if let Some(frequency) = amplifier_pump.cancellation_source_frequency {
                pump_builder
                    .reborrow()
                    .get_cancellation_source_frequency()
                    .set_value(frequency);
            }

            self.set_value_from_py(
                &amplifier_pump.pump_power,
                &mut pump_builder
                    .reborrow()
                    .get_pump_power()
                    .map_err(LabOneQError::from_err)?,
            )?;

            self.set_value_from_py(
                &amplifier_pump.pump_frequency,
                &mut pump_builder
                    .reborrow()
                    .get_pump_frequency()
                    .map_err(LabOneQError::from_err)?,
            )?;

            self.set_value_from_py(
                &amplifier_pump.probe_power,
                &mut pump_builder
                    .reborrow()
                    .get_probe_power()
                    .map_err(LabOneQError::from_err)?,
            )?;

            self.set_value_from_py(
                &amplifier_pump.probe_frequency,
                &mut pump_builder
                    .reborrow()
                    .get_probe_frequency()
                    .map_err(LabOneQError::from_err)?,
            )?;

            self.set_value_from_py(
                &amplifier_pump.cancellation_phase,
                &mut pump_builder
                    .reborrow()
                    .get_cancellation_phase()
                    .map_err(LabOneQError::from_err)?,
            )?;

            self.set_value_from_py(
                &amplifier_pump.cancellation_attenuation,
                &mut pump_builder
                    .reborrow()
                    .get_cancellation_attenuation()
                    .map_err(LabOneQError::from_err)?,
            )?;
        }

        // Range
        if let Some(quantity) = &signal.range {
            let mut range_builder = calibration_builder.reborrow().init_range();
            range_builder.set_value(quantity.value);
            if let Some(unit) = &quantity.unit {
                match unit {
                    UnitPy::Volt => range_builder.set_unit("volt"),
                    UnitPy::DBm => range_builder.set_unit("dbm"),
                }
            }
        }

        // Added outputs
        let mut added_outputs_builder = calibration_builder
            .reborrow()
            .init_added_outputs(signal.added_outputs.len() as u32);
        for (i, output) in signal.added_outputs.iter().enumerate() {
            let mut output_builder = added_outputs_builder.reborrow().get(i as u32);

            output_builder.set_source_signal(output.source.to_str()?);
            self.set_value_from_py(
                &output.amplitude_scaling,
                &mut output_builder
                    .reborrow()
                    .get_amplitude_scaling()
                    .map_err(LabOneQError::from_err)?,
            )?;
            self.set_value_from_py(
                &output.phase_shift,
                &mut output_builder
                    .reborrow()
                    .get_phase_shift()
                    .map_err(LabOneQError::from_err)?,
            )?;
        }

        // Discrimination threshold(s)
        if let Some(threshold) = &signal.threshold {
            let mut threshold_builder = calibration_builder
                .reborrow()
                .init_threshold(threshold.len() as u32);
            for (i, value) in threshold.iter().enumerate() {
                threshold_builder.set(i as u32, *value);
            }
        }

        // Mixer calibration
        if let Some(mixer_calibration) = &signal.mixer_calibration {
            let mut mixer_builder = calibration_builder.reborrow().init_mixer_calibration();

            // Voltage offsets for I and Q. We use the presence of each entry to determine whether to set it,
            if let Some(voltage_offsets) = &mixer_calibration.voltage_offsets {
                if let Some(value) = voltage_offsets.first() {
                    self.set_value_from_py(
                        value,
                        &mut mixer_builder
                            .reborrow()
                            .get_voltage_offset_i()
                            .map_err(LabOneQError::from_err)?,
                    )?;
                }
                if let Some(value) = voltage_offsets.get(1) {
                    self.set_value_from_py(
                        value,
                        &mut mixer_builder
                            .reborrow()
                            .get_voltage_offset_q()
                            .map_err(LabOneQError::from_err)?,
                    )?;
                }
            }

            // Correction matrix
            if let Some(correction_matrix) = &mixer_calibration.correction_matrix
                && !correction_matrix.is_empty()
            {
                if correction_matrix.len() != 2
                    || correction_matrix.iter().any(|row| row.len() != 2)
                {
                    bail!("Correction matrix must be 2x2 for I/Q mixer calibration");
                }

                // Correction matrix (2x2). We flatten the nested Vec<Vec<T>> into a single row-major Vec<T> for capnp.
                let mut correction_matrix_builder =
                    mixer_builder.reborrow().init_correction_matrix(4);
                for (flat_index, correction) in correction_matrix
                    .iter()
                    .flat_map(|row| row.iter())
                    .enumerate()
                {
                    let mut entry_builder =
                        correction_matrix_builder.reborrow().get(flat_index as u32);
                    self.set_value_from_py(correction, &mut entry_builder)?;
                }
            }
        }
        Ok(())
    }
}

// === Public entry point ===

/// Serializes a Python experiment object tree to Cap'n Proto bytes.
///
/// `setup_description` is the root of the target backend's own setup-description message,
/// stored as-is in the opaque `DeviceSetup.setupDescription`; `backend` names its schema.
///
/// Returns the serialized bytes as a `Vec<u8>`.
#[instrument(
    level = "debug",
    name = "laboneq.compiler.serialize_experiment",
    skip_all
)]
pub(crate) fn serialize_experiment(
    py: Python,
    experiment: ExperimentCapnpPy,
    backend: &str,
    setup_description: any_pointer::Reader<'_>,
    packed: bool,
) -> Result<Vec<u8>> {
    let mut ser = Serializer::new(py)?;

    let mut message = capnp::message::Builder::new_default();
    let mut exp_builder = message.init_root::<experiment_capnp::experiment::Builder<'_>>();

    ser.serialize_device_setup(backend, setup_description, exp_builder.reborrow())?;

    // Signals are indexed first (sorted alphabetically for determinism) so that
    // signal references written during section tree traversal use final indices.
    ser.serialize_signals(&experiment, exp_builder.reborrow())?;

    // HQCS entities that sections reference must be indexed before the
    // section traversal.
    ser.serialize_hqcs_coprocessors(&experiment, exp_builder.reborrow())?;
    ser.serialize_hqcs_streams(&experiment, exp_builder.reborrow())?;

    // Traverse the section tree. All entity collections (pulses, parameters,
    // handles) accumulate into `ser.entities` with final zero-based indices
    // assigned at first insertion.
    ser.serialize_root_sections(&experiment, exp_builder.reborrow())?;

    // Write entity definition lists. Indices used for cross-references within
    // definitions (e.g. PulseParamValue::ParameterRef) are already final.
    ser.write_parameters(exp_builder.reborrow())?;
    ser.write_pulses(exp_builder.reborrow())?;
    ser.write_handles(exp_builder.reborrow())?;
    // Variables are discovered during the stream and section passes.
    ser.write_hqcs_variables(exp_builder.reborrow())?;

    let mut metadata = exp_builder.reborrow().init_metadata();
    if let Some(uid) = experiment.uid {
        metadata.set_uid(uid.to_str()?);
    }
    metadata.set_schema_version(experiment_capnp::SCHEMA_VERSION);
    metadata.set_created_by(format!("laboneq/{}", env!("CARGO_PKG_VERSION")));

    let bytes = if packed {
        let mut buf = Vec::new();
        capnp::serialize_packed::write_message(&mut buf, &message)
            .map_err(|e| laboneq_error!("Failed to write packed Cap'n Proto message: {e}"))?;
        buf
    } else {
        capnp::serialize::write_message_to_words(&message)
    };
    Ok(bytes)
}

// === Pure helper functions (no serialization state) ===

fn set_linear_start_stop(
    lin: &mut sweep_capnp::linear_sweep::Builder<'_>,
    start: &NumericValue,
    stop: &NumericValue,
) {
    match start {
        NumericValue::Real(v) => lin.reborrow().init_start().set_real(*v),
        NumericValue::Complex(re, im) => {
            let mut c = lin.reborrow().init_start().init_complex();
            c.set_real(*re);
            c.set_imag(*im);
        }
        // The schema has no integer variant; cast to f64. This is lossless for
        // all practical sweep parameter values (exact up to 2^53).
        NumericValue::Int(v) => lin.reborrow().init_start().set_real(*v as f64),
    }
    match stop {
        NumericValue::Real(v) => lin.reborrow().init_stop().set_real(*v),
        NumericValue::Complex(re, im) => {
            let mut c = lin.reborrow().init_stop().init_complex();
            c.set_real(*re);
            c.set_imag(*im);
        }
        // See comment on start branch above.
        NumericValue::Int(v) => lin.reborrow().init_stop().set_real(*v as f64),
    }
}

/// Map an HQCS type-catalog class (e.g. `laboneq.dsl.coprocessor.types.Int32`)
/// to the capnp enum by class name. The catalog is closed; unknown names are
/// an error.
fn coproc_type_from_py(type_obj: &Bound<'_, PyAny>) -> Result<coprocessor_capnp::VarType> {
    let py = type_obj.py();
    let name_obj = type_obj.getattr(intern!(py, "__name__"))?;
    let name: &str = name_obj.extract()?;
    use coprocessor_capnp::VarType;
    Ok(match name {
        "Int8" => VarType::Int8,
        "Int16" => VarType::Int16,
        "Int32" => VarType::Int32,
        "Int64" => VarType::Int64,
        "Uint8" => VarType::Uint8,
        "Uint16" => VarType::Uint16,
        "Uint32" => VarType::Uint32,
        "Uint64" => VarType::Uint64,
        "Phase" => VarType::Phase,
        "Frequency" => VarType::Frequency,
        "Amplitude" => VarType::Amplitude,
        "DiscriminationDataPacked" => VarType::DiscriminationDataPacked,
        "IqDataPacked" => VarType::IqDataPacked,
        "ScopeShot" => VarType::ScopeShot,
        "WaveformUpdate" => VarType::WaveformUpdate,
        other => {
            bail!("Unknown HQCS stream field type: {other}");
        }
    })
}

/// Serialize a Python scalar coprocessor value into a `VariableValue` builder.
fn set_variable_value(
    mut builder: coprocessor_capnp::variable_value::Builder<'_>,
    value: &Bound<'_, PyAny>,
) -> Result<()> {
    let py = value.py();
    if let Ok(v) = value.extract::<i64>() {
        builder.set_int_value(v);
    } else if let Ok(v) = value.extract::<f64>() {
        builder.set_float_value(v);
    } else if let Some(x) = value.getattr_opt(intern!(py, "radians"))? {
        builder.set_phase_radians(x.extract()?);
    } else if let Some(x) = value.getattr_opt(intern!(py, "hz"))? {
        builder.set_frequency_hz(x.extract()?);
    } else if let Some(x) = value.getattr_opt(intern!(py, "value"))? {
        builder.set_amplitude(x.extract()?);
    } else {
        bail!("Unsupported HQCS value: {value}");
    }
    Ok(())
}

/// Extract alignment from a Python section object and convert to capnp enum.
fn extract_alignment_capnp(obj: &Bound<'_, PyAny>) -> Result<Option<section_capnp::Alignment>> {
    let py = obj.py();
    let alignment_py = obj.getattr(intern!(py, "alignment"))?;
    if alignment_py.is_none() {
        return Ok(None);
    }
    let alignment_obj = alignment_py.getattr(intern!(py, "name"))?;
    let name: &str = alignment_obj.extract()?;
    match name {
        "LEFT" => Ok(Some(section_capnp::Alignment::Left)),
        "RIGHT" => Ok(Some(section_capnp::Alignment::Right)),
        _ => Err(laboneq_error!("Unknown section alignment: {name}")),
    }
}

/// Extract section timing mode from a Python section object and convert to capnp enum.
fn extract_section_timing_mode_capnp(
    obj: &Bound<'_, PyAny>,
) -> Result<Option<section_capnp::SectionTimingMode>> {
    let py = obj.py();
    let section_timing_mode_py = obj.getattr(intern!(py, "section_timing_mode"))?;
    if section_timing_mode_py.is_none() {
        return Ok(None);
    }
    let section_timing_mode_obj = section_timing_mode_py.getattr(intern!(py, "name"))?;
    let name: &str = section_timing_mode_obj.extract()?;
    match name {
        "RELAXED" => Ok(Some(section_capnp::SectionTimingMode::Relaxed)),
        "STRICT" => Ok(Some(section_capnp::SectionTimingMode::Strict)),
        _ => Err(laboneq_error!(
            "Unknown section section timing mode: {name}"
        )),
    }
}

/// Warn if the Python section object has inherited `Section` fields set that
/// are not supported by its section kind in the capnp schema.
fn warn_unsupported_section_fields(
    obj: &Bound<'_, PyAny>,
    supports_play_after: bool,
) -> Result<()> {
    let py = obj.py();
    let kind = obj
        .get_type()
        .name()
        .map_or_else(|_| "Section".to_owned(), |n| n.to_string());

    let length_py = obj.getattr(intern!(py, "length"))?;
    if !length_py.is_none() {
        laboneq_log::warn!("{} does not support 'length' — value will be ignored", kind);
    }

    let on_system_grid = obj
        .getattr(intern!(py, "on_system_grid"))?
        .extract::<Option<bool>>()?
        .unwrap_or(false);
    if on_system_grid {
        laboneq_log::warn!(
            "{} does not support 'on_system_grid' — value will be ignored",
            kind
        );
    }

    let trigger_py = obj.getattr(intern!(py, "trigger"))?;
    if let Ok(trigger_dict) = trigger_py.cast::<PyDict>().map_err(PyErr::from)
        && !trigger_dict.is_empty()
    {
        laboneq_log::warn!(
            "{} does not support 'trigger' — value will be ignored",
            kind
        );
    }

    if !supports_play_after {
        let play_after_py = obj.getattr(intern!(py, "play_after"))?;
        if !play_after_py.is_none() {
            laboneq_log::warn!(
                "{} does not support 'play_after' — value will be ignored",
                kind
            );
        }
    }

    Ok(())
}

fn collect_play_after_names(obj: &Bound<'_, PyAny>) -> Result<Vec<String>> {
    let py = obj.py();
    let play_after_py = obj.getattr(intern!(py, "play_after"))?;
    if play_after_py.is_none() {
        return Ok(Vec::new());
    }

    let mut names = Vec::new();
    if play_after_py.is_instance(&py.get_type::<PyList>())? {
        for item in play_after_py.try_iter()? {
            let item = item?;
            let uid_str = if item.is_instance(&py.get_type::<PyString>())? {
                item.extract::<String>()?
            } else {
                item.getattr(intern!(py, "uid"))?.extract::<String>()?
            };
            names.push(uid_str);
        }
    } else {
        let uid_str = if play_after_py.is_instance(&py.get_type::<PyString>())? {
            play_after_py.extract::<String>()?
        } else {
            play_after_py
                .getattr(intern!(py, "uid"))?
                .extract::<String>()?
        };
        names.push(uid_str);
    }

    Ok(names)
}

fn extract_acquisition_type_capnp(
    obj: &Bound<'_, PyAny>,
) -> Result<operation_capnp::AcquisitionType> {
    if obj.is_none() {
        return Ok(operation_capnp::AcquisitionType::Integration);
    }
    let acq_type_obj = obj.getattr(intern!(obj.py(), "name"))?;
    let name: &str = acq_type_obj.extract()?;
    Ok(match name {
        "INTEGRATION" => operation_capnp::AcquisitionType::Integration,
        "SPECTROSCOPY" | "SPECTROSCOPY_IQ" => operation_capnp::AcquisitionType::SpectroscopyIq,
        "SPECTROSCOPY_PSD" => operation_capnp::AcquisitionType::SpectroscopyPsd,
        "DISCRIMINATION" => operation_capnp::AcquisitionType::Discrimination,
        "RAW" => operation_capnp::AcquisitionType::Raw,
        _ => {
            bail!("Unknown acquisition type: {name}");
        }
    })
}

fn extract_py_numeric(obj: &Bound<'_, PyAny>) -> Result<NumericValue> {
    if let Ok(v) = obj.extract::<i64>() {
        return Ok(NumericValue::Int(v));
    }
    if obj.is_instance_of::<PyComplex>() {
        let c: num_complex::Complex64 = obj.extract()?;
        return Ok(NumericValue::Complex(c.re, c.im));
    }
    if let Ok(v) = obj.extract::<f64>() {
        return Ok(NumericValue::Real(v));
    }
    Err(laboneq_error!("Expected a numeric value"))
}

fn extract_explicit_values(
    obj: &Bound<'_, PyAny>,
    np: &Bound<'_, PyModule>,
) -> Result<ExplicitValues> {
    // Try extracting as numpy array or list of floats.
    // First check if complex by trying the first element.
    let list_len = obj
        .len()
        .map_err(|e| laboneq_error!("Failed to get length of explicit values: {e}"))?;
    if list_len == 0 {
        return Ok(ExplicitValues::Real(vec![]));
    }

    let first = obj.get_item(0)?;
    if first.is_instance_of::<PyComplex>() {
        let mut vals = Vec::with_capacity(list_len);
        for i in 0..list_len {
            let item = obj.get_item(i)?;
            let c: num_complex::Complex64 = item.extract()?;
            vals.push((c.re, c.im));
        }
        Ok(ExplicitValues::Complex(vals))
    } else {
        // Fast path via NumericArray: avoids Python-side `astype("float64")` overhead.
        if let Ok(arr) = NumericArray::from_py(obj) {
            return match arr {
                NumericArray::Integer64(v) => Ok(ExplicitValues::Int(v)),
                NumericArray::Float64(v) => Ok(ExplicitValues::Real(v)),
                // Complex was already handled by the PyComplex branch above.
                NumericArray::Complex64(v) => {
                    Ok(ExplicitValues::Real(v.into_iter().map(|c| c.re).collect()))
                }
            };
        }
        // Fall back to numpy for non-array inputs not handled by NumericArray::from_py.
        let py = obj.py();
        let as_array = np
            .call_method1(intern!(py, "asarray"), (obj,))?
            .call_method1(intern!(py, "astype"), (intern!(py, "float64"),))?;
        let flat: Vec<f64> = as_array.extract()?;
        Ok(ExplicitValues::Real(flat))
    }
}

fn extract_chunk_count(obj: &Bound<'_, PyAny>) -> Result<u32> {
    if let Ok(v) = obj.extract::<Option<u32>>() {
        let v = v.unwrap_or(1);
        if v < 1 {
            bail!("Chunk count must be >= 1, but {} was provided.", v);
        }
        return Ok(v);
    }
    if let Ok(v) = obj.extract::<i64>()
        && v < 1
    {
        bail!("Chunk count must be >= 1, but {} was provided.", v);
    }
    Err(laboneq_error!("Chunk count must be >= 1."))
}

fn set_sweep_value_from_py(
    obj: &Bound<'_, PyAny>,
    builder: &mut common_capnp::value::Builder<'_>,
) -> Result<()> {
    if obj.is_none() {
        return Ok(());
    }
    // Floats are the most common values, so check them first.
    // This does not cast ints to float so this can stay before
    // the int extraction.
    if let Ok(f) = obj.cast::<PyFloat>() {
        builder.reborrow().init_constant().set_real(f.value());
        return Ok(());
    }
    if let Ok(v) = obj.extract::<i64>() {
        builder.reborrow().init_constant().set_integer(v);
        return Ok(());
    }
    if obj.is_instance_of::<PyComplex>() {
        let c: num_complex::Complex64 = obj.extract()?;
        let mut cv = builder.reborrow().init_constant().init_complex();
        cv.set_real(c.re);
        cv.set_imag(c.im);
        return Ok(());
    }
    if let Ok(v) = obj.extract::<&str>() {
        builder.reborrow().init_constant().set_string_value(v);
        return Ok(());
    }
    Err(laboneq_error!("Cannot convert value to sweep value: {obj}"))
}

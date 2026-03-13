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

use std::collections::HashMap;

use crate::error::{Error, Result};
use crate::py_conversion::{DslType, DslTypes};
use crate::py_helpers::is_exact_type;
use numeric_array::NumericArray;

use laboneq_capnp::pulse::v1::{
    common_capnp, experiment_capnp, operation_capnp, pulse_capnp, section_capnp, sweep_capnp,
};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyComplex, PyDict, PyList, PyString};

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
    Pickled(Vec<u8>),
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
    /// Maps driver parameter index → derived parameter indices it drives.
    driving_parameters: HashMap<u32, Vec<u32>>,

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
            driving_parameters: HashMap::new(),
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
    /// Cached `pickle.dumps` callable — imported once to avoid per-call module lookup.
    pickle_dumps: Py<PyAny>,
}

impl<'py> Serializer<'py> {
    fn new(py: Python<'py>) -> Result<Self> {
        let pickle_dumps = py
            .import(intern!(py, "pickle"))?
            .getattr(intern!(py, "dumps"))?
            .unbind();
        Ok(Self {
            dsl_types: DslTypes::new(py)?,
            np: py.import(intern!(py, "numpy"))?,
            entities: EntityIndex::new(),
            signal_order: Vec::new(),
            signal_indices: HashMap::new(),
            pickle_dumps,
        })
    }

    // === Index lookup helpers ===
    // These always return the final (zero-based) index for the entity.

    fn get_signal_index(&self, uid: &str) -> Result<u32> {
        self.signal_indices.get(uid).copied().ok_or_else(|| {
            let mut available: Vec<&str> = self.signal_indices.keys().map(String::as_str).collect();
            available.sort();
            Error::new(format!(
                "Signal '{}' is not available in the experiment definition. \
                 Available signals are: '{}'.",
                uid,
                available.join(", ")
            ))
        })
    }

    // === Top-level serialization steps ===

    fn collect_signals(&mut self, experiment: &Bound<'_, PyAny>) -> Result<()> {
        let py = experiment.py();
        let signals_py = experiment.getattr(intern!(py, "signals"))?;
        let signals_list: Vec<Bound<'_, PyAny>> = crate::py_helpers::signal_iterable(&signals_py)?
            .try_iter()?
            .collect::<PyResult<_>>()?;

        // Collect UIDs.
        let mut uid_strings: Vec<String> = Vec::with_capacity(signals_list.len());
        for signal in &signals_list {
            let uid_obj = signal.getattr(intern!(py, "uid"))?;
            let uid_str: String = uid_obj.extract()?;
            uid_strings.push(uid_str);
        }

        // Sort alphabetically for deterministic ordering (matches py_conversion.rs).
        // Adjacent-window duplicate check is O(N log N) vs the prior O(N²) linear scan.
        uid_strings.sort();
        if let Some(w) = uid_strings.windows(2).find(|w| w[0] == w[1]) {
            return Err(Error::new(format!(
                "Duplicate signal uid '{}' in experiment.signals",
                w[0]
            )));
        }

        // Assign indices in sorted order.
        for uid_str in uid_strings {
            let idx = self.signal_order.len() as u32;
            self.signal_indices.insert(uid_str.clone(), idx);
            self.signal_order.push(uid_str);
        }
        Ok(())
    }

    fn serialize_signals(
        &mut self,
        experiment: &Bound<'_, PyAny>,
        mut exp_builder: experiment_capnp::experiment::Builder<'_>,
    ) -> Result<()> {
        self.collect_signals(experiment)?;
        let mut signals_builder = exp_builder
            .reborrow()
            .init_signals(self.signal_order.len() as u32);
        for (i, uid_str) in self.signal_order.iter().enumerate() {
            let mut sig_builder = signals_builder.reborrow().get(i as u32);
            sig_builder.set_uid(uid_str.as_str());
        }
        Ok(())
    }

    fn serialize_root_sections(
        &mut self,
        experiment: &Bound<'_, PyAny>,
        mut exp_builder: experiment_capnp::experiment::Builder<'_>,
    ) -> Result<()> {
        let py = experiment.py();
        let sections_py = experiment.getattr(intern!(py, "sections"))?;
        let sections_list: Vec<Bound<'_, PyAny>> =
            sections_py.try_iter()?.collect::<PyResult<_>>()?;

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
            match &param.kind {
                SweepParameterKind::Linear { start, stop, count } => {
                    let mut lin = pb.init_linear();
                    set_linear_start_stop(&mut lin, start, stop);
                    lin.set_count(*count);
                }
                SweepParameterKind::Explicit { values } => {
                    let mut explicit = pb.init_explicit_values();
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
            let mut amp = pb.reborrow().get_amplitude().map_err(Error::new)?;
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
                                PulseParamValue::Pickled(bytes) => {
                                    val.reborrow().init_constant().set_pickled_value(bytes);
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

    // === Section serialization ===

    fn serialize_section(
        &mut self,
        obj: &Bound<'_, PyAny>,
        mut builder: section_capnp::section::Builder<'_>,
    ) -> Result<()> {
        let py = obj.py();
        let uid_obj = obj.getattr(intern!(py, "uid"))?;
        let uid_str: &str = uid_obj.extract()?;
        builder.set_name(uid_str);

        let is_sweep = obj
            .get_type()
            .is(self.dsl_types.laboneq_type(DslType::Sweep));

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
            .is(self.dsl_types.laboneq_type(DslType::Section))
        {
            self.serialize_regular_section(obj, &mut builder)?;
        } else {
            return Err(Error::new(format!(
                "Unknown section type: {}",
                obj.get_type()
            )));
        }

        if !is_sweep {
            self.serialize_section_children(obj, &mut builder)?;
        }

        Ok(())
    }

    fn serialize_section_children(
        &mut self,
        obj: &Bound<'_, PyAny>,
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
            || ty.is(self.dsl_types.laboneq_type(DslType::PrngLoop)))
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
        obj: &Bound<'_, PyAny>,
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
        let mut all_param_indices = param_indices.clone();
        for &driver_idx in &param_indices {
            collect_derived_parameter_indices(
                driver_idx,
                &self.entities.driving_parameters,
                &mut all_param_indices,
            );
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

        {
            let mut params_list = sweep
                .reborrow()
                .init_parameters(all_param_indices.len() as u32);
            for (i, idx) in all_param_indices.iter().enumerate() {
                params_list.set(i as u32, *idx);
            }
        }
        sweep.set_reset_oscillator_phase(reset_oscillator_phase);

        if auto_chunking {
            sweep.reborrow().init_chunking().set_auto(());
        } else if chunk_count > 1 {
            sweep.reborrow().init_chunking().set_count(chunk_count);
        }

        Ok(())
    }

    fn collect_parameter(&mut self, obj: &Bound<'_, PyAny>) -> Result<u32> {
        let py = obj.py();
        // Keep the binding alive so we can borrow it as &str for the lookup,
        // avoiding a String allocation on the hot cache-hit path.
        let uid_binding = obj.getattr(intern!(py, "uid"))?;
        let uid_str: &str = uid_binding.extract()?;

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

        if is_exact_type(obj, linear_type)? {
            let start = extract_py_numeric(&obj.getattr(intern!(py, "start"))?)?;
            let stop = extract_py_numeric(&obj.getattr(intern!(py, "stop"))?)?;
            let count: u32 = obj.getattr(intern!(py, "count"))?.extract()?;
            self.entities.parameters.push(CollectedParameter {
                alias: uid_str.to_owned(),
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
                kind: SweepParameterKind::Explicit { values },
            });
            self.register_driving_parameters(idx, obj)?;
        } else {
            return Err(Error::new(format!(
                "Unknown parameter type: {}",
                obj.get_type()
            )));
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
            let driver_driven_by = driver.getattr_opt(intern!(py, "driven_by"))?;
            if driver_driven_by.as_ref().is_none_or(|v| v.is_none()) {
                // Driver is a root sweep parameter (not itself derived). Register it
                // and record that it drives `idx`. Intermediate parameters in a chain
                // (e.g. `par0`, `par1` from `delay_param * 1 * 1 * 1`) are skipped so
                // they don't appear as standalone experiment parameters without a sweep,
                // which would fail validation.
                let driver_idx = self.collect_parameter(driver)?;
                let entry = self
                    .entities
                    .driving_parameters
                    .entry(driver_idx)
                    .or_default();
                if !entry.contains(&idx) {
                    entry.push(idx);
                }
            } else {
                // Driver is itself a derived (intermediate) parameter — skip registering
                // it and recurse into its own drivers instead.
                self.register_driving_parameters(idx, driver)?;
            }
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
                return Err(Error::new("Sweep 'count' must be a positive integer"));
            }
            c
        } else {
            let c: f64 = count_py.extract()?;
            if c.fract() != 0.0 || c <= 0.0 || c > u32::MAX as f64 {
                return Err(Error::new("Sweep 'count' must be a positive integer"));
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
                    return Err(Error::new(format!("Unknown averaging mode: {name}")));
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
                    let t = repetition_time
                        .ok_or_else(|| Error::new("Repetition time required for CONSTANT mode"))?;
                    acq_loop.reborrow().init_repetition().set_constant(t);
                }
                "AUTO" => {
                    acq_loop.reborrow().init_repetition().set_auto(());
                }
                name => {
                    return Err(Error::new(format!("Unknown repetition mode: {name}")));
                }
            }
        }

        Ok(())
    }

    fn serialize_match_section(
        &mut self,
        obj: &Bound<'_, PyAny>,
        builder: &mut section_capnp::section::Builder<'_>,
    ) -> Result<()> {
        let py = obj.py();

        let handle_py = obj.getattr(intern!(py, "handle"))?;
        let user_register_py = obj.getattr(intern!(py, "user_register"))?;
        let sweep_parameter_py = obj.getattr(intern!(py, "sweep_parameter"))?;
        let prng_sample_py = obj.getattr(intern!(py, "prng_sample"))?;

        if [
            &handle_py,
            &user_register_py,
            &sweep_parameter_py,
            &prng_sample_py,
        ]
        .into_iter()
        .filter(|opt| !opt.is_none())
        .count()
            != 1
        {
            return Err(Error::new(
                "Match must have exactly one of handle, user_register, sweep_parameter, or prng_sample defined",
            ));
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

        let case_section = builder.reborrow().init_case_section();
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
        obj: &Bound<'_, PyAny>,
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
            serialize_call_op(obj, &mut builder)?;
        } else if ty.is(self.dsl_types.laboneq_type(DslType::SetNode)) {
            self.serialize_set_node_op(obj, &mut builder)?;
        } else if ty.is(self.dsl_types.laboneq_type(DslType::ResetOscillatorPhase)) {
            self.serialize_reset_oscillator_phase_op(obj, &mut builder)?;
        } else {
            return Err(Error::new(format!(
                "Unknown operation type: {}",
                obj.get_type()
            )));
        }

        Ok(())
    }

    fn serialize_set_node_op(
        &mut self,
        obj: &Bound<'_, PyAny>,
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
            self.set_sweep_value_from_py_or_param(
                &value_py,
                &mut set_node.reborrow().get_value().map_err(Error::new)?,
            )?;
        }

        Ok(())
    }

    fn serialize_play_op(
        &mut self,
        obj: &Bound<'_, PyAny>,
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
            self.set_sweep_value_from_py_or_param(
                &amplitude_py,
                &mut play.reborrow().get_amplitude().map_err(Error::new)?,
            )?;
        }

        // Phase
        let phase_py = obj.getattr(intern!(py, "phase"))?;
        if !phase_py.is_none() {
            self.set_sweep_value_from_py_or_param(
                &phase_py,
                &mut play.reborrow().get_phase().map_err(Error::new)?,
            )?;
        }

        // Increment oscillator phase
        let inc_osc_py = obj.getattr(intern!(py, "increment_oscillator_phase"))?;
        if !inc_osc_py.is_none() {
            self.set_sweep_value_from_py_or_param(
                &inc_osc_py,
                &mut play
                    .reborrow()
                    .get_increment_oscillator_phase()
                    .map_err(Error::new)?,
            )?;
        }

        // Set oscillator phase
        let set_osc_py = obj.getattr(intern!(py, "set_oscillator_phase"))?;
        if !set_osc_py.is_none() {
            self.set_sweep_value_from_py_or_param(
                &set_osc_py,
                &mut play
                    .reborrow()
                    .get_set_oscillator_phase()
                    .map_err(Error::new)?,
            )?;
        }

        // Length
        let length_py = obj.getattr(intern!(py, "length"))?;
        if !length_py.is_none() {
            self.set_sweep_value_from_py_or_param(
                &length_py,
                &mut play.reborrow().get_length().map_err(Error::new)?,
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
        obj: &Bound<'_, PyAny>,
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
            self.set_pulse_parameter_value(value, &mut val_builder)?;
        }
        Ok(())
    }

    fn serialize_markers(
        &mut self,
        obj: &Bound<'_, PyAny>,
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
                "marker1" => set_marker(markers.reborrow().get_marker1().map_err(Error::new)?)?,
                "marker2" => set_marker(markers.reborrow().get_marker2().map_err(Error::new)?)?,
                _ => return Err(Error::new(format!("Unknown marker: {marker_name}"))),
            }
        }
        Ok(())
    }

    fn serialize_delay_op(
        &mut self,
        obj: &Bound<'_, PyAny>,
        builder: &mut operation_capnp::operation::Builder<'_>,
    ) -> Result<()> {
        let py = obj.py();
        let mut delay = builder.reborrow().init_delay();

        let signal_obj = obj.getattr(intern!(py, "signal"))?;
        let signal_str: &str = signal_obj.extract()?;
        let signal_id = self.get_signal_index(signal_str)?;
        delay.set_signal(signal_id);

        let time_py = obj.getattr(intern!(py, "time"))?;
        self.set_sweep_value_from_py_or_param(
            &time_py,
            &mut delay.reborrow().get_time().map_err(Error::new)?,
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
        obj: &Bound<'_, PyAny>,
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
        obj: &Bound<'_, PyAny>,
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
            .map_err(|_| Error::new("Invalid type for field 'handle'"))?;
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
                .map_err(Error::new)?
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
                        self.set_pulse_parameter_value(value, &mut val_builder)?;
                    }
                }
            }
        }

        Ok(())
    }

    fn serialize_reset_oscillator_phase_op(
        &mut self,
        obj: &Bound<'_, PyAny>,
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

    // === Pulse collection ===

    fn collect_pulse(&mut self, obj: &Bound<'_, PyAny>) -> Result<u32> {
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
                            // Arbitrary Python objects (e.g. ExternalParameter, SciPy interp objects)
                            // are pickled into bytes for opaque round-tripping through capnp.
                            let pickled = self.pickle_dumps.bind(py).call1((&value,))?;
                            let bytes: Vec<u8> = pickled
                                .cast::<PyBytes>()
                                .map_err(Error::new)?
                                .as_bytes()
                                .to_vec();
                            PulseParamValue::Pickled(bytes)
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
                    crate::py_conversion::iq_to_complex(&self.np, &arr)
                        .map_err(|e| Error::new(e.to_string()))?
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

    fn set_pulse_parameter_value(
        &mut self,
        obj: &Bound<'_, PyAny>,
        builder: &mut common_capnp::value::Builder<'_>,
    ) -> Result<()> {
        match self.set_sweep_value_from_py_or_param(obj, builder) {
            Ok(()) => Ok(()),
            Err(_) => self.set_external_opaque_constant(obj, builder),
        }
    }

    fn set_external_opaque_constant(
        &self,
        obj: &Bound<'_, PyAny>,
        builder: &mut common_capnp::value::Builder<'_>,
    ) -> Result<()> {
        if let Ok(raw) = obj.cast::<PyBytes>() {
            // Plain bytes — pass through without pickling.
            builder
                .reborrow()
                .init_constant()
                .set_raw_bytes_value(raw.as_bytes());
            return Ok(());
        }
        // Pickling is strictly a fallback for arbitrary Python objects passed into custom
        // functional pulse parameters (e.g., a SciPy interpolation object). Because the
        // Rust compiler does not execute these (they are evaluated in Python during waveform
        // sampling), they must be passed opaquely.
        let py = obj.py();
        let pickled = self.pickle_dumps.bind(py).call1((obj,))?;
        let bytes = pickled.cast::<PyBytes>().map_err(Error::new)?.as_bytes();
        builder.reborrow().init_constant().set_pickled_value(bytes);
        Ok(())
    }

    fn set_sweep_value_from_py_or_param(
        &mut self,
        obj: &Bound<'_, PyAny>,
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
}

// === Public entry point ===

/// Serializes a Python experiment object tree to Cap'n Proto bytes.
///
/// Returns the serialized bytes as a `Vec<u8>`.
pub(crate) fn serialize_experiment(experiment: &Bound<'_, PyAny>, packed: bool) -> Result<Vec<u8>> {
    let py = experiment.py();
    let mut ser = Serializer::new(py)?;

    let mut message = capnp::message::Builder::new_default();
    let mut exp_builder = message.init_root::<experiment_capnp::experiment::Builder<'_>>();

    // Signals are indexed first (sorted alphabetically for determinism) so that
    // signal references written during section tree traversal use final indices.
    ser.serialize_signals(experiment, exp_builder.reborrow())?;

    // Traverse the section tree. All entity collections (pulses, parameters,
    // handles) accumulate into `ser.entities` with final zero-based indices
    // assigned at first insertion.
    ser.serialize_root_sections(experiment, exp_builder.reborrow())?;

    // Write entity definition lists. Indices used for cross-references within
    // definitions (e.g. PulseParamValue::ParameterRef) are already final.
    ser.write_parameters(exp_builder.reborrow())?;
    ser.write_pulses(exp_builder.reborrow())?;
    ser.write_handles(exp_builder.reborrow())?;

    let mut metadata = exp_builder.reborrow().init_metadata();
    let uid_py = experiment.getattr(intern!(py, "uid"))?;
    if let Ok(Some(uid_str)) = uid_py.extract::<Option<&str>>() {
        metadata.set_uid(uid_str);
    }
    metadata.set_schema_version(experiment_capnp::SCHEMA_VERSION);
    metadata.set_created_by(format!("laboneq/{}", env!("CARGO_PKG_VERSION")));

    let bytes = if packed {
        let mut buf = Vec::new();
        capnp::serialize_packed::write_message(&mut buf, &message)
            .map_err(|e| Error::new(format!("Failed to write packed Cap'n Proto message: {e}")))?;
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
        _ => Err(Error::new(format!("Unknown section alignment: {name}"))),
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

fn serialize_call_op(
    obj: &Bound<'_, PyAny>,
    builder: &mut operation_capnp::operation::Builder<'_>,
) -> Result<()> {
    let py = obj.py();
    let mut call = builder.reborrow().init_call();

    let func_name_py = obj.getattr(intern!(py, "func_name"))?;
    if !func_name_py.is_none() {
        let func_name: &str = func_name_py.extract()?;
        call.set_callback_id(func_name);
    }

    // TODO: serialize call args

    Ok(())
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
            return Err(Error::new(format!("Unknown acquisition type: {name}")));
        }
    })
}

fn collect_derived_parameter_indices(
    driver_idx: u32,
    driving_parameters: &HashMap<u32, Vec<u32>>,
    out: &mut Vec<u32>,
) {
    if let Some(derived) = driving_parameters.get(&driver_idx) {
        for &derived_idx in derived {
            if !out.contains(&derived_idx) {
                out.push(derived_idx);
                collect_derived_parameter_indices(derived_idx, driving_parameters, out);
            }
        }
    }
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
    Err(Error::new("Expected a numeric value"))
}

fn extract_explicit_values(
    obj: &Bound<'_, PyAny>,
    np: &Bound<'_, PyModule>,
) -> Result<ExplicitValues> {
    // Try extracting as numpy array or list of floats.
    // First check if complex by trying the first element.
    let list_len = obj
        .len()
        .map_err(|e| Error::new(format!("Failed to get length of explicit values: {e}")))?;
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
            return Err(Error::new(format!(
                "Chunk count must be >= 1, but {} was provided.",
                v
            )));
        }
        return Ok(v);
    }
    if let Ok(v) = obj.extract::<i64>()
        && v < 1
    {
        return Err(Error::new(format!(
            "Chunk count must be >= 1, but {} was provided.",
            v
        )));
    }
    Err(Error::new("Chunk count must be >= 1."))
}

fn set_sweep_value_from_py(
    obj: &Bound<'_, PyAny>,
    builder: &mut common_capnp::value::Builder<'_>,
) -> Result<()> {
    if obj.is_none() {
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
    if let Ok(v) = obj.extract::<f64>() {
        builder.reborrow().init_constant().set_real(v);
        return Ok(());
    }
    if let Ok(v) = obj.extract::<&str>() {
        builder.reborrow().init_constant().set_string_value(v);
        return Ok(());
    }
    Err(Error::new(format!(
        "Cannot convert value to sweep value: {obj}"
    )))
}

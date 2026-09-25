// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

//! This module contains the Cap'n Proto serialization logic for compiled experiment.

use std::collections::hash_map::Entry;
use std::sync::Arc;

use anyhow::Context;
use laboneq_capnp::artifacts::ArtifactsWriter;
use laboneq_capnp::compiled_experiment::v1::{
    compiled_experiment_capnp, execution_capnp, result_shape_capnp, rt_loop_properties_capnp,
};
use laboneq_capnp::convert::narrow;

use laboneq_common::named_id::NamedIdStore;
use laboneq_dsl::hqcs::CoprocessorTable;
use laboneq_dsl::types::{
    AcquisitionType, AveragingMode, ExternalParameterUid, HandleUid, ParameterUid, SweepParameter,
};
use laboneq_ir::pulse_sheet_schedule::PulseSheetSchedule;
use laboneq_py_utils::py_object_store::PyObjectStore;
use numeric_array::NumericArray;
use rustc_hash::FxHashMap;

use crate::capnp_artifacts::{ArtifactSerializerContext, resolve_symbol, write_numeric_array};
use crate::compiled_experiment::{
    CompiledExperiment, ExecutionTiming, Metadata, RealTimeProperties, ResultShapes,
};
use crate::compiler_backend::CompilerArtifact;
use crate::error::{Error, Result};
use crate::execution::{Execution, Statement};
use crate::result_shape::{AxisValues, HandleResultShape};

/// Serializes a [`CompiledExperiment`] into a Cap'n Proto message, returning the serialized
/// unpacked message.
pub(crate) fn serialize_compiled_experiment<A: CompilerArtifact>(
    compiled_experiment: &CompiledExperiment<A>,
) -> Result<Vec<u8>> {
    let mut message = capnp::message::Builder::new_default();
    let mut builder = CompiledExperimentCapnpBuilder::new(
        &mut message,
        &compiled_experiment.id_store,
        &compiled_experiment.external_parameters,
    );

    builder.serialize_metadata(&compiled_experiment.metadata);
    builder.serialize_device_setup_fingerprint(&compiled_experiment.device_setup_fingerprint);
    builder.serialize_parameters(&compiled_experiment.sweep_parameters)?;
    builder.serialize_acquisition_handles(&compiled_experiment.acquisition_handles)?;
    builder.serialize_execution(&compiled_experiment.execution)?;
    builder.serialize_rt_properties(&compiled_experiment.real_time_properties);
    builder.serialize_result_shapes(&compiled_experiment.result_shapes)?;
    builder.serialize_execution_timing(&compiled_experiment.execution_timing);
    builder.serialize_artifacts(&compiled_experiment.artifacts)?;
    builder.serialize_coprocessors(&compiled_experiment.coprocessors)?;
    if let Some(schedule) = &compiled_experiment.pulse_sheet_schedule {
        builder.serialize_pulse_sheet(schedule)?;
    }

    // TODO: Consider also packed serialization?
    Ok(capnp::serialize::write_message_to_words(&message))
}

struct CompiledExperimentCapnpBuilder<'a> {
    exp_builder: compiled_experiment_capnp::compiled_experiment::Builder<'a>,
    /// Resolved sweep parameter handles to its row in `CompiledExperiment.parameters`.
    parameter_indices: FxHashMap<ParameterUid, usize>,
    /// Resolved result shape's handle to its row in `CompiledExperiment.acquisitionHandles`.
    handle_indices: FxHashMap<HandleUid, u32>,

    id_store: &'a NamedIdStore,
    external_parameters: &'a PyObjectStore<ExternalParameterUid>,
}

impl<'a> CompiledExperimentCapnpBuilder<'a> {
    fn new(
        message: &'a mut capnp::message::Builder<capnp::message::HeapAllocator>,
        id_store: &'a NamedIdStore,
        external_parameters: &'a PyObjectStore<ExternalParameterUid>,
    ) -> Self {
        let exp_builder =
            message.init_root::<compiled_experiment_capnp::compiled_experiment::Builder<'a>>();
        CompiledExperimentCapnpBuilder {
            exp_builder,
            id_store,
            parameter_indices: Default::default(),
            handle_indices: Default::default(),
            external_parameters,
        }
    }

    fn serialize_metadata(&mut self, metadata: &Metadata) {
        let mut metadata_builder = self.exp_builder.reborrow().init_metadata();
        metadata_builder.set_schema_version(compiled_experiment_capnp::SCHEMA_VERSION);

        let created_by = format!("{}/{}", metadata.producer_name, metadata.producer_version);
        metadata_builder.set_created_by(created_by.as_str());
    }

    fn serialize_device_setup_fingerprint(&mut self, fingerprint: &str) {
        self.exp_builder
            .reborrow()
            .set_hardware_fingerprint(fingerprint);
    }

    /// Builds and writes `CompiledExperiment.parameters`.
    ///
    /// Populates `self.parameter_indices` with the index of each parameter in that table, for
    /// other fields to reference by `Common.Id`.
    fn serialize_parameters(&mut self, sweep_parameters: &[SweepParameter]) -> Result<()> {
        let mut params_builder = self
            .exp_builder
            .reborrow()
            .init_parameters(narrow(sweep_parameters.len(), "parameters")?);
        for (i, param) in sweep_parameters.iter().enumerate() {
            // Register the parameter's index for references.
            self.parameter_indices.insert(param.uid, i);

            // Write the parameter's uid and values into the Cap'n Proto table.
            let mut param_builder = params_builder.reborrow().get(i as u32);
            param_builder.set_uid(resolve_symbol(self.id_store, param.uid)?);
            write_numeric_array(param_builder.init_values(), param.values.as_ref())?;
        }
        Ok(())
    }

    /// Builds and writes `CompiledExperiment.acquisitionHandles`.
    ///
    /// Populates `self.handle_indices` with the index of each handle in that table, for
    /// other fields to reference by `Common.Id`.
    fn serialize_acquisition_handles(&mut self, handles: &[HandleUid]) -> Result<()> {
        let handle_count = narrow(handles.len(), "acquisitionHandles")?;

        let mut handles_builder = self
            .exp_builder
            .reborrow()
            .init_acquisition_handles(handle_count);
        for (i, handle) in handles.iter().enumerate() {
            // Register the handle's index for references.
            self.handle_indices.insert(*handle, i as u32);

            // Write the handle's uid into the Cap'n Proto table.
            handles_builder
                .reborrow()
                .get(i as u32)
                .set_uid(resolve_symbol(self.id_store, *handle)?);
        }
        Ok(())
    }

    /// Builds and writes `CompiledExperiment.coprocessors`.
    fn serialize_coprocessors(&mut self, coprocessors: &CoprocessorTable) -> Result<()> {
        let mut builder = self
            .exp_builder
            .reborrow()
            .init_coprocessors(narrow(coprocessors.mappings.len(), "coprocessors")?);
        for (i, (label, inventory_key)) in coprocessors.mappings.iter().enumerate() {
            let mut entry = builder.reborrow().get(i as u32);
            entry.set_label(label.as_str());
            entry.set_inventory_key(inventory_key.as_str());
            // Payloads are keyed by inventory key, and only declared ones are present.
            if let Some(payload) = coprocessors.payloads.get(inventory_key) {
                entry.set_payload(payload);
            }
        }
        Ok(())
    }

    /// Serializes `CompiledExperiment.resultShapes`, including the `explicitValues` table
    /// for `AxisValues::Explicit` axis values.
    ///
    /// `explicitValues` is local to result shapes -- not `CompiledExperiment.parameters`
    /// -- since these arrays back no named parameter and `execution` never references
    /// them. They dedup by `Arc` pointer identity (not contents), so an array shared
    /// across axes/handles (e.g. the same PRNG loop's samples, referenced by every
    /// acquisition inside it) is encoded exactly once. `Parameter` axis values resolve via
    /// `self.parameter_indices`; each shape's handle resolves via `self.handle_indices`
    /// (built by [`Self::serialize_acquisition_handles`]).
    fn serialize_result_shapes(&mut self, result_shapes: &ResultShapes) -> Result<()> {
        let handle_result_shapes = &result_shapes.handle_result_shapes;

        let mut explicit_values: Vec<&NumericArray> = Vec::new();
        let mut explicit_value_indices: FxHashMap<*const NumericArray, u32> = FxHashMap::default();
        for shape in handle_result_shapes {
            for axis in &shape.axis_values {
                for value in axis {
                    if let AxisValues::Explicit(arr) = value {
                        // `Entry::Vacant` rather than `or_insert_with`: the index is
                        // fallible, and a closure cannot propagate that with `?`.
                        if let Entry::Vacant(slot) = explicit_value_indices.entry(Arc::as_ptr(arr))
                        {
                            slot.insert(narrow(explicit_values.len(), "explicitValues")?);
                            explicit_values.push(arr.as_ref());
                        }
                    }
                }
            }
        }

        let mut result_shapes_builder = self.exp_builder.reborrow().init_result_shapes();

        let mut explicit_values_builder = result_shapes_builder
            .reborrow()
            .init_explicit_values(narrow(explicit_values.len(), "explicitValues")?);
        for (i, arr) in explicit_values.iter().enumerate() {
            write_numeric_array(explicit_values_builder.reborrow().get(i as u32), arr)?;
        }

        let mut shapes_builder = result_shapes_builder
            .reborrow()
            .init_shapes(narrow(handle_result_shapes.len(), "shapes")?);
        for (i, shape) in handle_result_shapes.iter().enumerate() {
            let mut entry_builder = shapes_builder.reborrow().get(i as u32);
            let Some(handle) = self.handle_indices.get(&shape.handle).copied() else {
                return Err(Error::new("Result shape missing acquisition handle"));
            };
            entry_builder.set_handle(handle);
            serialize_handle_result_shape(
                self.id_store,
                &self.parameter_indices,
                &explicit_value_indices,
                shape,
                entry_builder.init_shape(),
            )?;
        }
        Ok(())
    }

    fn serialize_execution(&mut self, execution: &Execution) -> Result<()> {
        let mut exec_builder = self.exp_builder.reborrow().init_execution();
        let statements = &execution.statements;
        let instruction_count = ExecutionSerializer::count_instructions(statements);
        let stmts_builder = exec_builder.reborrow().init_instructions(instruction_count);

        let ctx = ArtifactSerializerContext::new(
            self.id_store,
            self.external_parameters,
            &self.parameter_indices,
            &self.handle_indices,
        );
        ExecutionSerializer { ctx: &ctx }.serialize_statements(stmts_builder, statements)
    }

    fn serialize_rt_properties(&mut self, properties: &RealTimeProperties) {
        let mut rt_builder = self.exp_builder.reborrow().init_real_time_properties();
        rt_builder.set_acquisition_type(serialize_acquisition_type(properties.acquisition_type));
        rt_builder.set_averaging_mode(serialize_averaging_mode(properties.averaging_mode));
        rt_builder.set_shots(properties.shots.get());
        let mut chunk_count_builder = rt_builder.init_chunk_count();
        match properties.chunk_count {
            Some(chunk_count) => chunk_count_builder.set_value(chunk_count),
            None => chunk_count_builder.set_none(()),
        }
    }

    fn serialize_execution_timing(&mut self, timing: &ExecutionTiming) {
        let mut timing_builder = self.exp_builder.reborrow().init_timing();
        timing_builder.set_total_duration(timing.total_execution_time);
        timing_builder.set_max_step_duration(timing.max_step_execution_time);
    }

    fn serialize_pulse_sheet(&mut self, schedule: &PulseSheetSchedule) -> Result<()> {
        let json_bytes = serde_json::to_vec(schedule).map_err(Error::new)?;
        self.exp_builder
            .reborrow()
            .init_pulse_sheet()
            .set_json(&json_bytes);
        Ok(())
    }

    /// Fills `CompiledExperiment.artifacts` (an opaque `AnyPointer`) from the backend's
    /// own [`ArtifactsWriter`], built via
    /// [`CompilerArtifact::to_capnp_writer`]. The writer resolves `parameters`/
    /// `acquisitionHandles` references through the same context [`serialize_execution`]
    /// uses, so backend artifacts and `execution` agree on indices.
    ///
    /// [`ArtifactsWriter`]: laboneq_capnp::artifacts::ArtifactsWriter
    /// [`serialize_execution`]: Self::serialize_execution
    fn serialize_artifacts<A: CompilerArtifact>(&mut self, artifacts: &A) -> Result<()> {
        let ctx = ArtifactSerializerContext::new(
            self.id_store,
            self.external_parameters,
            &self.parameter_indices,
            &self.handle_indices,
        );
        let writer = artifacts.to_capnp_writer(&ctx);
        writer
            .write(self.exp_builder.reborrow().init_artifacts())
            .map_err(Error::from)
            .with_context(|| "Error while generating backend artifacts")?;
        Ok(())
    }
}

fn serialize_acquisition_type(value: AcquisitionType) -> rt_loop_properties_capnp::AcquisitionType {
    match value {
        AcquisitionType::Integration => rt_loop_properties_capnp::AcquisitionType::Integration,
        AcquisitionType::Raw => rt_loop_properties_capnp::AcquisitionType::Raw,
        AcquisitionType::Discrimination => {
            rt_loop_properties_capnp::AcquisitionType::Discrimination
        }
        // The schema has no generic "spectroscopy" variant; IQ is the default flavor,
        // mirroring the inverse mapping in `capnp_deserializer::deserialize_acquisition_type`.
        AcquisitionType::Spectroscopy | AcquisitionType::SpectroscopyIq => {
            rt_loop_properties_capnp::AcquisitionType::SpectroscopyIq
        }
        AcquisitionType::SpectroscopyPsd => {
            rt_loop_properties_capnp::AcquisitionType::SpectroscopyPsd
        }
    }
}

fn serialize_handle_result_shape(
    id_store: &NamedIdStore,
    parameter_indices: &FxHashMap<ParameterUid, usize>,
    explicit_value_indices: &FxHashMap<*const NumericArray, u32>,
    shape: &HandleResultShape,
    mut builder: result_shape_capnp::handle_result_shape::Builder<'_>,
) -> Result<()> {
    let mut shape_list = builder
        .reborrow()
        .init_shape(narrow(shape.shape.len(), "shape")?);
    for (i, dim) in shape.shape.iter().enumerate() {
        shape_list.set(i as u32, narrow(*dim, "shape")?);
    }

    let mut axes_builder = builder
        .reborrow()
        .init_axes(narrow(shape.axis_names.len(), "axes")?);
    for (i, (names, values)) in shape.axis_names.iter().zip(&shape.axis_values).enumerate() {
        let mut params_builder = axes_builder
            .reborrow()
            .get(i as u32)
            .init_parameters(narrow(names.len(), "axes.parameters")?);
        for (j, (name, value)) in names.iter().zip(values).enumerate() {
            let mut param_builder = params_builder.reborrow().get(j as u32);
            param_builder.set_name(resolve_symbol(id_store, *name)?);
            serialize_axis_parameter(
                parameter_indices,
                explicit_value_indices,
                value,
                &mut param_builder,
            )?;
        }
    }

    match shape.chunked_axis_index {
        Some(idx) => builder
            .reborrow()
            .init_chunked_axis_index()
            .set_value(narrow(idx, "chunkedAxisIndex")?),
        None => builder.reborrow().init_chunked_axis_index().set_none(()),
    }

    let mut mask_builder =
        builder.init_match_case_mask(narrow(shape.match_case_mask.len(), "matchCaseMask")?);
    for (i, (axis, rows)) in shape.match_case_mask.iter().enumerate() {
        let mut mask_entry = mask_builder.reborrow().get(i as u32);
        mask_entry.set_axis(narrow(*axis, "matchCaseMask.axis")?);
        let mut rows_builder = mask_entry.init_rows(narrow(rows.len(), "matchCaseMask.rows")?);
        for (j, row) in rows.iter().enumerate() {
            rows_builder.set(j as u32, narrow(*row, "matchCaseMask.rows")?);
        }
    }
    Ok(())
}

fn serialize_axis_parameter(
    parameter_indices: &FxHashMap<ParameterUid, usize>,
    explicit_value_indices: &FxHashMap<*const NumericArray, u32>,
    value: &AxisValues,
    builder: &mut result_shape_capnp::axis_parameter::Builder<'_>,
) -> Result<()> {
    match value {
        AxisValues::Parameter(uid) => {
            let idx = parameter_indices
                .get(uid)
                .copied()
                .expect("Parameter index not found");
            builder.set_parameter_ref(narrow(idx, "parameterRef")?);
        }
        AxisValues::Shots(count) => builder.set_shots(narrow(*count, "shots")?),
        AxisValues::Explicit(arr) => {
            let idx = explicit_value_indices[&Arc::as_ptr(arr)];
            builder.set_explicit_ref(idx);
        }
    }
    Ok(())
}

fn serialize_averaging_mode(value: AveragingMode) -> rt_loop_properties_capnp::AveragingMode {
    match value {
        AveragingMode::Sequential => rt_loop_properties_capnp::AveragingMode::Sequential,
        AveragingMode::Cyclic => rt_loop_properties_capnp::AveragingMode::Cyclic,
        AveragingMode::SingleShot => rt_loop_properties_capnp::AveragingMode::SingleShot,
    }
}

// ========================================================================================
// Execution serializer

struct ExecutionSerializer<'a> {
    ctx: &'a ArtifactSerializerContext<'a>,
}

impl ExecutionSerializer<'_> {
    /// Number of flat `Instruction` slots `statements` serializes to. A `Loop` isn't a
    /// single slot: it contributes two framing slots (`loopStart`/`loopEnd`) around its
    /// recursively-flattened body, since the capnp schema represents nesting as a flat
    /// list of paired markers rather than a recursive tree.
    fn count_instructions(statements: &[Statement]) -> u32 {
        statements
            .iter()
            .map(|stmt| match stmt {
                Statement::Loop { body, .. } => 2 + Self::count_instructions(body),
                _ => 1,
            })
            .sum()
    }

    fn serialize_statements(
        &self,
        mut builder: capnp::struct_list::Builder<'_, execution_capnp::instruction::Owned>,
        statements: &[Statement],
    ) -> Result<()> {
        let mut next = 0u32;
        self.write_statements(&mut builder, &mut next, statements)
    }

    /// Writes `statements` into `builder` starting at `*next`, advancing `*next` past
    /// every instruction written — including, for a `Loop`, its framing
    /// `loopStart`/`loopEnd` and its recursively-flattened body in between.
    fn write_statements(
        &self,
        builder: &mut capnp::struct_list::Builder<'_, execution_capnp::instruction::Owned>,
        next: &mut u32,
        statements: &[Statement],
    ) -> Result<()> {
        for stmt in statements {
            self.write_statement(builder, next, stmt)?;
        }
        Ok(())
    }

    fn write_statement(
        &self,
        builder: &mut capnp::struct_list::Builder<'_, execution_capnp::instruction::Owned>,
        next: &mut u32,
        statement: &Statement,
    ) -> Result<()> {
        let Statement::Loop { count, body } = statement else {
            self.serialize_simple_statement(builder.reborrow().get(*next), statement)?;
            *next += 1;
            return Ok(());
        };

        builder
            .reborrow()
            .get(*next)
            .init_loop_start()
            .set_count(count.get());
        *next += 1;

        self.write_statements(builder, next, body)?;

        builder.reborrow().get(*next).set_loop_end(());
        *next += 1;
        Ok(())
    }

    /// Serializes a single non-`Loop` statement into its already-indexed slot.
    fn serialize_simple_statement(
        &self,
        mut builder: execution_capnp::instruction::Builder<'_>,
        statement: &Statement,
    ) -> Result<()> {
        match statement {
            Statement::SetParameter { parameter, .. } => {
                builder
                    .reborrow()
                    .init_set_parameter()
                    .set_parameter_ref(self.ctx.parameter_ref(*parameter)?);
            }
            Statement::Loop { .. } => unreachable!("Loop is flattened by write_statement"),
            Statement::ExecRealTime => {
                builder.reborrow().init_exec_real_time();
            }
            Statement::ExecCallback { callback_id, args } => {
                let mut cb_builder = builder.reborrow().init_exec_callback();
                cb_builder
                    .reborrow()
                    .set_callback_id(self.ctx.resolve_symbol(*callback_id)?);
                let mut args_builder = cb_builder.reborrow().init_args(narrow(args.len(), "args")?);
                for (i, arg) in args.iter().enumerate() {
                    let mut entry = args_builder.reborrow().get(i as u32);
                    entry.reborrow().set_key(self.ctx.resolve_symbol(arg.key)?);
                    let mut val_builder = entry.reborrow().init_value();
                    self.ctx
                        .serialize_external_or_value(arg.value.clone(), &mut val_builder)?;
                }
            }
            Statement::SetNode { path, value } => {
                let mut node_builder = builder.reborrow().init_set_node();
                node_builder
                    .reborrow()
                    .set_path(self.ctx.resolve_symbol(*path)?);
                let mut val_builder = node_builder.reborrow().init_value();
                self.ctx
                    .serialize_value_or_parameter_literal(value.clone(), &mut val_builder)?;
            }
        }
        Ok(())
    }
}

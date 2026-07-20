// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::collections::HashSet;
use std::num::NonZero;
use std::ops::Range;
use std::sync::Arc;
use std::vec;

use indexmap::IndexMap;

use laboneq_common::named_id::NamedId;
use laboneq_common::named_id::NamedIdStore;
use laboneq_common::prng_generator_qccs::PrngGeneratorQccs;
use laboneq_dsl::ExperimentNode;
use laboneq_dsl::operation::Operation;
use laboneq_dsl::operation::PrngLoop;
use laboneq_dsl::operation::Sweep;
use laboneq_dsl::types::AcquisitionType;
use laboneq_dsl::types::AveragingMode;
use laboneq_dsl::types::HandleUid;
use laboneq_dsl::types::ParameterUid;

use laboneq_dsl::operation::{Acquire, AveragingLoop, Match};
use laboneq_dsl::types::SignalUid;
use laboneq_dsl::types::{MatchTarget, SweepParameter};

use numeric_array::NumericArray;

use crate::error::{Error, Result};

/// Extract experiment handle result shapes.
///
/// The result shape evaluation is a two-stage process:
///
/// - Determine the overall shapes based on the experiment structure
/// - Extend the shapes according to resolved raw acquisition lengths by using the [`ResultShapes::get_shapes`] method.
///
/// This is due to fact that sample precise raw acquisition length is not known until code generation
/// and it can be affected by chunking
pub(crate) fn extract_result_shapes<'a>(
    root: &ExperimentNode,
    sweep_parameters: impl Iterator<Item = &'a SweepParameter> + 'a,
    id_store: &mut NamedIdStore,
) -> Result<ResultShapes> {
    let mut extractor = ResultShapeExtractor::new(sweep_parameters);
    extractor.visit_node(root)?;
    Ok(ResultShapes {
        handle_result_shapes: extractor.handle_result_shapes,
        raw_acquisition_axis_name: id_store.get_or_insert("samples"),
        needs_raw_acquisition_injection: extractor.needs_raw_acquisition_injection,
    })
}

/// Result shapes for all handles in the experiment.
///
/// The shapes are determined by the structure of the experiment, e.g. loops and match statements, and can be extended with an additional axis for raw acquisition length if needed.
/// The final shapes for each handle can be obtained by calling the [`ResultShapes::get_shapes`] method with the resolved raw acquisition lengths for each signal-handle pair, if applicable.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ResultShapes {
    handle_result_shapes: Vec<HandleResultShape>,
    raw_acquisition_axis_name: NamedId,
    needs_raw_acquisition_injection: bool,
}

impl ResultShapes {
    pub(crate) fn raw_acquisitions(&self) -> impl Iterator<Item = (SignalUid, HandleUid)> {
        // Deduplicate by handle/signal combination, as there can be multiple shapes for the same handle/signal if the handle is used in multiple match cases.
        let mut seen = HashSet::new();
        self.handle_result_shapes
            .iter()
            .filter(|_| self.needs_raw_acquisition_injection)
            .filter(move |shape| seen.insert((shape.signal, shape.handle)))
            .map(|shape| (shape.signal, shape.handle))
    }

    /// Get the final result shape.
    pub(crate) fn get_shapes(
        &self,
        raw_acquisition_lengths: impl Iterator<Item = (SignalUid, HandleUid, usize)>,
    ) -> Result<Vec<HandleResultShape>> {
        let raw_length = raw_acquisition_lengths
            .map(|(signal, handle, length)| ((signal, handle), length))
            .collect::<HashMap<_, _>>();

        // Get fresh result shapes between calls, as this method may be called multiple times with different combined output
        // e.g. in the case of of chunking.
        let mut handle_result_shapes = self.handle_result_shapes.clone();

        // Extend the shape with an additional axis for the raw acquisition length.
        handle_result_shapes.iter_mut().for_each(|handle_shape| {
            if let Some(length) = raw_length.get(&(handle_shape.signal, handle_shape.handle)) {
                // Inject a new axis for the raw acquisition.
                handle_shape.shape.push(*length);
                handle_shape
                    .axis_names
                    .push(vec![self.raw_acquisition_axis_name]);
                handle_shape
                    .axis_values
                    .push(vec![AxisValues::Range(0..*length)]);
            }
        });

        // Group by handle and merge shapes while keeping ordering
        handle_result_shapes
            .drain(..)
            .fold(IndexMap::new(), |mut acc, shape| {
                acc.entry(shape.handle).or_insert_with(Vec::new).push(shape);
                acc
            })
            .into_values()
            .map(merge_shapes)
            .collect()
    }
}

/// Merge handle result shapes.
///
/// If there are multiple shapes for the same handle, and they are compatible, we combine them:
///
///   1. If respective acquisitions are inside match-case, match-case masks are combined
///   2. If not, a bigger shape is created where the last axis is for multiple handles.
///   3. If shapes are incompatible for any of the operations above, raises error.
fn merge_shapes(shapes: Vec<HandleResultShape>) -> Result<HandleResultShape> {
    if shapes.is_empty() {
        panic!("At least one shape should be provided for merging");
    }
    if shapes.len() == 1 {
        return Ok(shapes.into_iter().next().unwrap());
    }
    let n_shapes = shapes.len();
    let mut shapes = shapes.into_iter();
    let mut first_shape = shapes.next().unwrap();

    let handle = first_shape.handle;
    let combined_match_case_mask = &mut first_shape.match_case_mask;

    for shape in shapes {
        if shape.shape != first_shape.shape {
            let msg = format!(
                "Multiple acquire events with the same handle ('{}') and different result shapes are not allowed.",
                handle.0
            );
            return Err(Error::new(msg));
        }

        if shape.match_case_mask.is_empty() != combined_match_case_mask.is_empty() {
            let msg = format!(
                "Cannot use the same handle ('{}') in a different context if already used inside a match-case context",
                handle.0
            );
            return Err(Error::new(msg));
        }

        if shape.match_context_uid != first_shape.match_context_uid {
            let msg = format!(
                "Cannot use the same handle ('{}') in different match-case contexts",
                handle.0
            );
            return Err(Error::new(msg));
        }

        // Combine the match case masks by loop depth.
        let mut match_case_mask = shape.match_case_mask.into_iter().collect::<Vec<_>>();
        match_case_mask.sort_by_key(|(loop_depth, _)| *loop_depth);
        for (loop_depth, case_indices) in match_case_mask {
            let existing_case_indices: HashSet<_> =
                combined_match_case_mask[&loop_depth].iter().collect();
            let new_case_indices: HashSet<_> = case_indices.iter().collect();
            if !existing_case_indices.is_disjoint(&new_case_indices) {
                let msg = format!(
                    "Cannot use the same handle ('{}') multiple times in a single case context of a match section",
                    handle.0
                );
                return Err(Error::new(msg));
            }
            if let Some(combined_case_indices) = combined_match_case_mask.get_mut(&loop_depth) {
                combined_case_indices.extend(new_case_indices);
            }
        }
    }

    // Combine everything into the first shape.
    if combined_match_case_mask.is_empty() {
        first_shape.shape.push(n_shapes);
        first_shape.axis_names.push(vec![handle.into()]);
        first_shape
            .axis_values
            .push(vec![AxisValues::Range(0..n_shapes)]);
    } else {
        // The shapes are in match-case, so we don't add extra dimension for handle, just update the mask
        combined_match_case_mask
            .values_mut()
            .for_each(|case_indices| {
                case_indices.sort();
            });
    }
    Ok(first_shape)
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct HandleResultShape {
    pub(crate) handle: HandleUid,
    pub(crate) shape: Vec<usize>,
    pub(crate) axis_names: Vec<Vec<NamedId>>,
    pub(crate) axis_values: Vec<Vec<AxisValues>>,
    pub(crate) chunked_axis_index: Option<usize>,
    // For each loop depth, if the acquisition is within a match statement that is conditional on a sweep parameter, store the case indices of the match statement.
    pub(crate) match_case_mask: IndexMap<usize, Vec<usize>>,

    // The signal associated with the handle.
    signal: SignalUid,
    // Each handle is associated with at most one match statement, so we can store the match context here.
    match_context_uid: Option<usize>,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum AxisValues {
    Range(Range<usize>),
    Explicit(Arc<NumericArray>),
}

struct LoopEntry {
    count: NonZero<u32>,
    is_chunked: bool,
    axis_names: Vec<NamedId>,
    axis_points: Vec<AxisValues>,
    sweep_parameters: Vec<ParameterUid>,
}

#[derive(Debug, Clone, Default)]
struct MatchContext {
    counter: usize,
    uid_stack: Vec<usize>,
}

impl MatchContext {
    fn enter(&mut self) {
        self.counter += 1;
        self.uid_stack.push(self.counter);
    }

    fn pop(&mut self) {
        self.uid_stack.pop();
    }

    fn current(&self) -> Option<usize> {
        self.uid_stack.last().copied()
    }
}

struct ResultShapeExtractor<'a> {
    // --- Input data ---
    sweep_parameters: HashMap<ParameterUid, &'a SweepParameter>,

    // --- State during traversal ---
    // Loop stack to track nested loops and their associated sweep parameters and axis information.
    loop_stack: Vec<LoopEntry>,
    // Stack of active PRNG generators for nested PRNG loops.
    prng_stack: Vec<PrngGeneratorQccs>,
    // Match context stack to track nested match statements.
    match_context: MatchContext,
    // Sweep depth to case index for match statements.
    conditional_sweep_cases: Vec<(usize, usize)>,

    // --- Collected intermediate result shape information ---
    handle_result_shapes: Vec<HandleResultShape>,
    needs_raw_acquisition_injection: bool,
}

impl<'a> ResultShapeExtractor<'a> {
    fn new(sweep_parameters: impl Iterator<Item = &'a SweepParameter>) -> Self {
        Self {
            sweep_parameters: sweep_parameters.map(|sp| (sp.uid, sp)).collect(),
            loop_stack: Vec::new(),
            conditional_sweep_cases: Vec::new(),
            match_context: MatchContext::default(),
            handle_result_shapes: Vec::new(),
            needs_raw_acquisition_injection: false,
            prng_stack: Vec::new(),
        }
    }

    fn sweep_parameter_by_uid(&self, uid: &ParameterUid) -> Result<&'a SweepParameter> {
        self.sweep_parameters
            .get(uid)
            .copied()
            .ok_or_else(|| Error::new(format!("Sweep parameter with UID '{}' not found.", uid.0)))
    }
}

impl ResultShapeExtractor<'_> {
    fn visit_node(&mut self, node: &ExperimentNode) -> Result<()> {
        match &node.kind {
            Operation::AveragingLoop(op) => self.visit_averaging_loop(node, op)?,
            Operation::Sweep(op) => self.visit_regular_loop(node, op)?,
            Operation::PrngSetup(op) => self.visit_setup_prng(node, op)?,
            Operation::PrngLoop(op) => self.visit_prng_loop(node, op)?,
            Operation::Match(op) => self.visit_match(node, op)?,
            Operation::Acquire(op) => self.visit_acquire(node, op)?,
            _ => self.generic_visit(node)?,
        }
        Ok(())
    }

    fn generic_visit(&mut self, node: &ExperimentNode) -> Result<()> {
        for child in &node.children {
            self.visit_node(child)?;
        }
        Ok(())
    }

    fn visit_averaging_loop(&mut self, node: &ExperimentNode, op: &AveragingLoop) -> Result<()> {
        if op.acquisition_type == AcquisitionType::Raw {
            self.needs_raw_acquisition_injection = true;
        }
        // Averaging loop in non single shot mode is irrelevant for result shapes (does not introduce an axis)
        if op.averaging_mode != AveragingMode::SingleShot {
            self.generic_visit(node)?;
            return Ok(());
        }

        let entry = LoopEntry {
            count: op.count,
            is_chunked: false,
            axis_names: vec![op.uid.into()],
            axis_points: vec![AxisValues::Range(0..op.count.get() as usize)],
            sweep_parameters: vec![],
        };

        self.loop_stack.push(entry);
        self.generic_visit(node)?;
        self.loop_stack.pop();
        Ok(())
    }

    fn visit_regular_loop(&mut self, node: &ExperimentNode, op: &Sweep) -> Result<()> {
        let count = op.count;
        let is_chunked = op.is_chunked();
        let mut axis_names = Vec::new();
        let mut axis_points = Vec::new();
        let mut sweep_parameters = Vec::new();

        for sweep_parameter_uid in &op.direct_parameters {
            let param = self.sweep_parameter_by_uid(sweep_parameter_uid)?;
            // If the parameter has an explicit axis name, use it. Otherwise, use the parameter name as axis name.
            let axis_name = param
                .axis_name
                .unwrap_or_else(|| (*sweep_parameter_uid).into());

            axis_names.push(axis_name);
            axis_points.push(AxisValues::Explicit(Arc::clone(param.inner_values())));
            sweep_parameters.push(param.uid);
        }

        let entry = LoopEntry {
            count,
            is_chunked,
            axis_names,
            axis_points,
            sweep_parameters,
        };

        self.loop_stack.push(entry);
        self.generic_visit(node)?;
        self.loop_stack.pop();
        Ok(())
    }

    fn visit_setup_prng(
        &mut self,
        node: &ExperimentNode,
        op: &laboneq_dsl::operation::PrngSetup,
    ) -> Result<()> {
        let (seed, lower, upper) = (op.seed, 0u32, op.range - 1);
        let prng_generator =
            PrngGeneratorQccs::new(seed, lower, upper).map_err(|e| Error::new(e.to_string()))?;

        self.prng_stack.push(prng_generator);
        self.generic_visit(node)?;
        self.prng_stack.pop();
        Ok(())
    }

    fn visit_prng_loop(&mut self, node: &ExperimentNode, op: &PrngLoop) -> Result<()> {
        let prng = self
            .prng_stack
            .last_mut()
            .ok_or_else(|| Error::new("PRNG loop must be inside a PRNG setup"))?;
        let values = (0..op.count.get())
            .map(|_| prng.generate())
            .map(|v| v as i64)
            .collect::<Vec<_>>();

        let entry = LoopEntry {
            count: op.count,
            is_chunked: false,
            axis_names: vec![op.sample_uid.into()],
            axis_points: vec![AxisValues::Explicit(NumericArray::Integer64(values).into())],
            sweep_parameters: vec![],
        };

        self.loop_stack.push(entry);
        self.generic_visit(node)?;
        self.loop_stack.pop();
        Ok(())
    }

    fn visit_match(&mut self, node: &ExperimentNode, op: &Match) -> Result<()> {
        self.match_context.enter();
        if let MatchTarget::SweepParameter(param) = op.target {
            // Find the loop depth of the associated sweep parameter by looking through the loop stack.
            // Take the closest one in case of nested loops with the same parameter.
            let loop_depth = self
                .loop_stack
                .iter()
                .rev()
                .position(|item| item.sweep_parameters.contains(&param))
                .map(|pos| self.loop_stack.len() - 1 - pos)
                .ok_or_else(|| {
                    Error::new(format!(
                        "Sweep parameter '{}' not found in any loop.",
                        param.0
                    ))
                })?;

            // Sort the match cases according to the order of the sweep parameter values.
            let case_order = sort_match_cases_to_parameter(
                node.children.iter().map(|c| c.as_ref()),
                self.sweep_parameter_by_uid(&param)?,
            )?;
            for (case_index, case_node) in case_order.enumerate() {
                self.conditional_sweep_cases.push((loop_depth, case_index));
                let node = case_node?;
                self.generic_visit(node)?;
                self.conditional_sweep_cases.pop();
            }
        } else {
            self.generic_visit(node)?;
        }
        self.match_context.pop();
        Ok(())
    }

    fn visit_acquire(&mut self, _node: &ExperimentNode, op: &Acquire) -> Result<()> {
        let match_case_mask = self
            .conditional_sweep_cases
            .iter()
            .map(|(loop_depth, case_index)| (*loop_depth, vec![*case_index]))
            .collect();
        let shape = self
            .loop_stack
            .iter()
            .map(|entry| entry.count.get() as usize)
            .collect::<Vec<usize>>();
        let axis_names = self
            .loop_stack
            .iter()
            .map(|entry| entry.axis_names.clone())
            .collect::<Vec<_>>();
        let axis_values = self
            .loop_stack
            .iter()
            .map(|entry| entry.axis_points.clone())
            .collect::<Vec<_>>();

        // Assume that at most one loop is chunked.
        let chunked_axis_index = self.loop_stack.iter().position(|entry| entry.is_chunked);

        let result_shape = HandleResultShape {
            handle: op.handle,
            signal: op.signal,
            shape,
            axis_names,
            axis_values,
            chunked_axis_index,
            match_case_mask,
            match_context_uid: self.match_context.current(),
        };
        self.handle_result_shapes.push(result_shape);
        Ok(())
    }
}

fn sort_match_cases_to_parameter<'a>(
    nodes: impl Iterator<Item = &'a ExperimentNode>,
    sweep_parameter: &'a SweepParameter,
) -> Result<impl Iterator<Item = Result<&'a ExperimentNode>> + 'a> {
    let nodes = nodes.collect::<Vec<_>>();
    let state_to_index = nodes
        .iter()
        .enumerate()
        .filter_map(|(idx, child)| {
            if let Operation::Case(case) = &child.kind {
                Some((case.state.to_float(), idx))
            } else {
                None
            }
        })
        .collect::<HashMap<_, _>>();

    let out = sweep_parameter.values().map(move |sweep_value| {
        let target_idx = state_to_index.get(&sweep_value.to_float()).ok_or_else(|| {
            Error::new(format!(
                "Using a match statement for sweep parameter must cover all values.
        Match statement for parameter '{}' is missing a case for value '{}'.",
                sweep_parameter.uid.0, sweep_value,
            ))
        })?;
        Ok(nodes[*target_idx])
    });
    Ok(out)
}

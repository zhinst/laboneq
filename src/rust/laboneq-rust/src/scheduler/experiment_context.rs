// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use laboneq_dsl::{
    ExperimentNode,
    operation::Operation,
    types::{AcquisitionType, HandleUid, SignalUid},
};

use crate::scheduler::experiment::Experiment;

/// Immutable lookup/index data derived from the experiment
///
/// Any information that is added to the context shall not be
/// mutated during compilation process.
pub(crate) struct ExperimentContext {
    /// Map from acquisition handle UIDs to signal UIDs
    pub handle_to_signal: HashMap<HandleUid, SignalUid>,
    pub acquisition_type: AcquisitionType,
}

impl ExperimentContext {
    pub(crate) fn signal_by_handle(&self, handle: &HandleUid) -> Option<&SignalUid> {
        self.handle_to_signal.get(handle)
    }
}

/// Create an [`ExperimentContext`] from an [`Experiment`].
pub(crate) fn experiment_context_from_experiment(experiment: &Experiment) -> ExperimentContext {
    let mut context = ExperimentContext {
        handle_to_signal: HashMap::new(),
        acquisition_type: AcquisitionType::Integration,
    };
    visit_node(&experiment.root, &mut context);
    context
}

fn visit_node(node: &ExperimentNode, context: &mut ExperimentContext) {
    match &node.kind {
        Operation::Acquire(obj) => {
            context.handle_to_signal.insert(obj.handle, obj.signal);
        }
        Operation::AveragingLoop(obj) => {
            context.acquisition_type = obj.acquisition_type;
            for child in &node.children {
                visit_node(child, context);
            }
        }
        _ => {
            for child in &node.children {
                visit_node(child, context);
            }
        }
    }
}

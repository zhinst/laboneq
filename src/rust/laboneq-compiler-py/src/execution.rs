// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::num::NonZeroU32;
use std::sync::Arc;

use numeric_array::NumericArray;

use laboneq_common::named_id::NamedId;
use laboneq_common::types::Literal;

use laboneq_dsl::ExperimentNode;
use laboneq_dsl::operation::{NearTimeCallback, Operation, SetNode, Sweep, ValueEntry};
use laboneq_dsl::types::{ParameterUid, ValueOrParameter};

use crate::error::Result;
use crate::experiment::Experiment;

#[derive(Debug, Clone)]
pub(crate) enum Statement {
    SetParameter {
        parameter: ParameterUid,
        values: Arc<NumericArray>,
    },
    /// Loop a fixed number of times.
    Loop {
        count: NonZeroU32,
        body: Vec<Statement>,
    },
    /// Execute a real-time program.
    ExecRealTime,
    /// Execute a user defined callback.
    ExecCallback {
        callback_id: NamedId,
        args: Vec<ValueEntry>,
    },
    /// Set a node value.
    SetNode {
        path: NamedId,
        value: ValueOrParameter<Literal>,
    },
}

/// Creates a near-time execution for Controller.
pub(crate) fn create_execution(experiment: &Experiment) -> Result<Vec<Statement>> {
    NearTimeExecutionBuilder::build(experiment)
}

struct NearTimeExecutionBuilder<'a> {
    experiment: &'a Experiment,
}

impl<'a> NearTimeExecutionBuilder<'a> {
    fn build(experiment: &Experiment) -> Result<Vec<Statement>> {
        NearTimeExecutionBuilder { experiment }.visit_node(&experiment.root)
    }

    fn visit_node(&mut self, node: &'a ExperimentNode) -> Result<Vec<Statement>> {
        match &node.kind {
            Operation::Sweep(sweep) => self.visit_sweep(node, sweep),
            Operation::NearTimeCallback(callback) => self.visit_callback(callback),
            Operation::SetNode(set_node) => self.visit_set_node(set_node),
            Operation::RealTimeBoundary => Ok(vec![Statement::ExecRealTime]),
            _ => self.generic_visit(node),
        }
    }

    fn generic_visit(&mut self, node: &'a ExperimentNode) -> Result<Vec<Statement>> {
        let mut statements = Vec::with_capacity(node.children.len());
        for child in &node.children {
            statements.extend(self.visit_node(child)?);
        }
        Ok(statements)
    }

    fn visit_sweep(&mut self, node: &'a ExperimentNode, op: &'a Sweep) -> Result<Vec<Statement>> {
        let mut body = Vec::with_capacity(op.parameters.len() + 1);
        for parameter in &op.parameters {
            let sweep_param = self.experiment.get_sweep_parameter(parameter)?;
            body.push(Statement::SetParameter {
                parameter: *parameter,
                values: Arc::clone(sweep_param.inner_values()),
            });
        }

        body.extend(self.generic_visit(node)?);

        let statement = Statement::Loop {
            count: op.count,
            body,
        };
        Ok(vec![statement])
    }

    fn visit_callback(&mut self, op: &'a NearTimeCallback) -> Result<Vec<Statement>> {
        let statement = Statement::ExecCallback {
            callback_id: op.callback_id,
            args: op.args.clone(),
        };
        Ok(vec![statement])
    }

    fn visit_set_node(&mut self, op: &'a SetNode) -> Result<Vec<Statement>> {
        let statement = Statement::SetNode {
            path: op.path,
            value: op.value.clone(),
        };
        Ok(vec![statement])
    }
}

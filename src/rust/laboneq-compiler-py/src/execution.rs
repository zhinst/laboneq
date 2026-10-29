// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::num::NonZeroU32;

use laboneq_common::named_id::NamedId;
use laboneq_common::types::Literal;

use laboneq_dsl::ExperimentNode;
use laboneq_dsl::operation::{NearTimeCallback, Operation, SetNode, Sweep, ValueEntry};
use laboneq_dsl::types::{ParameterUid, ValueOrParameter};

use crate::error::Result;

#[derive(Debug, Clone)]
pub(crate) struct Execution {
    pub statements: Vec<Statement>,
}

#[derive(Debug, Clone)]
pub(crate) enum Statement {
    SetParameter {
        parameter: ParameterUid,
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
pub(crate) fn create_execution(root: &ExperimentNode) -> Result<Execution> {
    Ok(Execution {
        statements: NearTimeExecutionBuilder::build(root)?,
    })
}

struct NearTimeExecutionBuilder {}

impl NearTimeExecutionBuilder {
    fn build(root: &ExperimentNode) -> Result<Vec<Statement>> {
        NearTimeExecutionBuilder {}.visit_node(root)
    }

    fn visit_node(&mut self, node: &ExperimentNode) -> Result<Vec<Statement>> {
        match &node.kind {
            Operation::Sweep(sweep) => self.visit_sweep(node, sweep),
            Operation::NearTimeCallback(callback) => self.visit_callback(callback),
            Operation::SetNode(set_node) => self.visit_set_node(set_node),
            Operation::RealTimeBoundary => Ok(vec![Statement::ExecRealTime]),
            _ => self.generic_visit(node),
        }
    }

    fn generic_visit(&mut self, node: &ExperimentNode) -> Result<Vec<Statement>> {
        let mut statements = Vec::with_capacity(node.children.len());
        for child in &node.children {
            statements.extend(self.visit_node(child)?);
        }
        Ok(statements)
    }

    fn visit_sweep(&mut self, node: &ExperimentNode, op: &Sweep) -> Result<Vec<Statement>> {
        let mut body = Vec::with_capacity(op.parameters.len() + 1);
        for parameter in &op.parameters {
            body.push(Statement::SetParameter {
                parameter: *parameter,
            });
        }

        body.extend(self.generic_visit(node)?);

        let statement = Statement::Loop {
            count: op.count,
            body,
        };
        Ok(vec![statement])
    }

    fn visit_callback(&mut self, op: &NearTimeCallback) -> Result<Vec<Statement>> {
        let statement = Statement::ExecCallback {
            callback_id: op.callback_id,
            args: op.args.clone(),
        };
        Ok(vec![statement])
    }

    fn visit_set_node(&mut self, op: &SetNode) -> Result<Vec<Statement>> {
        let statement = Statement::SetNode {
            path: op.path,
            value: op.value.clone(),
        };
        Ok(vec![statement])
    }
}

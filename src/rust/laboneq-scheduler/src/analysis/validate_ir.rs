// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use crate::ScheduledNode;
use crate::error::{Error, Result};
use crate::ir::{IrKind, MatchTarget};

/// Validate the IR for correctness.
///
/// * Checks that no acquisitions are present within match statements
///   when matching against targets other than sweep parameters.
pub fn validate_ir(node: &ScheduledNode) -> Result<()> {
    let mut ctx = ValidationContext::new();
    validate_ir_impl(node, &mut ctx)?;
    Ok(())
}

struct ValidationContext {
    contains_acquisitions: bool,
}

impl ValidationContext {
    fn new() -> Self {
        Self {
            contains_acquisitions: false,
        }
    }
}

fn validate_ir_impl(node: &ScheduledNode, ctx: &mut ValidationContext) -> Result<()> {
    match &node.kind {
        IrKind::Match(obj) => {
            let mut context = ValidationContext::new();
            let disallow_acquisitions = !matches!(obj.target, MatchTarget::SweepParameter(_));
            for child in node.children.iter() {
                validate_ir_impl(&child.node, &mut context)?;
                if disallow_acquisitions && context.contains_acquisitions {
                    let msg = format!(
                        "Acquisitions are not allowed within match statements when matching against {}",
                        obj.target.description()
                    );
                    return Err(Error::new(&msg));
                }
            }
        }
        IrKind::Acquire(_) => {
            ctx.contains_acquisitions = true;
        }
        _ => {
            for child in node.children.iter() {
                validate_ir_impl(&child.node, ctx)?;
            }
        }
    }
    Ok(())
}

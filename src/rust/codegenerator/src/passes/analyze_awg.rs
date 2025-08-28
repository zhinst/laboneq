// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use crate::ir::{IrNode, NodeKind, PpcDevice, experiment::Handle};
use std::sync::Arc;

#[derive(Default)]
pub struct AwgCompilationInfo {
    ppc_device: Option<Arc<PpcDevice>>,
    feedback_handles: Vec<Handle>,
}

impl AwgCompilationInfo {
    pub fn has_readout_feedback(&self) -> bool {
        !self.feedback_handles.is_empty()
    }

    pub fn ppc_device(&self) -> Option<&Arc<PpcDevice>> {
        self.ppc_device.as_ref()
    }

    pub fn feedback_handles(&self) -> &Vec<Handle> {
        &self.feedback_handles
    }

    fn add_ppc_device(&mut self, ppc_device: &Arc<PpcDevice>) {
        if self.ppc_device.is_none() {
            self.ppc_device = Some(Arc::clone(ppc_device));
        } else if let Some(unique_ppc) = &self.ppc_device {
            assert_eq!(
                unique_ppc, ppc_device,
                "Internal error: Multiple SHFPPC devices found in the same AWG. \
                Only a single device and a single channel is supported."
            )
        }
    }
}

fn traverse_awg_ir(node: &IrNode, info: &mut AwgCompilationInfo) {
    match node.data() {
        NodeKind::Match(ob) => {
            if let Some(handle) = ob.handle.as_ref() {
                info.feedback_handles.push(handle.clone());
            }
        }
        NodeKind::PpcSweepStep(ob) => {
            info.add_ppc_device(&ob.ppc_device);
        }
        _ => {}
    }
    for child in node.iter_children() {
        traverse_awg_ir(child, info);
    }
}

pub fn analyze_awg_ir(node: &IrNode) -> AwgCompilationInfo {
    let mut info = AwgCompilationInfo::default();
    traverse_awg_ir(node, &mut info);
    info
}

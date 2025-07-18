// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use crate::ir::{IrNode, NodeKind, PpcDevice};
use std::sync::Arc;

#[derive(Default)]
pub struct AwgCompilationInfo {
    ppc_device: Option<Arc<PpcDevice>>,
    has_readout_feedback: bool,
}

impl AwgCompilationInfo {
    pub fn has_readout_feedback(&self) -> bool {
        self.has_readout_feedback
    }

    pub fn ppc_device(&self) -> Option<&Arc<PpcDevice>> {
        self.ppc_device.as_ref()
    }

    fn add_readout_feedback(&mut self) {
        self.has_readout_feedback = true;
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
            if ob.handle.is_some() {
                info.add_readout_feedback();
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

// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use crate::Result;
use crate::ir;
use crate::ir::SectionId;
use anyhow::anyhow;
use std::collections::HashSet;

fn handle_prng_recursive(
    node: &ir::IrNode,
    cut_points: &mut HashSet<ir::Samples>,
    parent_prng_setup_section: Option<SectionId>,
    active_prng_sample: &mut Option<String>,
) -> Result<()> {
    let mut parent_prng_setup_section_here = parent_prng_setup_section;
    for child in node.iter_children() {
        match child.data() {
            ir::NodeKind::SetupPrng(data) => {
                if let Some(ref section_id) = parent_prng_setup_section_here {
                    if section_id != &data.section_info.id {
                        return Err(anyhow::anyhow!(
                            "PRNG setup already exists while processing section {}",
                            data.section_info.name
                        )
                        .into());
                    }
                }
                parent_prng_setup_section_here = Some(data.section_info.id);
                cut_points.insert(*child.offset());
            }
            ir::NodeKind::DropPrngSetup => {
                parent_prng_setup_section_here = None;
            }
            ir::NodeKind::Loop(ob) => {
                let mut reset_active_prng_sample = false;
                if let Some(sample_name) = &ob.prng_sample {
                    if let Some(other_sample) = active_prng_sample {
                        return Err(anyhow!(
                            "Nested PRNG loops are not allowed: '{sample_name}' (current) vs '{other_sample}' (existing)"
                        )
                        .into());
                    }
                    *active_prng_sample = Some(sample_name.clone());
                    reset_active_prng_sample = true;
                }
                handle_prng_recursive(
                    child,
                    cut_points,
                    parent_prng_setup_section_here,
                    active_prng_sample,
                )?;
                if reset_active_prng_sample {
                    *active_prng_sample = None;
                }
            }
            ir::NodeKind::Match(data) => {
                if let Some(sample_name) = &data.prng_sample {
                    if data.prng_sample != *active_prng_sample {
                        return Err(anyhow!(
                        "In section '{}': cannot match PRNG sample '{}' here. The only available PRNG sample is '{}'.",
                        data.section_info.name,
                        sample_name,
                                                  active_prng_sample
                            .as_ref()
                            .unwrap_or(&String::from("<empty>")),
                    ).into());
                    }
                }
                handle_prng_recursive(
                    child,
                    cut_points,
                    parent_prng_setup_section_here,
                    active_prng_sample,
                )?;
            }
            _ => {
                handle_prng_recursive(
                    child,
                    cut_points,
                    parent_prng_setup_section_here,
                    active_prng_sample,
                )?;
            }
        }
    }
    Ok(())
}

pub fn handle_prng(node: &ir::IrNode, cut_points: &mut HashSet<ir::Samples>) -> Result<()> {
    let mut active_prng_sample = Option::<String>::None;
    handle_prng_recursive(node, cut_points, None, &mut active_prng_sample)
}

// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use crate::Result;
use crate::ir;
use crate::ir::SectionId;
use anyhow::anyhow;
use std::collections::HashSet;

pub fn handle_prng_recursive<'a>(
    node: &'a ir::IrNode,
    cut_points: &mut HashSet<ir::Samples>,
    parent_prng_setup_section: Option<SectionId>,
    active_prng_sample: &mut Option<String>,
    active_loop: Option<&'a ir::SectionInfo>,
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
                handle_prng_recursive(
                    child,
                    cut_points,
                    parent_prng_setup_section_here,
                    active_prng_sample,
                    ob.section_info.as_ref().into(),
                )?;
            }
            ir::NodeKind::LoopIteration(data) => {
                let mut reset_active_prng_sample = false;
                if let Some(sample_name) = &data.prng_sample {
                    if let Some(other_sample) = active_prng_sample {
                        return Err(anyhow!(
                        "In section '{}': Can't draw sample '{}' from PRNG, when other sample '{}' is still required at the same time",
                        active_loop
                            .as_ref()
                            .map_or("unknown", |l| l.name.as_str()),
                        sample_name,
                        other_sample
                    ).into());
                    }
                    *active_prng_sample = Some(sample_name.clone());
                    reset_active_prng_sample = true;
                }
                handle_prng_recursive(
                    child,
                    cut_points,
                    parent_prng_setup_section_here,
                    active_prng_sample,
                    active_loop,
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
                    None,
                )?;
            }
            _ => {
                handle_prng_recursive(
                    child,
                    cut_points,
                    parent_prng_setup_section_here,
                    active_prng_sample,
                    None,
                )?;
            }
        }
    }
    Ok(())
}

pub fn handle_prng(node: &ir::IrNode, cut_points: &mut HashSet<ir::Samples>) -> Result<()> {
    let mut active_prng_sample = Option::<String>::None;
    handle_prng_recursive(node, cut_points, None, &mut active_prng_sample, None)
}

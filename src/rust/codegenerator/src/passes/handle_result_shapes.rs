// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use crate::Result;
use crate::context::CodeGenContext;
use crate::device_traits::scope_memory_size_samples;
use crate::ir::compilation_job::AwgCore;
use crate::ir::experiment::{AcquisitionType, Handle};
use crate::ir::{AcquirePulse, IrNode, NodeKind, Samples};
use crate::passes::handle_measure_times::IntegrationLength;
use crate::result::ResultSource;

use laboneq_dsl::types::{AveragingMode, SignalUid};
use laboneq_error::{bail, bail_resource_usage};
use std::collections::HashMap;

pub(crate) struct AwgMeasurementShapes {
    pub result_length: Option<usize>,
    pub result_handle_maps: HashMap<ResultSource, Vec<Vec<Handle>>>,
}

/// Calculate the shapes of the results for all acquisitions, and check for memory limits.
pub(crate) fn calculate_measure_shapes(
    node: &IrNode,
    awg: &AwgCore,
    integration_lengths: &[IntegrationLength],
    ctx: &CodeGenContext,
) -> Result<Option<AwgMeasurementShapes>> {
    let acquisitions = AcquisitionCounter::run(node);
    if acquisitions.is_empty() {
        // No acquisitions, so no result length to calculate.
        return Ok(None);
    }
    let result_length = calculate_result_length(node, awg, integration_lengths, ctx)?;
    let result_handle_maps = construct_result_handle_maps(&acquisitions, awg, ctx);

    Ok(Some(AwgMeasurementShapes {
        result_length,
        result_handle_maps,
    }))
}

#[derive(Debug, Clone)]
struct Acquisition {
    handle: Handle,
    signal: SignalUid,
}

struct AcquisitionCounter {
    acquisitions: HashMap<Samples, Vec<Acquisition>>,
}

impl AcquisitionCounter {
    /// Collect acquisitions and return them sorted by their absolute start time.
    fn run(node: &IrNode) -> Vec<Vec<Acquisition>> {
        let mut counter = AcquisitionCounter {
            acquisitions: HashMap::new(),
        };
        counter.visit_node(node, 0);
        // Sort acquisitions by their absolute start time.
        let mut acquisitions: Vec<(Samples, Vec<Acquisition>)> =
            counter.acquisitions.into_iter().collect();
        acquisitions.sort_by_key(|(start, _)| *start);
        acquisitions.into_iter().map(|(_, acq)| acq).collect()
    }

    fn push_acquisition(&mut self, absolute_start: Samples, acquisition: Acquisition) {
        let count = self.acquisitions.entry(absolute_start).or_default();
        count.push(acquisition);
    }

    fn visit_node(&mut self, node: &IrNode, offset: Samples) {
        let absolute_start = offset + *node.offset();
        match node.data() {
            NodeKind::AcquirePulse(ob) => self.visit_acquire(ob, absolute_start),
            NodeKind::Acquire(_) => panic!("Unexpected acquire node in IR"),
            NodeKind::Loop(_) => self.visit_loop_node(node, absolute_start),
            _ => {
                for child in node.iter_children() {
                    self.visit_node(child, absolute_start);
                }
            }
        }
    }

    fn visit_acquire(&mut self, acquire: &AcquirePulse, absolute_start: Samples) {
        let acquisition = Acquisition {
            handle: acquire.handle.clone(),
            signal: acquire.signal.uid,
        };
        self.push_acquisition(absolute_start, acquisition);
    }

    fn visit_loop_node(&mut self, loop_node: &IrNode, absolute_start: Samples) {
        if let NodeKind::Loop(ob) = loop_node.data() {
            // Unroll compressed loops to get real number of acquisitions.
            // The unroll logic should actually apply for all loops that are compressed and are not averaging loops.
            // Currently only PRNG behaves like this, so we only handle it here.
            if ob.prng_sample.is_some() {
                let acquisition_so_far = std::mem::take(&mut self.acquisitions);
                self.acquisitions = HashMap::new();
                for child in loop_node.iter_children() {
                    self.visit_node(child, 0);
                }
                let acquisition_after = std::mem::take(&mut self.acquisitions);
                self.acquisitions = acquisition_so_far;
                for (acquisition_start, acquisitions) in acquisition_after.into_iter() {
                    for iteration in 0..ob.count {
                        let abs_start_iteration = absolute_start
                            + (ob.length / ob.count as Samples) * iteration as Samples;
                        let start_abs = abs_start_iteration + acquisition_start;
                        for acquisition in &acquisitions {
                            self.push_acquisition(start_abs, acquisition.clone());
                        }
                    }
                }
            } else {
                for child in loop_node.iter_children() {
                    self.visit_node(child, absolute_start);
                }
            }
        } else {
            panic!("Expected a loop node");
        }
    }
}

/// Calculate the length of the result vector for integrated acquisitions, and check for memory limits for raw acquisitions.
fn calculate_result_length(
    node: &IrNode,
    awg: &AwgCore,
    integration_lengths: &[IntegrationLength],
    ctx: &CodeGenContext,
) -> Result<Option<usize>> {
    let acquisitions = AcquisitionCounter::run(node);
    if acquisitions.is_empty() {
        // No acquisitions, so no result length to calculate.
        return Ok(None);
    }
    let device_traits = awg.device_kind().traits();
    let n_acq = acquisitions.len();
    let result_length = if matches!(ctx.averaging_mode, AveragingMode::SingleShot) {
        n_acq * ctx.averaging_count.get() as usize
    } else {
        // For integrated acquisitions, the length of the result is determined by the number of acquisitions that are integrated together.
        n_acq
    };

    // Validate result length against limits
    for measurement in integration_lengths {
        if matches!(ctx.acquisition_type, AcquisitionType::RAW) {
            if let Some(max_segments) = device_traits.scope_max_segments
                && result_length > max_segments as usize
            {
                bail!(
                    "A maximum of {} raw result(s) is supported per real-time execution.",
                    max_segments
                );
            }

            let raw_acquire_samples: usize = measurement
                .duration()
                .try_into()
                .expect("Expected measurement length to fit usize");
            let scope_memory_consumption = result_length * raw_acquire_samples;
            let scope_memory_size_samples =
                scope_memory_size_samples(*awg.device_kind(), awg.is_shfqc);
            if scope_memory_consumption > scope_memory_size_samples {
                bail!(
                    "The total size of the requested raw traces exceeds the instrument's memory capacity."
                )
            }
        } else if !ctx.settings.ignore_resource_exhaustion {
            let max_result_vector_length = device_traits
                .max_result_vector_length
                .expect("max_result_vector_length should be set for this device");
            if result_length > max_result_vector_length {
                let usage = result_length as f64 / max_result_vector_length as f64;
                bail_resource_usage!(
                    "Result length for awg '{}' on device '{}'.",
                    awg.uid,
                    awg.device.uid(),
                , usage = usage);
            }
        }
    }
    Ok(Some(result_length))
}

fn construct_result_handle_maps(
    acquisitions: &[Vec<Acquisition>],
    awg: &AwgCore,
    ctx: &CodeGenContext,
) -> HashMap<ResultSource, Vec<Vec<Handle>>> {
    let mut result_handle_maps: HashMap<ResultSource, Vec<Vec<Handle>>> = HashMap::new();
    let mut signal_to_result_source = HashMap::new();
    for acquires in acquisitions {
        let mut result_map_for_this_round: HashMap<ResultSource, Vec<Handle>> = HashMap::new();
        for sig in &awg.signals {
            if sig.is_output() {
                continue;
            }
            let Some(integration_units) = ctx.integration_units_for_signal(sig.uid) else {
                continue;
            };
            let integrator_idx = match ctx.acquisition_type {
                AcquisitionType::RAW => None,
                _ => Some(integration_units[0]),
            };
            let result_source = ResultSource {
                device_id: awg.device.uid().clone(),
                awg_id: awg.uid,
                integrator_idx,
            };
            result_map_for_this_round
                .entry(result_source.clone())
                .or_default();
            signal_to_result_source.insert(&sig.uid, result_source);
        }
        for acq in acquires {
            let _ = result_map_for_this_round
                .entry(signal_to_result_source.get(&acq.signal).unwrap().clone())
                .and_modify(|entry| {
                    entry.push(acq.handle.clone());
                });
        }
        result_map_for_this_round
            .into_iter()
            .for_each(|(result_source, val)| {
                result_handle_maps
                    .entry(result_source)
                    .or_default()
                    .push(val);
            });
    }

    result_handle_maps
}

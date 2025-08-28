// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use crate::ir::compilation_job::{AwgCore, AwgKey, Signal};
use crate::ir::experiment::Handle;
use crate::ir::{IrNode, NodeKind, Samples};
use crate::{Error, Result};
use std::collections::BTreeMap;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub enum FeedbackRegisterAllocation {
    Local,
    Global { register: u16 },
}

#[derive(Debug, Clone)]
struct HandleInfo<'a> {
    signal: &'a str,
    global: bool,
    // Whether the handle is used for feedback or not
    is_feedback: bool,
    // Timestamps of the each acquisition for this handle
    timestamps: Vec<Samples>,
}

fn collect_handles<'a>(
    node: &'a IrNode,
    offset: &Samples,
    handles: &mut HashMap<Handle, HandleInfo<'a>>,
) -> Result<()> {
    match node.data() {
        NodeKind::AcquirePulse(ob) => {
            if let Some(handle) = handles.get_mut(&ob.handle) {
                handle.timestamps.push(offset + *node.offset());
            } else {
                handles.insert(
                    ob.handle.clone(),
                    HandleInfo {
                        signal: ob.signal.uid.as_str(),
                        global: false,
                        is_feedback: false,
                        timestamps: vec![offset + *node.offset()],
                    },
                );
            }
            return Ok(());
        }
        NodeKind::Match(ob) => {
            if let Some(handle) = &ob.handle {
                if let Some(handle_info) = handles.get_mut(handle) {
                    handle_info.global = !ob.local;
                    handle_info.is_feedback = true;
                } else {
                    return Err(Error::new(&format!(
                        "Handle '{handle}' not found for match.",
                    )));
                }
            }
            for child in node.iter_children() {
                collect_handles(child, node.offset(), handles)?;
            }
        }
        NodeKind::Loop(ob) => {
            // Unroll compressed loops to get real number of acquisitions.
            // The unroll logic should actually apply for all loops that are compressed and are not averaging loops.
            // Currently only PRNG behaves like this, so we only handle it here.
            if ob.prng_sample.is_some() {
                let mut new_handles = HashMap::new();
                for child in node.iter_children() {
                    collect_handles(child, &(node.offset() + offset), &mut new_handles)?;
                }
                for (handle, mut info) in new_handles.into_iter() {
                    let mut timestamps = vec![];
                    for timestamp in &info.timestamps {
                        for i in 0..ob.count as Samples {
                            let start_abs = (ob.length / ob.count as Samples) * i + timestamp;
                            timestamps.push(start_abs);
                        }
                    }
                    if let Some(existing_info) = handles.get_mut(&handle) {
                        existing_info.timestamps.extend(timestamps);
                    } else {
                        info.timestamps.extend(timestamps);
                        handles.insert(handle, info);
                    }
                }
            } else {
                for child in node.iter_children() {
                    collect_handles(child, &(node.offset() + offset), handles)?;
                }
            }
        }
        _ => {
            for child in node.iter_children() {
                collect_handles(child, &(node.offset() + offset), handles)?;
            }
        }
    }
    Ok(())
}

/// Allocate a feedback register on the PQSC.
///
/// This function allocates feedback registers for the given AWGs based on the
/// signals that are marked for feedback. The allocation is done in the order of the AWGs
/// and the first AWG that requires a feedback register will get the first one.
///
/// Each QA AWG can write to at most one feedback register. The feedback register
/// corresponds to the `result_address` in the `startQA` command and is zero based.
/// The bits in the register are assigned by the instrument following the integrator
/// order (the actual bit field layout will be elaborated later).
///
/// The maximum number of feedback registers is 32, when exceeding this limit,
/// an error is returned.
fn allocate_feedback_registers<'a>(
    awgs: &[&'a AwgCore],
    handles: &HashMap<Handle, HandleInfo<'a>>,
) -> Result<HashMap<AwgKey, FeedbackRegisterAllocation>> {
    let mut signal_to_handle = HashMap::new();
    for info in handles.values() {
        if !info.is_feedback {
            continue;
        }
        signal_to_handle.insert(info.signal, info);
    }
    const PQSC_FEEDBACK_REGISTER_COUNT: u16 = 32;
    let mut register: u16 = 0;
    let mut target_feedback_registers: HashMap<AwgKey, FeedbackRegisterAllocation> = HashMap::new();
    for awg in awgs.iter() {
        for signal in &awg.signals {
            if let Some(handle) = signal_to_handle.get(signal.uid.as_str()) {
                if target_feedback_registers.contains_key(&awg.key()) {
                    continue;
                }
                if !handle.global {
                    target_feedback_registers.insert(awg.key(), FeedbackRegisterAllocation::Local);
                } else {
                    if register >= PQSC_FEEDBACK_REGISTER_COUNT {
                        return Err(Error::new(&format!(
                            "Cannot allocate feedback register. \
                            All {PQSC_FEEDBACK_REGISTER_COUNT} registers of the PQSC are already allocated.",
                        )));
                    }
                    target_feedback_registers
                        .insert(awg.key(), FeedbackRegisterAllocation::Global { register });
                    register += 1;
                }
            }
        }
    }
    Ok(target_feedback_registers)
}

fn evaluate_simultaneous_acquires(
    handles: &HashMap<Handle, HandleInfo>,
) -> Result<Vec<Vec<Acquisition>>> {
    if handles.is_empty() {
        return Ok(vec![]);
    }
    // Simultaneous acquires must be ordered by timestamp
    let mut sim_acquires: BTreeMap<Samples, Vec<Acquisition>> = BTreeMap::new();
    for (handle, info) in handles.iter() {
        for timestamp in &info.timestamps {
            sim_acquires
                .entry(*timestamp)
                .or_default()
                .push(Acquisition {
                    signal: info.signal.to_string(),
                    handle: handle.clone(),
                });
        }
    }
    Ok(sim_acquires.into_values().collect())
}

pub struct FeedbackSource<'a> {
    pub signal: &'a Signal,
    pub awg_key: AwgKey,
}

pub struct Acquisition {
    pub handle: Handle,
    pub signal: String,
}

pub struct FeedbackConfig<'a> {
    target_feedback_registers: HashMap<AwgKey, FeedbackRegisterAllocation>,
    feedback_sources: HashMap<Handle, FeedbackSource<'a>>,
    acquisitions: Vec<Vec<Acquisition>>,
}

impl<'a> FeedbackConfig<'a> {
    fn new(
        target_feedback_registers: HashMap<AwgKey, FeedbackRegisterAllocation>,
        feedback_sources: HashMap<Handle, FeedbackSource<'a>>,
        acquisitions: Vec<Vec<Acquisition>>,
    ) -> Self {
        Self {
            target_feedback_registers,
            feedback_sources,
            acquisitions,
        }
    }

    /// Feedback register allocated for the given AWG.
    pub fn target_feedback_register(
        &self,
        awg_key: &AwgKey,
    ) -> Option<&FeedbackRegisterAllocation> {
        self.target_feedback_registers.get(awg_key)
    }

    /// Feedback source information for the given handle (QA signal information).
    pub fn feedback_source(&self, handle: &Handle) -> Option<&FeedbackSource<'a>> {
        self.feedback_sources.get(handle)
    }

    /// Iterator over all handles that are configured for feedback.
    pub fn handles(&self) -> impl Iterator<Item = &Handle> {
        self.feedback_sources.keys()
    }

    pub fn into_acquisitions(&mut self) -> Vec<Vec<Acquisition>> {
        std::mem::take(&mut self.acquisitions)
    }
}

/// Collect the feedback configuration from the given IR node for the given AWGs.
///
/// The function assigns feedback registers to the AWGs, where the assignment is based on the
/// order of the AWGs. It will also collect the signal information associated with each measurement handle.
pub fn collect_feedback_config<'a>(
    node: &IrNode,
    awgs: &[&'a AwgCore],
) -> Result<FeedbackConfig<'a>> {
    let mut handle_to_signal = HashMap::new();
    collect_handles(node, node.offset(), &mut handle_to_signal)?;
    let target_feedback_registers = allocate_feedback_registers(awgs, &handle_to_signal)?;
    let acquisitions = evaluate_simultaneous_acquires(&handle_to_signal)?;
    let mut signal_lookup: HashMap<&str, (&Signal, AwgKey)> = HashMap::new();
    for awg in awgs.iter() {
        for signal in &awg.signals {
            signal_lookup.insert(signal.uid.as_str(), (signal, awg.key()));
        }
    }
    let mut feedback_sources = HashMap::new();
    for (handle, handle_info) in handle_to_signal {
        let signal_info = signal_lookup
            .get(handle_info.signal)
            .expect("Internal error: Expected signal for handle");
        feedback_sources.insert(
            handle,
            FeedbackSource {
                signal: signal_info.0,
                awg_key: signal_info.1.clone(),
            },
        );
    }
    let config = FeedbackConfig::new(target_feedback_registers, feedback_sources, acquisitions);
    Ok(config)
}

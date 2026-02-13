// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use crate::ir::compilation_job::{AwgCore, AwgKey, DeviceUid, Signal};
use crate::ir::experiment::Handle;
use crate::ir::{IrNode, NodeKind, Samples, SignalUid};
use crate::result::IntegrationUnitAllocation;
use crate::{
    Error, FeedbackRegister, FeedbackRegisterLayout, Result, SingleFeedbackRegisterLayoutItem,
};
use std::collections::BTreeMap;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub(crate) enum FeedbackRegisterAllocation {
    Local,
    Global { register: u16 },
}

#[derive(Debug, Clone)]
struct HandleInfo {
    signal: SignalUid,
    global: bool,
    // Whether the handle is used for feedback or not
    is_feedback: bool,
    // Timestamps of the each acquisition for this handle
    timestamps: Vec<Samples>,
}

fn collect_handles(
    node: &IrNode,
    offset: &Samples,
    handles: &mut HashMap<Handle, HandleInfo>,
) -> Result<()> {
    match node.data() {
        NodeKind::AcquirePulse(ob) => {
            if let Some(handle) = handles.get_mut(&ob.handle) {
                handle.timestamps.push(offset + *node.offset());
            } else {
                handles.insert(
                    ob.handle.clone(),
                    HandleInfo {
                        signal: ob.signal.uid,
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
                    return Err(Error::new(format!(
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
fn allocate_feedback_registers(
    awgs: &[AwgCore],
    handles: &HashMap<Handle, HandleInfo>,
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
            if let Some(handle) = signal_to_handle.get(&signal.uid) {
                if target_feedback_registers.contains_key(&awg.key()) {
                    continue;
                }
                if !handle.global {
                    target_feedback_registers.insert(awg.key(), FeedbackRegisterAllocation::Local);
                } else {
                    if register >= PQSC_FEEDBACK_REGISTER_COUNT {
                        return Err(Error::new(format!(
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
                    signal: info.signal,
                    handle: handle.clone(),
                });
        }
    }
    Ok(sim_acquires.into_values().collect())
}

pub(crate) struct FeedbackSource {
    pub signal: SignalUid,
    pub awg_key: AwgKey,
}

pub struct Acquisition {
    pub handle: Handle,
    pub signal: SignalUid,
}

pub(crate) struct FeedbackConfig {
    target_feedback_registers: HashMap<AwgKey, FeedbackRegisterAllocation>,
    feedback_sources: HashMap<Handle, FeedbackSource>,
    acquisitions: Vec<Vec<Acquisition>>,
}

impl FeedbackConfig {
    fn new(
        target_feedback_registers: HashMap<AwgKey, FeedbackRegisterAllocation>,
        feedback_sources: HashMap<Handle, FeedbackSource>,
        acquisitions: Vec<Vec<Acquisition>>,
    ) -> Self {
        Self {
            target_feedback_registers,
            feedback_sources,
            acquisitions,
        }
    }

    /// Feedback register allocated for the given AWG.
    pub(crate) fn target_feedback_register(
        &self,
        awg_key: &AwgKey,
    ) -> Option<&FeedbackRegisterAllocation> {
        self.target_feedback_registers.get(awg_key)
    }

    /// Feedback source information for the given handle (QA signal information).
    pub(crate) fn feedback_source(&self, handle: &Handle) -> Option<&FeedbackSource> {
        self.feedback_sources.get(handle)
    }

    /// Iterator over all handles that are configured for feedback.
    pub(crate) fn handles(&self) -> impl Iterator<Item = &Handle> {
        self.feedback_sources.keys()
    }

    pub(crate) fn take_acquisitions(&mut self) -> Vec<Vec<Acquisition>> {
        std::mem::take(&mut self.acquisitions)
    }
}

/// Collect the feedback configuration from the given IR node for the given AWGs.
///
/// The function assigns feedback registers to the AWGs, where the assignment is based on the
/// order of the AWGs. It will also collect the signal information associated with each measurement handle.
pub(crate) fn collect_feedback_config(node: &IrNode, awgs: &[AwgCore]) -> Result<FeedbackConfig> {
    let mut handle_to_signal = HashMap::new();
    collect_handles(node, node.offset(), &mut handle_to_signal)?;
    let target_feedback_registers = allocate_feedback_registers(awgs, &handle_to_signal)?;
    let acquisitions = evaluate_simultaneous_acquires(&handle_to_signal)?;
    let mut signal_lookup: HashMap<SignalUid, (&Signal, AwgKey)> = HashMap::new();
    for awg in awgs.iter() {
        for signal in &awg.signals {
            signal_lookup.insert(signal.uid, (signal, awg.key()));
        }
    }
    let mut feedback_sources = HashMap::new();
    for (handle, handle_info) in handle_to_signal {
        let signal_info = signal_lookup
            .get(&handle_info.signal)
            .expect("Internal error: Expected signal for handle");
        feedback_sources.insert(
            handle,
            FeedbackSource {
                signal: signal_info.0.uid,
                awg_key: signal_info.1.clone(),
            },
        );
    }
    let config = FeedbackConfig::new(target_feedback_registers, feedback_sources, acquisitions);
    Ok(config)
}

/// Calculate the feedback register layout based on the integration unit allocations and
/// the feedback configuration.
pub(crate) fn calculate_feedback_register_layout(
    awgs: &[AwgCore],
    integration_unit_alloc: &[IntegrationUnitAllocation],
    feedback_config: &FeedbackConfig,
) -> FeedbackRegisterLayout {
    let mut feedback_register_alloc = Vec::new();
    for awg in awgs.iter() {
        if let Some(reg) = feedback_config.target_feedback_register(&awg.key()) {
            for signal in &awg.signals {
                feedback_register_alloc.push(FeedbackRegisterAlloc {
                    signal: signal.uid,
                    awg: awg.key(),
                    device: awg.device.uid().clone(),
                    is_local: matches!(reg, FeedbackRegisterAllocation::Local),
                });
            }
        }
    }
    calculate_feedback_register_layout_impl(integration_unit_alloc, &feedback_register_alloc)
}

struct FeedbackRegisterAlloc {
    signal: SignalUid,
    awg: AwgKey,
    device: DeviceUid,
    is_local: bool,
}

fn calculate_feedback_register_layout_impl(
    integration_unit_alloc: &[IntegrationUnitAllocation],
    feedback_register_alloc: &[FeedbackRegisterAlloc],
) -> FeedbackRegisterLayout {
    let mut integration_unit_alloc = integration_unit_alloc.iter().collect::<Vec<_>>();
    // Sort by channels to have a deterministic allocation order
    integration_unit_alloc.sort_by_key(|k| &k.channels);

    let mut feedback_register_layout = FeedbackRegisterLayout::default();
    for alloc in integration_unit_alloc {
        let Some(feedback_alloc) = feedback_register_alloc
            .iter()
            .find(|a| a.signal == alloc.signal)
        else {
            continue;
        };
        let register = if feedback_alloc.is_local {
            FeedbackRegister::Local {
                device: feedback_alloc.device.clone(),
            }
        } else {
            FeedbackRegister::Global {
                awg_key: feedback_alloc.awg.clone(),
            }
        };

        let bit_width = if feedback_alloc.is_local || alloc.kernel_count > 1 {
            2
        } else {
            1
        };

        let item = SingleFeedbackRegisterLayoutItem {
            width: bit_width,
            signal: Some(alloc.signal),
        };
        feedback_register_layout
            .entry(register.clone())
            .or_default()
            .push(item);

        if (bit_width as usize) < alloc.channels.len() {
            // On UHFQA, with `AcquisitionType.INTEGRATION`, we have
            // 2 integrators per signal. For discrimination, that 2nd integrator is irrelevant, so
            // we mark that bit as a 'dummy' field.
            feedback_register_layout.get_mut(&register).unwrap().push(
                SingleFeedbackRegisterLayoutItem {
                    width: 1,
                    signal: None,
                },
            );
        }
    }
    feedback_register_layout
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_integration_unit_alloc(
        signal: SignalUid,
        channels: Vec<u8>,
        kernel_count: u8,
    ) -> IntegrationUnitAllocation {
        IntegrationUnitAllocation {
            signal,
            channels,
            kernel_count,
        }
    }

    fn create_feedback_register_alloc(
        signal: SignalUid,
        device: &str,
        awg_index: u16,
        is_local: bool,
    ) -> FeedbackRegisterAlloc {
        FeedbackRegisterAlloc {
            signal,
            awg: AwgKey::new(device.into(), awg_index),
            device: device.into(),
            is_local,
        }
    }

    #[test]
    fn test_empty_inputs() {
        let layout = calculate_feedback_register_layout_impl(&[], &[]);
        assert!(layout.is_empty());
    }

    #[test]
    fn test_single_local_feedback_register() {
        let integration_allocs = vec![create_integration_unit_alloc(0.into(), vec![0], 1)];
        let feedback_allocs = vec![create_feedback_register_alloc(0.into(), "dev1", 0, true)];

        let layout = calculate_feedback_register_layout_impl(&integration_allocs, &feedback_allocs);

        assert_eq!(layout.len(), 1);
        let local_register = FeedbackRegister::Local {
            device: "dev1".into(),
        };
        let items = layout.get(&local_register).unwrap();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].width, 2); // Local feedback always gets width 2
        assert_eq!(items[0].signal, Some(0.into()));
    }

    #[test]
    fn test_single_global_feedback_register() {
        let integration_allocs = vec![create_integration_unit_alloc(0.into(), vec![0], 1)];
        let feedback_allocs = vec![create_feedback_register_alloc(0.into(), "dev1", 0, false)];

        let layout = calculate_feedback_register_layout_impl(&integration_allocs, &feedback_allocs);

        assert_eq!(layout.len(), 1);
        let global_register = FeedbackRegister::Global {
            awg_key: AwgKey::new("dev1".into(), 0),
        };
        let items = layout.get(&global_register).unwrap();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].width, 1); // Global with single kernel gets width 1
        assert_eq!(items[0].signal, Some(0.into()));
    }

    #[test]
    fn test_multiple_kernel_case() {
        let integration_allocs = vec![create_integration_unit_alloc(0.into(), vec![0, 1], 2)];
        let feedback_allocs = vec![create_feedback_register_alloc(0.into(), "dev1", 0, false)];

        let layout = calculate_feedback_register_layout_impl(&integration_allocs, &feedback_allocs);

        let global_register = FeedbackRegister::Global {
            awg_key: AwgKey::new("dev1".into(), 0),
        };
        let items = layout.get(&global_register).unwrap();
        assert_eq!(items[0].width, 2); // Multiple kernels get width 2 even if global
    }

    #[test]
    fn test_uhfqa_case_with_dummy_field() {
        // UHFQA case: bit_width (1) < channels.len() (2) should add dummy field
        let integration_allocs = vec![
            create_integration_unit_alloc(0.into(), vec![0, 1], 1), // 2 channels, 1 kernel
        ];
        let feedback_allocs = vec![
            create_feedback_register_alloc(0.into(), "dev1", 0, false), // global, single kernel -> width 1
        ];

        let layout = calculate_feedback_register_layout_impl(&integration_allocs, &feedback_allocs);

        let global_register = FeedbackRegister::Global {
            awg_key: AwgKey::new("dev1".into(), 0),
        };
        let items = layout.get(&global_register).unwrap();
        assert_eq!(items.len(), 2); // Original item + dummy field
        assert_eq!(items[0].width, 1);
        assert_eq!(items[0].signal, Some(0.into()));
        assert_eq!(items[1].width, 1); // Dummy field
        assert_eq!(items[1].signal, None); // Dummy has no signal
    }

    #[test]
    fn test_signal_without_feedback_allocation() {
        let integration_allocs = vec![
            create_integration_unit_alloc(0.into(), vec![0], 1),
            create_integration_unit_alloc(1.into(), vec![1], 1), // No feedback allocation for sig2
        ];
        let feedback_allocs = vec![create_feedback_register_alloc(0.into(), "dev1", 0, true)];

        let layout = calculate_feedback_register_layout_impl(&integration_allocs, &feedback_allocs);

        // Only sig1 should be in the layout, sig2 should be skipped
        assert_eq!(layout.len(), 1);
        let local_register = FeedbackRegister::Local {
            device: "dev1".into(),
        };
        let items = layout.get(&local_register).unwrap();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].signal, Some(0.into()));
    }

    #[test]
    fn test_multiple_signals_mixed_registers() {
        let integration_allocs = vec![
            create_integration_unit_alloc(0.into(), vec![0], 1),
            create_integration_unit_alloc(1.into(), vec![1], 1),
        ];
        let feedback_allocs = vec![
            create_feedback_register_alloc(0.into(), "dev1", 0, true), // local
            create_feedback_register_alloc(1.into(), "dev2", 0, false), // global
        ];

        let layout = calculate_feedback_register_layout_impl(&integration_allocs, &feedback_allocs);

        assert_eq!(layout.len(), 2); // Two different registers

        // Check local register
        let local_register = FeedbackRegister::Local {
            device: "dev1".into(),
        };
        let local_items = layout.get(&local_register).unwrap();
        assert_eq!(local_items[0].signal, Some(0.into()));
        assert_eq!(local_items[0].width, 2);

        // Check global register
        let global_register = FeedbackRegister::Global {
            awg_key: AwgKey::new("dev2".into(), 0),
        };
        let global_items = layout.get(&global_register).unwrap();
        assert_eq!(global_items[0].signal, Some(1.into()));
        assert_eq!(global_items[0].width, 1);
    }
}

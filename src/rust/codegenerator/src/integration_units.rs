// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::num::NonZero;

use laboneq_dsl::types::SignalUid;

use crate::ir::compilation_job::{AwgCore, AwgKey, ChannelIndex, DeviceKind, SignalKind};
use crate::ir::experiment::AcquisitionType;
use crate::ir::{IrNode, NodeKind};
use crate::result::IntegrationUnitAllocation;
use crate::{Error, Result};

/// Allocate integration units based on the acquisition nodes in the IR tree, the AWG cores,
/// and the acquisition type.
///
/// This function traverses the IR tree to identify acquisition nodes and determines the required
/// integration units for each signal based on the device capabilities and acquisition type.
pub(crate) fn allocate_integration_units(
    node: &IrNode,
    awgs: &[AwgCore],
    acquisition_type: &AcquisitionType,
) -> Result<Vec<IntegrationUnitAllocation>> {
    let mut kernels = KernelCollection::default();
    visit_nodes(node, &mut kernels)?;
    // TODO: Why allocate e.g. on UHFQA when no kernels are used?
    let kernel_counts = kernels.kernel_counts;
    let awg_by_signal = awgs
        .iter()
        .flat_map(|awg| awg.signals.iter().map(move |s| (&s.uid, awg)))
        .collect::<HashMap<_, _>>();

    // Collect integration signals
    let mut integration_signals: Vec<_> = awgs
        .iter()
        .flat_map(|awg| {
            awg.signals
                .iter()
                .filter(|s| matches!(s.kind, SignalKind::INTEGRATION))
        })
        .collect();

    // Sort for alignment in feedback register, place qudits before qubits
    integration_signals
        .sort_by_key(|s| kernel_counts.get(&s.uid).map(|a| a.get()).unwrap_or(0) <= 1);

    let mut integration_unit_alloc: HashMap<AwgKey, Vec<(SignalUid, Vec<ChannelIndex>)>> =
        HashMap::new();
    for signal in integration_signals.iter() {
        let awg_key = awg_by_signal[&signal.uid].key();
        let num_acquire_signals = integration_unit_alloc
            .get(&awg_key)
            .map_or(0, |v| v.len() as u8);
        let integrators_per_signal =
            integrators_per_signal(awg_by_signal[&signal.uid].device_kind(), acquisition_type)?;
        let unit_alloc = (0..integrators_per_signal)
            .map(|i| integrators_per_signal * num_acquire_signals + i)
            .collect::<Vec<_>>();
        integration_unit_alloc
            .entry(awg_key)
            .or_default()
            .push((signal.uid, unit_alloc));
    }

    let integration_unit_alloc: Vec<IntegrationUnitAllocation> = integration_unit_alloc
        .into_values()
        .flat_map(|signals| {
            signals.into_iter().map(|(signal, channels)| {
                let kernel_count = kernel_counts.get(&signal).map(|c| c.get());
                IntegrationUnitAllocation {
                    signal,
                    channels,
                    kernel_count: kernel_count.unwrap_or(0),
                }
            })
        })
        .collect();
    Ok(integration_unit_alloc)
}

#[derive(Default)]
struct KernelCollection {
    kernel_counts: HashMap<SignalUid, NonZero<ChannelIndex>>,
}

impl KernelCollection {
    fn register_acquisition(&mut self, signal: SignalUid, kernel_count: usize) -> Result<()> {
        let count = kernel_count.try_into().map_err(|_| {
            Error::new(format!(
                "Kernel count for signal {} exceeds u8 limit: {}",
                signal.0, kernel_count
            ))
        })?;
        let kernel_count = NonZero::new(count).ok_or_else(|| {
            Error::new(format!(
                "Kernel count for signal {} cannot be zero",
                signal.0
            ))
        })?;
        if let Some(existing) = self.kernel_counts.get(&signal)
            && *existing != kernel_count
        {
            return Err(Error::new(format!(
                "Inconsistent kernel counts for signal '{}'",
                signal.0
            )));
        } else {
            self.kernel_counts.insert(signal, kernel_count);
        }
        Ok(())
    }
}

fn visit_nodes(node: &IrNode, kernels: &mut KernelCollection) -> Result<()> {
    if let NodeKind::AcquirePulse(acq) = node.data() {
        kernels.register_acquisition(acq.signal.uid, acq.pulse_defs.len())?;
    }
    for child in node.iter_children() {
        visit_nodes(child, kernels)?;
    }
    Ok(())
}

fn integrators_per_signal(device: &DeviceKind, acquisition_type: &AcquisitionType) -> Result<u8> {
    if device == &DeviceKind::UHFQA && acquisition_type == &AcquisitionType::SPECTROSCOPY_PSD {
        return Err(Error::new(
            "Acquisition type 'Spectroscopy PSD' is not allowed on UHFQA",
        ));
    }
    if matches!(
        acquisition_type,
        AcquisitionType::RAW
            | AcquisitionType::INTEGRATION
            | AcquisitionType::SPECTROSCOPY_PSD
            | AcquisitionType::SPECTROSCOPY_IQ
    ) {
        Ok(device.traits().num_integration_units_per_acquire_signal)
    } else {
        Ok(1)
    }
}

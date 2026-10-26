// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};

use laboneq_common::types::AwgKey;

use crate::device_kind::DeviceKind;
use laboneq_dsl::ExperimentNode;
use laboneq_dsl::operation::Operation;
use laboneq_dsl::types::{ParameterUid, SignalUid, Trigger, ValueOrParameter};

use laboneq_error::bail;

use crate::Result;
use crate::preprocessor::{QccsBackendPreprocessedData, SignalProperties};
use crate::validation::NodeContext;

/// Checks that an SHFPPC sweep is not combined with a feature it excludes.
///
/// The sweep occupies the channel the feature would need, so the conflict is
/// per AWG rather than per signal.
pub(super) struct PpcSweepRules<'a> {
    data: &'a QccsBackendPreprocessedData,
    rt_sweep_parameters: HashSet<ParameterUid>,
    awgs_with_section_trigger: HashMap<AwgKey, SignalUid>,
}

impl<'a> PpcSweepRules<'a> {
    pub(super) fn new(data: &'a QccsBackendPreprocessedData) -> Self {
        PpcSweepRules {
            data,
            rt_sweep_parameters: HashSet::new(),
            awgs_with_section_trigger: HashMap::new(),
        }
    }

    pub(super) fn visit(&mut self, node: &ExperimentNode, ctx: NodeContext) {
        if ctx.inside_rt_bound
            && let Some(loop_info) = &node.kind.loop_info()
        {
            self.rt_sweep_parameters
                .extend(loop_info.parameters.iter().copied());
        }
        if let Operation::Section(op) = &node.kind {
            self.digest_section_triggers(&op.triggers);
        }
    }

    /// Report the signals that drive a swept SHFPPC and use a feature the
    /// sweep excludes.
    pub(super) fn report_conflicts(&self) -> Result<()> {
        let ppc_sweeps = self.awgs_with_ppc_sweeps();
        if let Some(signals) = conflicting_signals(&self.awgs_with_automute(), &ppc_sweeps) {
            bail!(
                "Signals on the following channels drive both SHFPPC sweeps, and use the \
                output auto-muting feature: {signals}"
            );
        }
        if let Some(signals) = conflicting_signals(&self.awgs_with_section_trigger, &ppc_sweeps) {
            bail!(
                "Signals on the following channels drive both SHFPPC sweeps, and use \
                section triggers: {signals}"
            );
        }
        Ok(())
    }

    /// Collect the AWGs whose signals are triggered by a section.
    fn digest_section_triggers(&mut self, triggers: &[Trigger]) {
        for trigger in triggers {
            let Some(signal) = self.data.signal(trigger.signal) else {
                continue;
            };
            if matches!(signal.device_kind, DeviceKind::Shfqa) {
                self.awgs_with_section_trigger
                    .insert(signal.awg_key, signal.uid);
            }
        }
    }

    /// The AWGs whose signals auto-mute their output.
    fn awgs_with_automute(&self) -> HashMap<AwgKey, SignalUid> {
        self.data
            .signal_properties()
            .filter(|signal| {
                signal.calibration.automute && matches!(signal.device_kind, DeviceKind::Shfqa)
            })
            .map(|signal| (signal.awg_key, signal.uid))
            .collect()
    }

    /// Collect the AWGs whose signals drive an SHFPPC swept in real time.
    fn awgs_with_ppc_sweeps(&self) -> HashMap<AwgKey, SignalUid> {
        let swept = |signal: &SignalProperties| {
            signal
                .calibration
                .amplifier_pump
                .as_ref()
                .into_iter()
                .flat_map(|pump| pump.values_or_parameters())
                .flatten()
                .any(|param| match param {
                    ValueOrParameter::Parameter(uid)
                    | ValueOrParameter::ResolvedParameter { uid, .. } => {
                        self.rt_sweep_parameters.contains(uid)
                    }
                    _ => false,
                })
        };
        self.data
            .signal_properties()
            .filter(swept)
            .map(|signal| (signal.awg_key, signal.uid))
            .collect()
    }
}

/// The PPC-sweeping signals on the AWGs that also use the conflicting feature.
fn conflicting_signals(
    feature: &HashMap<AwgKey, SignalUid>,
    ppc_sweeps: &HashMap<AwgKey, SignalUid>,
) -> Option<String> {
    let mut conflicts = feature
        .keys()
        .filter_map(|awg| ppc_sweeps.get(awg))
        .map(|uid| uid.0.to_string())
        .collect::<Vec<_>>();
    conflicts.sort();
    (!conflicts.is_empty()).then(|| conflicts.join(", "))
}

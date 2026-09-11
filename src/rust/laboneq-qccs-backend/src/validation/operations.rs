// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};

use laboneq_common::types::SignalKind;
use laboneq_dsl::ExperimentNode;
use laboneq_dsl::operation::{Operation, PlayPulse};
use laboneq_dsl::types::{MarkerSelector, PulseUid, SignalUid, Trigger};
use laboneq_error::{bail, laboneq_error};

use crate::Result;
use crate::device_kind::DeviceKind;
use crate::preprocessor::{QccsBackendPreprocessedData, SignalProperties};
use crate::validation::NodeContext;
/// Checks the operations of a QCCS experiment against what the device playing
/// them supports.
pub(super) struct OperationRules<'a> {
    data: &'a QccsBackendPreprocessedData,
    signal_pulse_map: HashMap<SignalUid, HashSet<PulseUid>>,
}

impl<'a> OperationRules<'a> {
    pub(super) fn new(data: &'a QccsBackendPreprocessedData) -> Self {
        OperationRules {
            data,
            signal_pulse_map: HashMap::new(),
        }
    }

    pub(super) fn visit(&mut self, node: &ExperimentNode, ctx: NodeContext) -> Result<()> {
        match &node.kind {
            Operation::PlayPulse(op) => {
                let signal = self
                    .data
                    .signal(op.signal)
                    .ok_or_else(|| laboneq_error!("Signal '{}' not found.", op.signal.0))?;
                check_markers(op, &signal, ctx)?;
                self.shfqa_unique_measure_pulse(op, &signal)?;
            }
            Operation::Section(op) => {
                self.check_triggers(&op.triggers)?;
            }
            _ => {}
        }
        Ok(())
    }

    /// An SHFQA generator holds a single waveform, so a measure line can only
    /// ever play one pulse.
    fn shfqa_unique_measure_pulse(
        &mut self,
        pulse: &PlayPulse,
        signal: &SignalProperties,
    ) -> Result<()> {
        let Some(pulse_uid) = pulse.pulse.as_ref() else {
            return Ok(());
        };
        if matches!(signal.kind, SignalKind::Integration)
            || !matches!(signal.device_kind, DeviceKind::Shfqa)
        {
            return Ok(());
        }

        let played = self.signal_pulse_map.entry(signal.uid).or_default();
        played.insert(*pulse_uid);
        if played.len() > 1 {
            bail!(
                "Multiple different pulses are being played on signal '{}'. SHFQA \
                generators can only hold a single pulse waveform. Therefore, playing \
                multiple readout pulses represented by different Python objects is \
                not possible on a SHFQA measurement line.",
                signal.uid.0,
            );
        }
        Ok(())
    }

    /// Muting the output would swallow the trigger the section drives.
    fn check_triggers(&self, triggers: &[Trigger]) -> Result<()> {
        for trigger in triggers.iter() {
            if let Some(signal) = self.data.signal(trigger.signal)
                && signal.calibration.automute
            {
                bail!(
                    "Automute cannot be used with triggers on same signal: '{}'.",
                    signal.uid.0
                );
            }
        }
        Ok(())
    }
}

fn check_markers(pulse: &PlayPulse, signal: &SignalProperties, ctx: NodeContext) -> Result<()> {
    let Some(pulse_uid) = pulse.pulse.as_ref() else {
        return Ok(());
    };

    let section_uid = ctx.section_uid()?;
    if matches!(signal.device_kind, DeviceKind::Hdawg)
        && matches!(signal.kind, SignalKind::Rf)
        && pulse
            .markers
            .iter()
            .any(|m| matches!(m.marker_selector, MarkerSelector::M2))
    {
        bail!(
            "Single channel RF Pulse '{}' referenced in section '{}' \
            has marker 2 enabled. Please only use marker 1 on RF channels.",
            pulse_uid.0,
            section_uid.0,
        );
    }
    if signal.device_kind.is_qa_device() && !pulse.markers.is_empty() {
        bail!(
            "Pulse '{}' referenced in section {} \
            has markers but is to be played on a QA device. QA \
            devices do not support markers.",
            pulse_uid.0,
            section_uid.0,
        );
    }

    // Check for marker / automute conflicts
    if !pulse.markers.is_empty() && signal.calibration.automute {
        bail!(
            "Automute cannot be used with markers on same signal: '{}'.",
            signal.uid.0
        );
    }
    Ok(())
}

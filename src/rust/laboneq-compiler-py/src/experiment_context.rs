// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::num::NonZeroU32;

use laboneq_dsl::ExperimentNode;
use laboneq_dsl::operation::{AveragingLoop, Operation};
use laboneq_dsl::types::{
    AcquisitionType, AveragingMode, HandleUid, PulseDef, PulseLength, PulseUid, SectionUid,
    SignalUid,
};
use laboneq_units::duration::{Duration, Second, seconds};

use crate::error::{Error, Result};
use crate::experiment::Experiment;

/// Immutable lookup/index data derived from the experiment
///
/// Any information that is added to the context shall not be
/// mutated during compilation process.
pub(crate) struct ExperimentContext {
    /// Map from acquisition handle UIDs to signal UIDs
    handle_to_signal: HashMap<HandleUid, SignalUid>,
    rt_properties: RealTimeLoopProperties,
    maximum_acquisition_lengths: HashMap<SignalUid, (Duration<Second>, usize)>,
}

impl ExperimentContext {
    pub(crate) fn handle_to_signal(&self) -> &HashMap<HandleUid, SignalUid> {
        &self.handle_to_signal
    }

    pub(crate) fn maximum_acquisition_lengths(
        &self,
        signal: &SignalUid,
    ) -> Option<&(Duration<Second>, usize)> {
        self.maximum_acquisition_lengths.get(signal)
    }

    pub(crate) fn acquisition_type(&self) -> &AcquisitionType {
        &self.rt_properties.acquisition_type
    }

    pub(crate) fn rt_properties(&self) -> &RealTimeLoopProperties {
        &self.rt_properties
    }
}

pub(crate) struct RealTimeLoopProperties {
    pub uid: SectionUid,
    pub acquisition_type: AcquisitionType,
    pub averaging_mode: AveragingMode,
    pub count: NonZeroU32,
}

/// Create an [`ExperimentContext`] from an [`Experiment`].
pub(crate) fn experiment_context_from_experiment(
    experiment: &Experiment,
) -> Result<ExperimentContext> {
    let mut context = ExperimentContextCollector::new(&experiment.pulses);
    context.visit_node(&experiment.root)?;
    context.into_context()
}

struct ExperimentContextCollector<'a> {
    pulses: &'a HashMap<PulseUid, PulseDef>,

    /// Map from acquisition handle UIDs to signal UIDs
    handle_to_signal: HashMap<HandleUid, SignalUid>,
    rt_properties: Option<RealTimeLoopProperties>,
    maximum_acquisition_lengths: HashMap<SignalUid, (Duration<Second>, usize)>,
}

impl<'a> ExperimentContextCollector<'a> {
    fn new(pulses: &'a HashMap<PulseUid, PulseDef>) -> Self {
        Self {
            pulses,
            handle_to_signal: HashMap::new(),
            rt_properties: None,
            maximum_acquisition_lengths: HashMap::new(),
        }
    }

    fn into_context(self) -> Result<ExperimentContext> {
        Ok(ExperimentContext {
            handle_to_signal: self.handle_to_signal,
            rt_properties: self.rt_properties.ok_or_else(|| {
                Error::new("Experiment must have exactly one real time acquisition loop")
            })?,
            maximum_acquisition_lengths: self.maximum_acquisition_lengths,
        })
    }

    fn visit_node(&mut self, node: &ExperimentNode) -> Result<()> {
        match &node.kind {
            Operation::Acquire(obj) => {
                self.visit_acquire(obj)?;
            }
            Operation::AveragingLoop(obj) => {
                self.set_rt_properties(obj)?;
                for child in &node.children {
                    self.visit_node(child)?;
                }
            }
            _ => {
                for child in &node.children {
                    self.visit_node(child)?;
                }
            }
        }
        Ok(())
    }

    fn visit_acquire(&mut self, acquire: &laboneq_dsl::operation::Acquire) -> Result<()> {
        self.add_handle_signal_mapping(acquire.handle, acquire.signal)?;

        // Collect acquisition length information. If the acquisition directly specifies a length, use that.
        if let Some(length) = acquire.length {
            self.register_acquisition_length_seconds(acquire.signal, length);
        } else {
            for kernel in &acquire.kernel {
                let pulse = self.pulses.get(kernel).ok_or_else(|| {
                    Error::new(format!(
                        "Kernel pulse '{}' not found for acquisition.",
                        kernel.0
                    ))
                })?;
                match pulse.length() {
                    PulseLength::Seconds(length) => {
                        self.register_acquisition_length_seconds(acquire.signal, length);
                    }
                    PulseLength::Samples(length) => {
                        self.register_acquisition_length_samples(acquire.signal, length.value());
                    }
                }
            }
        }
        Ok(())
    }

    fn set_rt_properties(&mut self, op: &AveragingLoop) -> Result<()> {
        if self.rt_properties.is_some() {
            return Err(Error::new(
                "Experiment must not contain multiple real-time averaging loops",
            ));
        }
        let rt_properties = RealTimeLoopProperties {
            uid: op.uid,
            acquisition_type: op.acquisition_type,
            averaging_mode: op.averaging_mode,
            count: op.count,
        };
        self.rt_properties = Some(rt_properties);
        Ok(())
    }

    fn add_handle_signal_mapping(&mut self, handle: HandleUid, signal: SignalUid) -> Result<()> {
        if let Some(existing_signal) = self.handle_to_signal.get(&handle)
            && existing_signal != &signal
        {
            return Err(Error::new(format!(
                "Acquisition handle '{}' is associated with multiple signals, only one allowed.",
                handle.0
            )));
        }
        self.handle_to_signal.insert(handle, signal);
        Ok(())
    }

    fn register_acquisition_length_seconds(&mut self, signal: SignalUid, length: Duration<Second>) {
        self.maximum_acquisition_lengths
            .entry(signal)
            .and_modify(|existing_length| {
                if existing_length.0 < length {
                    existing_length.0 = length;
                }
            })
            .or_insert((length, 0));
    }

    fn register_acquisition_length_samples(&mut self, signal: SignalUid, length: usize) {
        self.maximum_acquisition_lengths
            .entry(signal)
            .and_modify(|existing_length| {
                if existing_length.1 < length {
                    existing_length.1 = length;
                }
            })
            .or_insert((seconds(0.0), length));
    }
}

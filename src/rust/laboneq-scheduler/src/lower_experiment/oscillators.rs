// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use laboneq_units::duration::seconds;
use laboneq_units::tinysample::{TinySamples, seconds_to_tinysamples, tiny_samples};
use std::collections::HashSet;

use crate::error::{Error, Result};
use crate::experiment::types::{OscillatorKind, ParameterUid, ValueOrParameter};
use crate::ir::{
    InitialLocalOscillatorFrequency, InitialOscillatorFrequency, IrKind, SetOscillatorFrequency,
};
use crate::schedule_info::ScheduleInfoBuilder;
use crate::utils::ceil_to_grid;
use crate::{ParameterStore, ScheduledNode, SignalInfo};

pub(super) fn handle_initial_oscillator_frequency<T: SignalInfo + Sized>(
    signals: &[&T],
    parameters: &ParameterStore,
    system_grid: TinySamples,
) -> Result<ScheduledNode> {
    let mut values = Vec::new();
    for signal in signals {
        if let Some(osc) = signal.oscillator() {
            let freq_unwrapped = match &osc.frequency {
                ValueOrParameter::Parameter(obj) => {
                    // HW oscillators are either set by the controller, or swept in a RT
                    // loop. Either way, no need to set the initial frequency here. If we
                    // did include it in the schedule this would count as a dependency on
                    // the parameter, and force needless recompilation of each NT step.
                    if osc.kind == OscillatorKind::Hardware {
                        continue;
                    }
                    if let Some(freq) = parameters.get(obj) {
                        TryInto::<ValueOrParameter<f64>>::try_into(*freq).unwrap()
                    } else {
                        continue;
                    }
                }
                value => *value,
            };
            values.push((signal.uid(), freq_unwrapped));
        }
    }
    let node = ScheduledNode::new(
        IrKind::InitialOscillatorFrequency(InitialOscillatorFrequency { values }),
        ScheduleInfoBuilder::new()
            .length(0)
            .grid(system_grid)
            .build(),
    );
    Ok(node)
}

pub(super) fn handle_initial_local_oscillator_frequency<T: SignalInfo + Sized>(
    signals: &[&T],
    parameters: &ParameterStore,
    system_grid: TinySamples,
) -> Result<Vec<ScheduledNode>> {
    signals
        .iter()
        .filter_map(|signal| {
            if !signal.supports_initial_local_oscillator_frequency() {
                return None;
            }
            signal.lo_frequency().and_then(|&lo_freq| {
                let value = match lo_freq {
                    ValueOrParameter::Parameter(obj) => {
                        if let Some(freq) = parameters.get(&obj) {
                            TryInto::<ValueOrParameter<f64>>::try_into(*freq).unwrap()
                        } else {
                            return Some(Err(Error::new(
                                "Local oscillator sweep must be in near-time.",
                            )));
                        }
                    }
                    value => value,
                };
                let node = ScheduledNode::new(
                    IrKind::InitialLocalOscillatorFrequency(InitialLocalOscillatorFrequency {
                        signal: signal.uid(),
                        value,
                    }),
                    ScheduleInfoBuilder::new()
                        .length(0)
                        .grid(system_grid)
                        .build(),
                );
                Ok(node).into()
            })
        })
        .collect::<Result<Vec<_>>>()
}

/// Creates IR node to set oscillator frequencies for `signals` that have their oscillator frequency
/// controlled by one of the given `parameters`.
pub(super) fn handle_set_oscillator_frequency<T: SignalInfo + Sized>(
    signals: &[&T],
    parameters: HashSet<&ParameterUid>,
    system_grid: TinySamples,
) -> Result<Option<ScheduledNode>> {
    if parameters.is_empty() {
        return Ok(None);
    }
    let mut swept_hw_osc_signals: Vec<&T> = Vec::new();
    let mut signal_osc_value = vec![];
    let mut oscillator_type: Option<OscillatorKind> = None;
    for signal in signals {
        if let Some(osc) = signal.oscillator()
            && let ValueOrParameter::Parameter(param_uid) = osc.frequency
            && parameters.contains(&param_uid)
        {
            match oscillator_type.as_mut() {
                None => oscillator_type = Some(osc.kind),
                Some(existing_type) => {
                    if *existing_type != osc.kind {
                        return Err(Error::new(
                            "Cannot sweep mixed hardware and software oscillators.",
                        ));
                    }
                }
            }
            swept_hw_osc_signals.push(signal);
            signal_osc_value.push((signal.uid(), osc.frequency));
        }
    }
    if signal_osc_value.is_empty() {
        return Ok(None);
    }
    let length = if oscillator_type == Some(OscillatorKind::Hardware) {
        swept_hw_osc_signals
            .iter()
            .fold(seconds(0.0), |acc, signal| {
                acc.max(signal.device_traits().oscillator_set_latency)
            })
    } else {
        seconds(0.0)
    };
    let node = ScheduledNode::new(
        IrKind::SetOscillatorFrequency(SetOscillatorFrequency {
            values: signal_osc_value,
        }),
        ScheduleInfoBuilder::new()
            .grid(system_grid)
            .length(tiny_samples(ceil_to_grid(
                seconds_to_tinysamples(length).value(),
                system_grid.value(),
            )))
            .build(),
    );
    Ok(Some(node))
}

// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use crate::error::{Error, Result};
use crate::experiment::types::{OscillatorKind, RealValue, SignalUid};
use crate::ir::{InitialLocalOscillatorFrequency, InitialOscillatorFrequency, IrKind};
use crate::schedule_info::ScheduleInfoBuilder;
use crate::{ParameterStore, ScheduledNode, SignalInfo, TinySample};

pub fn handle_initial_oscillator_frequency<T: SignalInfo + Sized>(
    signals: &HashMap<SignalUid, T>,
    parameters: &ParameterStore,
    system_grid: TinySample,
) -> Result<ScheduledNode> {
    let mut values = Vec::new();
    for signal in signals.values() {
        if let Some(osc) = signal.oscillator() {
            let freq_unwrapped = match osc.frequency {
                RealValue::ParameterUid(obj) => {
                    // HW oscillators are either set by the controller, or swept in a RT
                    // loop. Either way, no need to set the initial frequency here. If we
                    // did include it in the schedule this would count as a dependency on
                    // the parameter, and force needless recompilation of each NT step.
                    if osc.kind == OscillatorKind::Hardware {
                        continue;
                    }
                    if let Some(freq) = parameters.get(&obj) {
                        TryInto::<RealValue>::try_into(*freq).unwrap()
                    } else {
                        continue;
                    }
                }
                value => value,
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

pub fn handle_initial_local_oscillator_frequency<T: SignalInfo + Sized>(
    signals: &HashMap<SignalUid, T>,
    parameters: &ParameterStore,
    system_grid: TinySample,
) -> Result<Vec<ScheduledNode>> {
    signals
        .values()
        .filter_map(|signal| {
            if !signal.supports_initial_local_oscillator_frequency() {
                return None;
            }
            signal.lo_frequency().and_then(|&lo_freq| {
                let value = match lo_freq {
                    RealValue::ParameterUid(obj) => {
                        if let Some(freq) = parameters.get(&obj) {
                            TryInto::<RealValue>::try_into(*freq).unwrap()
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

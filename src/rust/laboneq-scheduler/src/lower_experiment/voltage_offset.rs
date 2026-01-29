// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use crate::error::{Error, Result};

use crate::ir::{InitialVoltageOffset, IrKind};
use crate::schedule_info::ScheduleInfoBuilder;
use crate::{ParameterStore, ScheduledNode, SignalInfo};
use laboneq_dsl::types::ValueOrParameter;
use laboneq_units::tinysample::TinySamples;

pub(super) fn handle_initial_voltage_offset<T: SignalInfo + Sized>(
    signals: &[&T],
    parameters: &ParameterStore,
    system_grid: TinySamples,
) -> Result<Vec<ScheduledNode>> {
    signals
        .iter()
        .filter_map(|signal| {
            if !signal.supports_initial_voltage_offset() {
                return None;
            }
            signal.voltage_offset().and_then(|voltage_offset| {
                let value = match voltage_offset {
                    ValueOrParameter::Parameter(obj) => {
                        if let Some(freq) = parameters.get(obj) {
                            TryInto::<ValueOrParameter<f64>>::try_into(*freq).unwrap()
                        } else {
                            return Some(Err(Error::new(
                                "Voltage offset sweep must be in near-time.",
                            )));
                        }
                    }
                    value => *value,
                };
                let node = ScheduledNode::new(
                    IrKind::InitialVoltageOffset(InitialVoltageOffset {
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

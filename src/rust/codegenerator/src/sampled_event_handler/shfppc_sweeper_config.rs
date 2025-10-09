// // Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use crate::ir::{PpcDevice, experiment::SweepCommand};
use serde_json::Value;

fn swept_field_names(command: &SweepCommand) -> Vec<&str> {
    let mut fields = Vec::new();
    // Obey alphabetical order.
    if command.cancellation_attenuation.is_some() {
        fields.push("cancellation_attenuation");
    }
    if command.cancellation_phase.is_some() {
        fields.push("cancellation_phase");
    }
    if command.probe_frequency.is_some() {
        fields.push("probe_frequency");
    }
    if command.probe_power.is_some() {
        fields.push("probe_power");
    }
    if command.pump_frequency.is_some() {
        fields.push("pump_frequency");
    }
    if command.pump_power.is_some() {
        fields.push("pump_power");
    }
    fields
}

fn merge_if_none(this_command: &mut SweepCommand, other: &SweepCommand) {
    if this_command.pump_frequency.is_none() && other.pump_frequency.is_some() {
        this_command.pump_frequency = other.pump_frequency;
    }
    if this_command.pump_power.is_none() && other.pump_power.is_some() {
        this_command.pump_power = other.pump_power;
    }
    if this_command.probe_frequency.is_none() && other.probe_frequency.is_some() {
        this_command.probe_frequency = other.probe_frequency;
    }
    if this_command.probe_power.is_none() && other.probe_power.is_some() {
        this_command.probe_power = other.probe_power;
    }
    if this_command.cancellation_phase.is_none() && other.cancellation_phase.is_some() {
        this_command.cancellation_phase = other.cancellation_phase;
    }
    if this_command.cancellation_attenuation.is_none() && other.cancellation_attenuation.is_some() {
        this_command.cancellation_attenuation = other.cancellation_attenuation;
    }
}

fn are_all_fields_some(command: &SweepCommand) -> bool {
    command.pump_frequency.is_some()
        && command.pump_power.is_some()
        && command.probe_frequency.is_some()
        && command.probe_power.is_some()
        && command.cancellation_phase.is_some()
        && command.cancellation_attenuation.is_some()
}
pub struct SHFPPCSweeperConfig {
    pub count: u64,
    pub commands: Vec<SweepCommand>,
}

impl SHFPPCSweeperConfig {
    /// Build the table of values for the sweep commands.
    /// This function will return a JSON string that contains the table of values.
    /// It also fills in the default values for the swept fields, i.e., the first value
    /// which is set in the command list in case the corresponding field is not set yet.
    ///
    /// Do not modifying the commands after calling this function.
    ///
    pub fn finalize(&mut self, ppc_device: Arc<PpcDevice>) -> Value {
        let mut active_values = SweepCommand::default();

        // We start by finding the 'default' values, ie. those that will be set first.
        for command in &self.commands {
            merge_if_none(&mut active_values, command);
            if are_all_fields_some(&active_values) {
                break;
            }
        }
        let mut current_values = &active_values;

        // Next, we fill all the swept fields in all the commands.
        for command in &mut self.commands {
            merge_if_none(command, current_values);
            current_values = command;
        }

        // Finally, construct the flat list of all values.
        let flat_list: Vec<Vec<f64>> = self
            .commands
            .iter()
            .map(|command| {
                vec![
                    // Obey alphabetical order.
                    command.cancellation_attenuation,
                    command
                        .cancellation_phase
                        .map(|v| v * 180.0 / std::f64::consts::PI),
                    command.probe_frequency,
                    command.probe_power,
                    command.pump_frequency,
                    command.pump_power,
                ]
                .into_iter()
                .flatten()
                .collect()
            })
            .collect();

        serde_json::json!({
            "header": {"version": "1.0"},
            "dimensions": swept_field_names(&active_values),
            "flat_list": flat_list,
            "repetitions": self.count,
            "ppc_device": ppc_device.device,
            "ppc_channel": ppc_device.channel
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cancellation_phase_conversion() {
        let commands = vec![
            SweepCommand {
                cancellation_phase: Some(0.0),
                pump_frequency: Some(4e9),
                ..SweepCommand::default()
            },
            SweepCommand {
                pump_frequency: Some(5e9),
                ..SweepCommand::default()
            },
            SweepCommand {
                cancellation_phase: Some(std::f64::consts::PI / 2.0),
                ..SweepCommand::default()
            },
        ];

        let mut config = SHFPPCSweeperConfig { count: 1, commands };
        let ppc_device = Arc::new(PpcDevice {
            device: "SHFQA".to_string(),
            channel: 123,
        });
        let table = config.finalize(ppc_device);
        assert_eq!(
            table,
            serde_json::json!({
                "header": {"version": "1.0"},
                "dimensions": ["cancellation_phase", "pump_frequency"],
                "flat_list": [
                    [0.0, 4000000000.0],
                    [0.0, 5000000000.0],
                    [90.0, 5000000000.0],
                ],
                "repetitions": 1,
                "ppc_device": "SHFQA",
                "ppc_channel": 123
            })
        );
    }
}

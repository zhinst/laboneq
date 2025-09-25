// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use crate::{Result, awg_events::PlayWaveEvent};
use codegenerator::{
    ir::{
        ParameterOperation,
        compilation_job::{AwgKind, DeviceKind},
    },
    signature::WaveformSignature,
};
use seqc_tracker::wave_index_tracker::WaveIndex;
use serde_json::{Map, Value, json};

pub enum ParameterPhaseIncrement {
    Index(usize),
    ComplexUsage,
}

pub struct CommandTableResults {
    pub command_table: Value,
    pub parameter_phase_increment_map: HashMap<String, Vec<ParameterPhaseIncrement>>,
}
pub struct CommandTableTracker {
    command_table: Vec<(usize, Value)>,
    table_index_by_signature: HashMap<PlayWaveEvent, usize>,
    device_type: DeviceKind,
    parameter_phase_increment_map: HashMap<String, Vec<ParameterPhaseIncrement>>,
    phase_reset_entry: Option<usize>,
    signal_kind: AwgKind,
}

impl CommandTableTracker {
    pub fn new(device_type: DeviceKind, signal_kind: AwgKind) -> Self {
        Self {
            command_table: Vec::new(),
            table_index_by_signature: HashMap::new(),
            device_type,
            parameter_phase_increment_map: HashMap::new(),
            signal_kind,
            phase_reset_entry: None,
        }
    }

    pub fn get(&self, index: usize) -> Result<(&PlayWaveEvent, &Value), String> {
        if let Some(entry) = self.command_table.get(index) {
            for (signature, &idx) in &self.table_index_by_signature {
                if idx == entry.0 {
                    return Ok((signature, &entry.1));
                }
            }
            Err(format!("No signature associated with index {index}"))
        } else {
            Err(format!("Index {index} out of bounds"))
        }
    }

    /// Looks up the index of a command table entry in the command table by its playback signature.
    ///
    /// # Arguments
    /// * `signature` - The playback signature to look up.
    ///
    /// # Returns
    /// * `Option<usize>` - The index of the command table entry if found,
    ///   or `None` if the signature is not in the command table.
    ///
    pub fn lookup_index_by_signature(&self, signature: &PlayWaveEvent) -> Option<usize> {
        self.table_index_by_signature.get(signature).copied()
    }

    /// Returns the number of entries in the command table.
    ///
    /// # Returns
    /// * `usize` - The number of entries in the command table.
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.command_table.len()
    }

    /// Creates a new command table entry.
    ///
    /// # Arguments
    /// * `signature` - The playback signature for the entry.
    /// * `wave_index` - Optional index of the waveform.
    /// * `ignore_already_in_table` - If true, does not error if the signature is already in the table.
    ///
    /// # Returns
    /// * `Ok(usize)` - The index of the newly created entry.
    /// * `Err(String)` - An error message if the entry could not be created.
    ///
    /// # Errors
    /// Returns an error if the signature already exists in the table and `ignore_already_in_table` is false
    /// or if the command table exceeds the entry limit.
    ///
    pub fn create_entry(
        &mut self,
        signature: &PlayWaveEvent,
        wave_index: Option<WaveIndex>,
        ignore_already_in_table: bool, // Default to false
    ) -> Result<usize> {
        if !ignore_already_in_table && self.table_index_by_signature.contains_key(signature) {
            return Err(anyhow::anyhow!(
                "Signature {signature:?} already exists in command table"
            ));
        }
        let index = self.command_table.len();

        let complex_phase_increment = signature.increment_phase_params.len() > 1;
        for param in signature.increment_phase_params.iter().flatten() {
            self.parameter_phase_increment_map
                .entry(param.clone())
                .or_default()
                .push(if complex_phase_increment {
                    ParameterPhaseIncrement::ComplexUsage
                } else {
                    ParameterPhaseIncrement::Index(index)
                });
        }
        let mut json = if let Some(wave_index) = wave_index {
            json!({
                "index": index,
                "waveform": {
                    "index": wave_index
                }
            })
        } else if let WaveformSignature::Pulses { length, .. } = signature.waveform
            && length > 0
        {
            json!({
                "index": index,
                "waveform": {
                    "playZero": true,
                    "length": length
                }
            })
        } else {
            json!({
                "index": index
            })
        };
        let inner = json.as_object_mut().unwrap();
        self.add_oscillator_config(signature, inner);
        self.add_amplitude_config(signature, inner);
        self.command_table.push((index, json));
        if !self.table_index_by_signature.contains_key(signature) {
            self.table_index_by_signature
                .insert(signature.clone(), index);
        }
        Ok(index)
    }

    /// Creates a command table entry for a precompensation clear operation.
    ///
    /// # Arguments
    /// * `signature` - The playback signature for the entry.
    /// * `wave_index` - The index of the (playzero) waveform to use.
    ///
    /// # Returns
    /// * `usize` - The index of the newly created entry in the command table.
    ///
    pub fn create_precompensation_clear_entry(
        &mut self,
        signature: PlayWaveEvent,
        wave_index: WaveIndex,
    ) -> usize {
        assert!(signature.hw_oscillator.is_none());
        assert!(signature.increment_phase_params.is_empty());
        assert!(signature.amplitude.is_none());
        assert!(signature.state.is_none());
        if let Some(idx) = self.lookup_index_by_signature(&signature) {
            return idx;
        }
        let index = self.command_table.len();
        let json = json!({
            "index": index,
            "waveform": {
                "index": wave_index,
                "precompClear": true,
            }
        });
        self.command_table.push((index, json));
        self.table_index_by_signature.insert(signature, index);
        index
    }

    pub fn create_phase_reset_entry(&mut self) -> usize {
        if let Some(entry) = self.phase_reset_entry {
            return entry;
        }
        let index = self.command_table.len();
        let mut json = json!({"index": index});
        let inner = json.as_object_mut().expect("Expected a JSON object");
        if matches!(self.device_type, DeviceKind::HDAWG) {
            if matches!(self.signal_kind, AwgKind::SINGLE | AwgKind::DOUBLE) {
                inner.insert("phase0".into(), json!({"value": 90.0}));
                inner.insert("phase1".into(), json!({"value": 90.0}));
            } else {
                inner.insert("phase0".into(), json!({"value": 90.0}));
                inner.insert("phase1".into(), json!({"value": 0.0}));
            }
        } else if matches!(self.device_type, DeviceKind::SHFSG) {
            inner.insert("phase".into(), json!({"value": 0.0}));
        } else {
            panic!(
                "Internal error: Unsupported device type {:?} for phase configuration",
                self.device_type
            );
        }
        self.command_table.push((index, json));
        self.phase_reset_entry = Some(index);
        index
    }

    pub fn get_or_create_entry(
        &mut self,
        signature: &PlayWaveEvent,
        wave_index: Option<WaveIndex>,
    ) -> Result<usize> {
        if let Some(idx) = self.lookup_index_by_signature(signature) {
            Ok(idx)
        } else {
            self.create_entry(signature, wave_index, false)
        }
    }

    /// Returns the command table in JSON format.
    ///
    /// # Returns
    /// * `String` - The command table as a JSON string.
    ///
    fn build_command_table(
        device_type: &DeviceKind,
        mut command_table: Vec<(usize, Value)>,
    ) -> Option<Value> {
        let version = device_type.traits().ct_schema_version;
        let version = match version {
            Some(v) => v,
            None => {
                assert!(
                    command_table.is_empty(),
                    "Internal error: Trying to create command table on unsupported device"
                );
                return None;
            }
        };
        let (ct_schema, version) = match version {
            "hd_1.1.0" => (
                "https://docs.zhinst.com/hdawg/commandtable/v1_1/schema",
                "1.1.0",
            ),
            "sg_1.2.0" => (
                "https://docs.zhinst.com/shfsg/commandtable/v1_2/schema",
                "1.2.0",
            ),
            _ => panic!("Internal error: Unknown command table schema version: {version}",),
        };
        let json_value = json!({
            "$schema": ct_schema,
            "header": {"version": version},
            "table": command_table.iter_mut()
                .map(|(_, value)| value)
                .collect::<Vec<_>>(),
        });
        Some(json_value)
    }

    /// Returns the command table and the parameter phase increment map.
    ///
    /// This function may only be called once.
    ///
    pub fn finish(self) -> Option<CommandTableResults> {
        match CommandTableTracker::build_command_table(&self.device_type, self.command_table) {
            Some(ct) => Some(CommandTableResults {
                command_table: ct,
                parameter_phase_increment_map: self.parameter_phase_increment_map,
            }),
            None => None,
        }
    }

    fn add_oscillator_config(&self, signature: &PlayWaveEvent, json: &mut Map<String, Value>) {
        if let Some(oscillator) = &signature.hw_oscillator {
            json.insert(
                "oscillatorSelect".into(),
                json!({"value": oscillator.index}),
            );
        }
        let mut ct_phase: Option<f64> = None;
        let mut do_incr = false;
        if let Some(increment_phase) = signature.increment_phase {
            ct_phase = Some(increment_phase);
            do_incr = true;
        }
        if let Some(mut ct_phase) = ct_phase {
            ct_phase *= 180.0 / std::f64::consts::PI;
            ct_phase %= 360.0;
            if matches!(self.device_type, DeviceKind::HDAWG) {
                if do_incr {
                    json.insert(
                        "phase0".into(),
                        json!({"value": ct_phase, "increment": true}),
                    );
                    json.insert(
                        "phase1".into(),
                        json!({"value": ct_phase, "increment": true}),
                    );
                } else if matches!(self.signal_kind, AwgKind::SINGLE | AwgKind::DOUBLE) {
                    json.insert("phase0".into(), json!({"value": (ct_phase + 90.0) % 360.0}));
                    json.insert("phase1".into(), json!({"value": (ct_phase + 90.0) % 360.0}));
                } else {
                    json.insert("phase0".into(), json!({"value": (ct_phase + 90.0) % 360.0}));
                    json.insert("phase1".into(), json!({"value": ct_phase}));
                }
            } else if matches!(self.device_type, DeviceKind::SHFSG) {
                if do_incr {
                    json.insert("phase".into(), json!({"value": ct_phase,"increment": true}));
                } else {
                    json.insert("phase".into(), json!({"value": ct_phase}));
                }
            } else {
                panic!(
                    "Internal error: Unsupported device type {:?} for phase configuration",
                    self.device_type
                );
            }
        }
    }

    fn add_amplitude_config(&self, signature: &PlayWaveEvent, json: &mut Map<String, Value>) {
        if let Some(amplitude) = &signature.amplitude {
            let (increment, value) = match amplitude {
                ParameterOperation::SET(value) => (false, value),
                ParameterOperation::INCREMENT(value) => (true, value),
            };
            let mut dd = Map::new();
            dd.insert("value".into(), json!(*value));
            if increment {
                dd.insert("increment".into(), json!(true));
            }
            if matches!(self.device_type, DeviceKind::HDAWG) {
                dd.insert("register".into(), json!(signature.amplitude_register));
                json.insert("amplitude0".into(), serde_json::Value::Object(dd.clone()));
                json.insert("amplitude1".into(), serde_json::Value::Object(dd));
            } else if matches!(self.device_type, DeviceKind::SHFSG) {
                json.insert("amplitude00".into(), serde_json::Value::Object(dd.clone()));
                json.insert("amplitude10".into(), serde_json::Value::Object(dd.clone()));
                json.insert("amplitude11".into(), serde_json::Value::Object(dd));
                let mut dd = Map::new();
                dd.insert("value".into(), json!(-*value));
                if increment {
                    dd.insert("increment".into(), json!(true));
                }
                json.insert("amplitude01".into(), serde_json::Value::Object(dd));
            } else {
                panic!("Unsupported device type for amplitude configuration");
            }
        }
    }
}

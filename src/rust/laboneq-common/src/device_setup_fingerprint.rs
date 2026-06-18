// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

//! This module provides functionality to compute a fingerprint for a QCCS device setup,
//! which is used for comparing different setups and ensuring compatibility between experiments.

use serde::ser::SerializeStruct;

use crate::shfqc;

#[derive(Debug)]
pub enum InstrumentEntryType {
    /// Represents a non-SHFQC instrument type, such as "HDAWG" or "PQSC".
    NonShfqc(String),
    /// Represents an SHFQC, which consists of an SHFSG and optionally an SHFQA.
    /// The `has_qa` field indicates whether the QA channels are defined in the setup for the SHFQC, regardless of whether they are used in the experiment.
    Shfqc { has_qa: bool },
}

#[derive(Debug)]
pub struct InstrumentEntry {
    uid: String,
    kind: InstrumentEntryType,
    options: Vec<String>,
}

impl InstrumentEntry {
    /// Creates a new `InstrumentEntry`.
    pub fn new(uid: impl Into<String>, kind: InstrumentEntryType, options: Vec<String>) -> Self {
        InstrumentEntry {
            uid: uid.into(),
            kind,
            options,
        }
    }
}

/// Computes a fingerprint for a QCCS device setup.
pub fn device_setup_fingerprint(devices: Vec<InstrumentEntry>) -> String {
    let mut target_instruments: Vec<TargetInstrument> = devices
        .iter()
        .flat_map(target_instrument_from_entry)
        .collect();
    for device in target_instruments.iter_mut() {
        device.device_type.make_ascii_uppercase();
        device.device_type = device.device_type.trim().to_string();

        device.options.retain(|opt| !opt.is_empty());
        device
            .options
            .iter_mut()
            .for_each(|opt| opt.make_ascii_uppercase());
    }
    target_instruments.sort_by(|a, b| a.uid.cmp(&b.uid));

    serialize_json(target_instruments)
}

struct TargetInstrument {
    uid: String,
    device_type: String,
    options: Vec<String>,
}

fn target_instrument_from_entry(entry: &InstrumentEntry) -> Vec<TargetInstrument> {
    match &entry.kind {
        InstrumentEntryType::NonShfqc(t) => {
            vec![TargetInstrument {
                uid: entry.uid.clone(),
                device_type: t.clone(),
                options: entry.options.clone(),
            }]
        }
        InstrumentEntryType::Shfqc { has_qa } => {
            let sg = TargetInstrument {
                uid: shfqc::to_sg_uid(&entry.uid),
                device_type: "SHFSG".to_string(),
                options: entry.options.clone(),
            };
            if *has_qa {
                let qa = TargetInstrument {
                    uid: entry.uid.clone(),
                    device_type: "SHFQA".to_string(),
                    options: entry.options.clone(),
                };
                vec![qa, sg]
            } else {
                vec![sg]
            }
        }
    }
}

// --- JSON Serializer ---

fn serialize_json(devices: Vec<TargetInstrument>) -> String {
    let mut buf = Vec::new();
    let mut ser = serde_json::Serializer::with_formatter(&mut buf, SpacedFormatter);
    serde::Serialize::serialize(&devices, &mut ser)
        .expect("TargetInstrument serialization is infallible");
    String::from_utf8(buf).expect("serde_json output is valid UTF-8")
}

impl serde::Serialize for TargetInstrument {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut state = serializer.serialize_struct("InstrumentEntry", 3)?;
        state.serialize_field("uid", &self.uid)?;
        state.serialize_field("type", &self.device_type)?;
        state.serialize_field(
            "options",
            &(!self.options.is_empty()).then(|| self.options.join("/")),
        )?;
        state.end()
    }
}

/// JSON formatter that uses `", "` and `": "` as separators, matching Python's
/// `json.dumps(obj, separators=(', ', ': '))`.
///
/// This is for backwards compatibility with the Python implementation of `device_setup_fingerprint()`
/// for LabOne Q versions < 26.07.
struct SpacedFormatter;

impl serde_json::ser::Formatter for SpacedFormatter {
    fn begin_object_key<W: ?Sized + std::io::Write>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> std::io::Result<()> {
        if !first {
            writer.write_all(b", ")?;
        }
        Ok(())
    }

    fn begin_object_value<W: ?Sized + std::io::Write>(
        &mut self,
        writer: &mut W,
    ) -> std::io::Result<()> {
        writer.write_all(b": ")
    }

    fn begin_array_value<W: ?Sized + std::io::Write>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> std::io::Result<()> {
        if !first {
            writer.write_all(b", ")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sorted_by_uid() {
        let devices1 = vec![
            InstrumentEntry::new(
                "dev_b",
                InstrumentEntryType::NonShfqc("HDAWG".into()),
                vec![],
            ),
            InstrumentEntry::new(
                "dev_a",
                InstrumentEntryType::NonShfqc("SHFQA".into()),
                vec![],
            ),
        ];
        let fingerprint1 = device_setup_fingerprint(devices1);

        let devices2 = vec![
            InstrumentEntry::new(
                "dev_a",
                InstrumentEntryType::NonShfqc("SHFQA".into()),
                vec![],
            ),
            InstrumentEntry::new(
                "dev_b",
                InstrumentEntryType::NonShfqc("HDAWG".into()),
                vec![],
            ),
        ];
        let fingerprint2 = device_setup_fingerprint(devices2);

        assert_eq!(fingerprint1, fingerprint2);
    }

    #[test]
    fn test_empty_setup() {
        assert_eq!(
            device_setup_fingerprint(vec![]),
            device_setup_fingerprint(vec![])
        );
    }

    #[test]
    fn test_same_setup_same_fingerprint() {
        let make = || {
            vec![
                InstrumentEntry::new(
                    "hdawg",
                    InstrumentEntryType::NonShfqc("HDAWG".into()),
                    vec!["HDAWG8".into()],
                ),
                InstrumentEntry::new("pqsc", InstrumentEntryType::NonShfqc("PQSC".into()), vec![]),
            ]
        };
        assert_eq!(
            device_setup_fingerprint(make()),
            device_setup_fingerprint(make())
        );
    }

    #[test]
    fn test_shfqc_with_and_without_qa() {
        let with_qa = vec![InstrumentEntry::new(
            "shfqc1",
            InstrumentEntryType::Shfqc { has_qa: true },
            vec![],
        )];
        let without_qa = vec![InstrumentEntry::new(
            "shfqc1",
            InstrumentEntryType::Shfqc { has_qa: false },
            vec![],
        )];
        assert_ne!(
            device_setup_fingerprint(with_qa),
            device_setup_fingerprint(without_qa)
        );
    }
}

// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::any::Any;
use std::collections::HashMap;
use std::collections::HashSet;

use indexmap::IndexMap;

use laboneq_common::types::AuxiliaryDeviceKind;
use laboneq_common::types::ReferenceClock;
use laboneq_common::types::SignalKind;
use laboneq_compiler_py::compiler_backend::ExperimentView;
use laboneq_compiler_py::compiler_backend::PreprocessOutput;
use laboneq_dsl::device_setup::AuxiliaryDevice;
use laboneq_dsl::device_setup::DeviceSignal;
use laboneq_dsl::device_setup::InstrumentKind;
use laboneq_dsl::signal_calibration::SignalCalibration;
use laboneq_units::duration::Duration;
use laboneq_units::duration::Second;
use laboneq_units::duration::seconds;
use smallvec::SmallVec;

use laboneq_common::types::AwgKey;
use laboneq_common::types::DeviceKind;
use laboneq_compiler_py::compiler_backend::CompilerBackendResult;
use laboneq_compiler_py::compiler_backend::PreprocessedBackendData;
use laboneq_dsl::types::DeviceUid;
use laboneq_dsl::types::SignalUid;
use laboneq_error::{bail, laboneq_error};
use laboneq_ir::system::AwgDevice;

use crate::Result;
use crate::experiment_view::ExperimentViewWrapper;
use crate::ports::parse_port;
use crate::setup_processor::create_awg_devices;

pub struct QccsBackendPreprocessedData {
    awg_devices: Vec<AwgDevice>,
    signals: Vec<BackendSignal>,
    signal_indices: HashMap<SignalUid, usize>,
    lead_delays: HashMap<DeviceUid, Duration<Second>>,
}

impl QccsBackendPreprocessedData {
    fn new(
        signals: Vec<BackendSignal>,
        awg_devices: Vec<AwgDevice>,
        lead_delays: HashMap<DeviceUid, Duration<Second>>,
    ) -> Self {
        let signal_indices = signals
            .iter()
            .enumerate()
            .map(|(i, s)| (s.uid, i))
            .collect::<HashMap<_, _>>();
        QccsBackendPreprocessedData {
            awg_devices,
            signals,
            signal_indices,
            lead_delays,
        }
    }

    pub fn get_signal(&self, signal_uid: SignalUid) -> Option<&BackendSignal> {
        self.signal_indices
            .get(&signal_uid)
            .map(|&index| &self.signals[index])
    }
}

pub struct BackendSignal {
    pub uid: SignalUid,
    pub device_uid: DeviceUid,
    pub channels: SmallVec<[u16; 4]>,
    pub awg_key: AwgKey,
    pub awg_index: u16,
}

impl PreprocessedBackendData for QccsBackendPreprocessedData {
    fn awg_key(&self, signal_uid: SignalUid) -> Result<AwgKey> {
        self.get_signal(signal_uid)
            .map(|s| s.awg_key)
            .ok_or_else(|| laboneq_error!("Expected AWG key for signal UID {}", signal_uid.0))
    }

    fn channels(&self, signal_uid: SignalUid) -> Option<&SmallVec<[u16; 4]>> {
        self.get_signal(signal_uid).map(|s| &s.channels)
    }

    fn lead_delay(&self, signal_uid: SignalUid) -> Duration<Second> {
        self.get_signal(signal_uid)
            .map(|s| {
                self.lead_delays
                    .get(&s.device_uid)
                    .cloned()
                    .unwrap_or_else(|| seconds(0.0))
            })
            .unwrap_or_else(|| seconds(0.0))
    }

    fn awg_devices(&self) -> &[AwgDevice] {
        &self.awg_devices
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[derive(Debug, Default)]
struct PreProcessedSetup<'a> {
    signals: IndexMap<SignalUid, DeviceSignal>,

    awg_devices: IndexMap<DeviceUid, AwgDevice>,
    auxiliary_devices: &'a [AuxiliaryDevice],
}

impl<'a> PreProcessedSetup<'a> {
    fn new(signals: Vec<DeviceSignal>, auxiliary_devices: &'a [AuxiliaryDevice]) -> Self {
        let signals = signals
            .into_iter()
            .map(|s| (s.uid, s))
            .collect::<IndexMap<_, _>>();
        PreProcessedSetup {
            signals,
            awg_devices: IndexMap::new(),
            auxiliary_devices,
        }
    }

    fn get_device(&self, device_uid: DeviceUid) -> Result<&AwgDevice> {
        self.awg_devices
            .get(&device_uid)
            .ok_or_else(|| laboneq_error!("Device with UID {} not found in setup", device_uid.0))
    }

    fn add_devices(&mut self, device: impl Iterator<Item = AwgDevice>) {
        for device in device {
            self.awg_devices.insert(device.uid(), device);
        }
    }

    fn add_signal(&mut self, signal: DeviceSignal) {
        self.signals.insert(signal.uid, signal);
    }

    fn signals(&self) -> impl Iterator<Item = &DeviceSignal> {
        self.signals.values()
    }

    fn devices_in_use(&self) -> Result<Vec<&AwgDevice>> {
        self.signals
            .values()
            .map(|s| self.get_device(s.device_uid))
            .collect()
    }
}

pub(crate) fn preprocess_experiment(
    experiment: ExperimentView,
) -> CompilerBackendResult<PreprocessOutput<QccsBackendPreprocessedData>> {
    let mut experiment = ExperimentViewWrapper::from_experiment_view(experiment);
    validate_setup(&experiment)?;

    // Add triggering signal for small UHFQA+HDAWG setups without sync devices to enable synchronization.
    let hdawg_triggering_signal = create_uhfqa_hdawg_triggering_signal(&mut experiment);
    // Create AWG devices and reassign signals to the created AWG devices.
    let (awg_devices, reassigned_signals) = create_awg_devices(&mut experiment)?;

    let mut preprocessed_setup =
        PreProcessedSetup::new(experiment.signals, experiment.auxiliary_devices);
    if let Some(hdawg_triggering_signal) = hdawg_triggering_signal {
        preprocessed_setup.add_signal(hdawg_triggering_signal);
    }

    reassigned_signals
        .into_iter()
        .for_each(|reassigned_signal| {
            let device_signal = preprocessed_setup
                .signals
                .get_mut(&reassigned_signal.signal)
                .unwrap();
            device_signal.device_uid = reassigned_signal.to_device_uid;
        });
    preprocessed_setup.add_devices(awg_devices.into_iter());

    // Calculate lead delays for all devices in use.
    let lead_delays = calculate_lead_delay(
        preprocessed_setup.devices_in_use()?,
        preprocessed_setup.auxiliary_devices,
    );

    // Process signals
    let mut signals = Vec::with_capacity(preprocessed_setup.signals.len());
    let mut awg_cores: HashMap<(Vec<u16>, DeviceUid), AwgKey> = HashMap::new();
    for signal in preprocessed_setup.signals() {
        let device = preprocessed_setup.get_device(signal.device_uid)?;

        // Extract the channel numbers from the signal's ports and sort them
        let mut channels: SmallVec<[u16; 4]> = signal
            .ports
            .iter()
            .map(|p| parse_port(p, device.kind()).map(|p| p.channel as u16))
            .collect::<Result<_, _>>()?;
        if channels.is_empty() {
            bail!(
                "Signal with UID {} does not have any valid ports",
                signal.uid.0
            );
        }
        channels.sort();

        // Evaluate the AWG index
        let awg_channels = channels
            .iter()
            .map(|ch| eval_awg_number(*ch, device))
            .collect::<HashSet<u16>>();
        let awg_index = if let Some(index) = awg_channels.iter().next() {
            *index
        } else {
            bail!(
                "Signal with UID {} has inconsistent ports, the ports must be neighboring channels e.g. '0' and '1'",
                signal.uid.0
            );
        };

        let awg_id = (awg_channels.into_iter().collect(), signal.device_uid);
        let awg_key = if let Some(awg_key) = awg_cores.get(&awg_id) {
            *awg_key
        } else {
            let new_key = AwgKey(awg_cores.len() as u64);
            awg_cores.insert(awg_id, new_key);
            new_key
        };

        signals.push(BackendSignal {
            uid: signal.uid,
            device_uid: signal.device_uid,
            channels,
            awg_key,
            awg_index,
        });
    }

    let awg_devices = preprocessed_setup.awg_devices.into_values().collect();
    let device_signals = preprocessed_setup.signals;

    Ok(PreprocessOutput::new(
        QccsBackendPreprocessedData::new(signals, awg_devices, lead_delays),
        device_signals.into_values().collect(),
    ))
}

fn eval_awg_number(channel: u16, device: &AwgDevice) -> u16 {
    if device.kind() == DeviceKind::Uhfqa {
        0
    } else {
        channel / device.traits().channels_per_awg
    }
}

/// Create virtual triggering signal for small systems with only UHFQA and HDAWG to enable synchronization.
///
/// HDAWG AWG0 (channels 0/1) is used for triggering in UHFQA+HDAWG systems without sync devices.
/// The signal is only added if there is no existing signal connected to HDAWG channels 0/1 to avoid conflicts with user-defined signals.
fn create_uhfqa_hdawg_triggering_signal(
    experiment: &mut ExperimentViewWrapper,
) -> Option<DeviceSignal> {
    if contains_sync_devices(experiment.auxiliary_devices) {
        return None;
    }
    let all_devices = experiment
        .instruments
        .iter()
        .map(|d| d.kind)
        .collect::<HashSet<_>>();
    let only_hdawg_and_uhfqa = all_devices.contains(&InstrumentKind::Hdawg)
        && all_devices.contains(&InstrumentKind::Uhfqa)
        && all_devices.len() == 2;
    // TODO: Do we still need to add triggering signal for standalone HDAWG as well?
    let hdawg_standalone = all_devices.contains(&InstrumentKind::Hdawg) && all_devices.len() == 1;
    if !only_hdawg_and_uhfqa && !hdawg_standalone {
        return None;
    }

    let first_hdawg = experiment
        .instruments
        .iter()
        .find(|d| d.kind == InstrumentKind::Hdawg)
        .unwrap();
    let mut triggering_signal = None;

    let has_channel_0_on_hdawg = experiment.signals.iter().any(|s| {
        s.device_uid == first_hdawg.uid
            && s.ports.iter().any(|p| p == "SIGOUTS/0" || p == "SIGOUTS/1") // TODO: Proper channel - port converter
    });

    if !has_channel_0_on_hdawg {
        triggering_signal = Some(DeviceSignal {
            uid: experiment
                .id_store
                .get_or_insert("__small_system_trigger__")
                .into(),
            device_uid: first_hdawg.uid,
            ports: vec!["SIGOUTS/0".to_string(), "SIGOUTS/1".to_string()], // TODO: Proper channel - port converter
            kind: SignalKind::Iq,
            calibration: SignalCalibration::default(),
        });
    }
    triggering_signal
}

fn validate_setup(experiment: &ExperimentViewWrapper) -> Result<()> {
    let all_devices = experiment
        .instruments
        .iter()
        .map(|dev| dev.kind)
        .collect::<HashSet<_>>();

    if all_devices.contains(&InstrumentKind::Zqcs) {
        bail!("ZQCS devices are not supported in the QCCS backend");
    }

    let has_sync_devices = contains_sync_devices(experiment.auxiliary_devices);
    if !has_sync_devices
        && all_devices.contains(&InstrumentKind::Hdawg)
        && all_devices.contains(&InstrumentKind::Uhfqa)
    {
        // Check that no internal reference clock is used for UHFQA+HDAWG.
        // TODO: Shall we move this to the Controller? This is the only place where
        // the reference clock is accessed.
        for device in &experiment.instruments {
            if device.kind == InstrumentKind::Hdawg
                && let Some(ReferenceClock::Internal) = device.reference_clock
            {
                bail!(
                    "HDAWG+UHFQA system can only be used with an external clock connected to HDAWG in order to prevent jitter."
                );
            }
        }
    }

    let used_devices = experiment
        .signals
        .iter()
        .map(|s| {
            let device = experiment.get_device_by_uid(s.device_uid)?;
            Ok(device.kind)
        })
        .collect::<Result<HashSet<_>>>()?;

    let has_shf = used_devices.contains(&InstrumentKind::Shfqa)
        || used_devices.contains(&InstrumentKind::Shfsg)
        || used_devices.contains(&InstrumentKind::Shfqc);

    if used_devices.contains(&InstrumentKind::Hdawg)
        && used_devices.contains(&InstrumentKind::Uhfqa)
        && has_shf
    {
        bail!(
            "Setups with signals on each of HDAWG, UHFQA and SHF type instruments are not supported"
        );
    }

    let is_desktop_setup = match used_devices.len() {
        // Allow empty experiment (used in tests)
        0 => true,
        // Standalone single devices are allowed, as well as small setups with only UHFQA and HDAWG or only SHF devices.
        1 => {
            used_devices.contains(&InstrumentKind::Hdawg)
                || used_devices.contains(&InstrumentKind::Uhfqa)
                || used_devices.contains(&InstrumentKind::Shfqa)
                || used_devices.contains(&InstrumentKind::Shfsg)
                || used_devices.contains(&InstrumentKind::Shfqc)
        }
        2 => {
            used_devices.contains(&InstrumentKind::Hdawg)
                && used_devices.contains(&InstrumentKind::Uhfqa)
        }
        _ => false,
    };

    if !is_desktop_setup && !has_sync_devices {
        bail!(
            "Unsupported device combination for small setup: '{:?}'",
            used_devices
        );
    }
    Ok(())
}

/// Check if the setup contains any sync devices (PQSC or QHUB) which can be used for synchronization.
fn contains_sync_devices(devices: &[AuxiliaryDevice]) -> bool {
    devices.iter().any(|i| {
        matches!(
            i.kind(),
            AuxiliaryDeviceKind::Pqsc | AuxiliaryDeviceKind::Qhub
        )
    })
}

/// Calculate lead delays for all devices in use based on the device types and setup configuration.
fn calculate_lead_delay(
    devices_in_use: Vec<&AwgDevice>,
    auxiliary_devices: &[AuxiliaryDevice],
) -> HashMap<DeviceUid, Duration<Second>> {
    use laboneq_common::device_traits::{
        DEFAULT_HDAWG_LEAD_DESKTOP_SETUP, DEFAULT_HDAWG_LEAD_DESKTOP_SETUP_2GHZ,
        DEFAULT_HDAWG_LEAD_PQSC, DEFAULT_HDAWG_LEAD_PQSC_2GHZ, DEFAULT_SHFQA_LEAD_PQSC,
        DEFAULT_SHFSG_LEAD_PQSC, DEFAULT_UHFQA_LEAD_PQSC,
    };
    let has_shf = devices_in_use
        .iter()
        .any(|d| d.kind() == DeviceKind::Shfqa || d.kind() == DeviceKind::Shfsg);
    let has_sync_devices = contains_sync_devices(auxiliary_devices);

    devices_in_use
        .iter()
        .map(|device| {
            let lead_delay = match device.kind() {
                DeviceKind::Hdawg => {
                    let hdawg_uses_2ghz = has_shf;
                    if has_sync_devices {
                        if hdawg_uses_2ghz {
                            DEFAULT_HDAWG_LEAD_PQSC_2GHZ
                        } else {
                            DEFAULT_HDAWG_LEAD_PQSC
                        }
                    } else if hdawg_uses_2ghz {
                        DEFAULT_HDAWG_LEAD_DESKTOP_SETUP_2GHZ
                    } else {
                        DEFAULT_HDAWG_LEAD_DESKTOP_SETUP
                    }
                }
                DeviceKind::Uhfqa => DEFAULT_UHFQA_LEAD_PQSC,
                DeviceKind::Shfqa => DEFAULT_SHFQA_LEAD_PQSC,
                DeviceKind::Shfsg => DEFAULT_SHFSG_LEAD_PQSC,
                _ => panic!("Unsupported device kind for lead delay evaluation"),
            };
            (device.uid(), lead_delay)
        })
        .collect()
}

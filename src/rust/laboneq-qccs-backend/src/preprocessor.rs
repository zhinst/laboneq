// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::any::Any;
use std::collections::HashMap;
use std::collections::HashSet;

use indexmap::IndexMap;

use laboneq_common::device_options::DeviceOptions;
use laboneq_common::types::AuxiliaryDeviceKind;
use laboneq_common::types::ReferenceClock;
use laboneq_common::types::SignalKind;
use laboneq_compiler_py::compiler_backend::DeviceSignal;
use laboneq_compiler_py::compiler_backend::ExperimentView;
use laboneq_compiler_py::compiler_backend::PreprocessOutput;
use laboneq_dsl::device_setup::InstrumentKind;
use laboneq_dsl::setup_description_qccs::AuxiliaryDevice;

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
use crate::experiment_view::ExperimentSignal;
use crate::experiment_view::ExperimentViewWrapper;
use crate::ports::parse_port;
use crate::setup_processor::create_awg_devices;

pub struct QccsBackendPreprocessedData {
    auxiliary_devices: Vec<AuxiliaryDevice>,
    signals: Vec<BackendSignal>,
    signal_indices: HashMap<SignalUid, usize>,
    lead_delays: HashMap<DeviceUid, Duration<Second>>,
    routed_output_channel_map: HashMap<String, u8>,
}

impl QccsBackendPreprocessedData {
    fn new(
        signals: Vec<BackendSignal>,
        auxiliary_devices: Vec<AuxiliaryDevice>,
        lead_delays: HashMap<DeviceUid, Duration<Second>>,
        routed_output_channel_map: HashMap<String, u8>,
    ) -> Self {
        let signal_indices = signals
            .iter()
            .enumerate()
            .map(|(i, s)| (s.uid, i))
            .collect::<HashMap<_, _>>();
        QccsBackendPreprocessedData {
            signals,
            auxiliary_devices,
            signal_indices,
            lead_delays,
            routed_output_channel_map,
        }
    }

    pub fn get_signal(&self, signal_uid: SignalUid) -> Option<&BackendSignal> {
        self.signal_indices
            .get(&signal_uid)
            .map(|&index| &self.signals[index])
    }

    pub(crate) fn auxiliary_devices(&self) -> &[AuxiliaryDevice] {
        &self.auxiliary_devices
    }

    pub(crate) fn signals(&self) -> impl Iterator<Item = &BackendSignal> {
        self.signals.iter()
    }

    pub(crate) fn routed_output_channel_map(&self) -> &HashMap<String, u8> {
        &self.routed_output_channel_map
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

    fn as_any(&self) -> &dyn Any {
        self
    }
}

pub(crate) fn preprocess_experiment(
    experiment: ExperimentView,
) -> CompilerBackendResult<PreprocessOutput<QccsBackendPreprocessedData>> {
    let mut experiment = ExperimentViewWrapper::from_experiment_view(experiment)?;
    fill_missing_device_options(&mut experiment);
    validate_setup(&experiment)?;
    let routed_output_channel_map = resolve_output_routing_map(&experiment)?;

    let hdawg_triggering_signal = create_uhfqa_hdawg_triggering_signal(&mut experiment);
    let (awg_devices, reassigned_signals) = create_awg_devices(&mut experiment)?;

    for reassigned in reassigned_signals {
        experiment
            .get_signal_mut(reassigned.signal)
            .unwrap()
            .device_uid = reassigned.to_device_uid;
    }
    // Add triggering signal to the signal list so it is processed uniformly below.
    if let Some(trigger) = hdawg_triggering_signal {
        experiment.signals.push(trigger);
    }

    let awg_device_map: IndexMap<DeviceUid, AwgDevice> =
        awg_devices.into_iter().map(|d| (d.uid(), d)).collect();

    let lead_delays = calculate_lead_delay(
        &experiment
            .signals
            .iter()
            .filter_map(|s| awg_device_map.get(&s.device_uid))
            .collect::<Vec<_>>(),
        &experiment.auxiliary_devices,
    );

    let mut backend_signals = Vec::with_capacity(experiment.signals.len());
    let mut device_signals = Vec::with_capacity(experiment.signals.len());
    let mut awg_cores: HashMap<(u16, DeviceUid), AwgKey> = HashMap::new();

    for signal in &experiment.signals {
        let device = awg_device_map.get(&signal.device_uid).ok_or_else(|| {
            laboneq_error!("Device with UID {} not found in setup", signal.device_uid.0)
        })?;

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

        let awg_cores_used: HashSet<u16> = channels
            .iter()
            .map(|ch| eval_awg_number(*ch, device))
            .collect();
        if awg_cores_used.len() > 1 {
            bail!(
                "Signal with UID {} has ports spanning multiple AWG cores",
                signal.uid.0
            );
        }
        // safe: channels is non-empty so awg_cores_used has exactly one element
        let awg_index = awg_cores_used.into_iter().next().unwrap();

        let awg_id = (awg_index, signal.device_uid);
        let awg_key = if let Some(awg_key) = awg_cores.get(&awg_id) {
            *awg_key
        } else {
            let new_key = AwgKey(awg_cores.len() as u64);
            awg_cores.insert(awg_id, new_key);
            new_key
        };

        backend_signals.push(BackendSignal {
            uid: signal.uid,
            device_uid: signal.device_uid,
            channels,
            awg_key,
            awg_index,
        });
        device_signals.push(DeviceSignal {
            uid: signal.uid,
            device_uid: signal.device_uid,
            ports: signal.ports.clone(),
            kind: signal.signal_type.clone(),
            calibration: signal.calibration.clone(),
        });
    }

    Ok(PreprocessOutput::new(
        QccsBackendPreprocessedData::new(
            backend_signals,
            experiment.auxiliary_devices,
            lead_delays,
            routed_output_channel_map,
        ),
        device_signals,
        awg_device_map.into_values().collect(),
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
) -> Option<ExperimentSignal> {
    if contains_sync_devices(&experiment.auxiliary_devices) {
        return None;
    }
    let all_devices = experiment
        .instruments
        .values()
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

    let hdawg = experiment
        .instruments
        .values()
        .find(|d| d.kind == InstrumentKind::Hdawg)
        .unwrap();

    let has_channel_0_on_hdawg = experiment.signals.iter().any(|s| {
        s.device_uid == hdawg.uid && s.ports.iter().any(|p| p == "SIGOUTS/0" || p == "SIGOUTS/1") // TODO: Proper channel - port converter
    });

    if has_channel_0_on_hdawg {
        return None;
    }
    Some(ExperimentSignal {
        uid: experiment
            .id_store
            .get_or_insert("__small_system_trigger__")
            .into(),
        device_uid: hdawg.uid,
        ports: vec!["SIGOUTS/0".to_string(), "SIGOUTS/1".to_string()], // TODO: Proper channel - port converter
        signal_type: SignalKind::Iq,
        calibration: SignalCalibration::default(),
    })
}

fn validate_setup(experiment: &ExperimentViewWrapper) -> Result<()> {
    let all_devices = experiment
        .instruments
        .values()
        .map(|dev| dev.kind)
        .collect::<HashSet<_>>();

    if all_devices.contains(&InstrumentKind::Zqcs) {
        bail!("ZQCS devices are not supported in the QCCS backend");
    }

    let has_sync_devices = contains_sync_devices(&experiment.auxiliary_devices);
    if !has_sync_devices
        && all_devices.contains(&InstrumentKind::Hdawg)
        && all_devices.contains(&InstrumentKind::Uhfqa)
    {
        // Check that no internal reference clock is used for UHFQA+HDAWG.
        // TODO: Shall we move this to the Controller? This is the only place where
        // the reference clock is accessed.
        for device in experiment.instruments.values() {
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
    devices_in_use: &[&AwgDevice],
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
                k => panic!("Unsupported device kind for lead delay evaluation: {k:?}"),
            };
            (device.uid(), lead_delay)
        })
        .collect()
}

/// Fill missing device options in the device setup.
///
/// This is a compatibility layer when users have omitted the device options in the device setup.
///
/// TODO: Enforce the options in the device setup and remove this compatibility layer in the future.
fn fill_missing_device_options(experiment_view: &mut ExperimentViewWrapper) {
    // TODO(2K): Add warning for missing options in the device setup
    for instrument in experiment_view.instruments.values_mut() {
        if instrument.options.is_empty() {
            let defaults: &[&'static str] = match instrument.kind {
                InstrumentKind::Uhfqa => &["UHFQA"],
                InstrumentKind::Hdawg => &["HDAWG8"],
                InstrumentKind::Shfqc => &["SHFQC", "QC6CH"],
                InstrumentKind::Shfqa => &["SHFQA2"],
                InstrumentKind::Shfsg => &["SHFSG8"],
                _ => continue,
            };
            instrument.options =
                DeviceOptions::new(defaults.iter().map(|s| s.to_string()).collect());
        }
    }
}

/// Resolve the source channels for all output signals based on the assigned ports.
fn resolve_output_routing_map(experiment: &ExperimentViewWrapper) -> Result<HashMap<String, u8>> {
    let mut routed_output_channel_map = HashMap::new();
    for signal in &experiment.signals {
        for output in signal.calibration.added_outputs.iter() {
            let port = parse_port(&output.source_channel, DeviceKind::Shfsg)?;
            routed_output_channel_map.insert(output.source_channel.clone(), port.channel);
        }
    }
    Ok(routed_output_channel_map)
}

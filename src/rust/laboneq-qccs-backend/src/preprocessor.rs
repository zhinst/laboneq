// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::any::Any;
use std::collections::HashMap;
use std::collections::HashSet;

use laboneq_common::types::AuxiliaryDeviceKind;
use laboneq_common::types::ReferenceClock;
use laboneq_common::types::SignalKind;
use laboneq_compiler_py::compiler_backend::ExperimentView;
use laboneq_dsl::device_setup::DeviceSignal;
use laboneq_dsl::signal_calibration::SignalCalibration;
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
use crate::ports::parse_port;

pub struct QccsBackendPreprocessedData {
    pub signals: Vec<BackendSignal>,
    additional_signals: Vec<DeviceSignal>,
    signal_indices: HashMap<SignalUid, usize>,
}

impl QccsBackendPreprocessedData {
    pub fn new(signals: Vec<BackendSignal>, additional_signals: Vec<DeviceSignal>) -> Self {
        let signal_indices = signals
            .iter()
            .enumerate()
            .map(|(i, s)| (s.uid, i))
            .collect::<HashMap<_, _>>();
        QccsBackendPreprocessedData {
            signals,
            additional_signals,
            signal_indices,
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

    fn additional_signals(&self) -> &[DeviceSignal] {
        &self.additional_signals
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

pub(crate) fn preprocess_experiment(
    mut experiment: ExperimentView,
) -> CompilerBackendResult<QccsBackendPreprocessedData> {
    analyze_setup(&experiment)?;

    let mut signals = Vec::with_capacity(experiment.signals.len());
    let mut awg_cores: HashMap<(Vec<u16>, DeviceUid), AwgKey> = HashMap::new();

    let mut experiment_signals = experiment.signals.iter().collect::<Vec<_>>();

    let additional_signal = create_uhfqa_hdawg_triggering_signal(&mut experiment);
    if let Some(additional_signal) = &additional_signal {
        experiment_signals.push(additional_signal);
    }

    for signal in experiment_signals {
        let device = experiment
            .awg_devices
            .iter()
            .find(|d| d.uid() == signal.device_uid)
            .expect(
                "Signal references a device UID that is not present in the list of AWG devices",
            );

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
            channels,
            awg_key,
            awg_index,
        });
    }

    let additional_signals = additional_signal.into_iter().collect();
    Ok(QccsBackendPreprocessedData::new(
        signals,
        additional_signals,
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
fn create_uhfqa_hdawg_triggering_signal(experiment: &mut ExperimentView) -> Option<DeviceSignal> {
    if has_sync_devices(experiment) {
        return None;
    }
    let all_devices = experiment
        .awg_devices
        .iter()
        .map(|d| d.kind())
        .collect::<HashSet<_>>();
    let only_hdawg_and_uhfqa = all_devices.contains(&DeviceKind::Hdawg)
        && all_devices.contains(&DeviceKind::Uhfqa)
        && all_devices.len() == 2;
    // TODO: Do we still need to add triggering signal for standalone HDAWG as well?
    let hdawg_standalone = all_devices.contains(&DeviceKind::Hdawg) && all_devices.len() == 1;
    if !only_hdawg_and_uhfqa && !hdawg_standalone {
        return None;
    }

    let first_hdawg = experiment
        .awg_devices
        .iter()
        .find(|d| d.kind() == DeviceKind::Hdawg)
        .unwrap();
    let mut triggering_signal = None;

    let has_channel_0_on_hdawg = experiment.signals.iter().any(|s| {
        s.device_uid == first_hdawg.uid()
            && s.ports.iter().any(|p| p == "SIGOUTS/0" || p == "SIGOUTS/1") // TODO: Proper channel - port converter
    });

    if !has_channel_0_on_hdawg {
        triggering_signal = Some(DeviceSignal {
            uid: experiment
                .id_store
                .get_or_insert("__small_system_trigger__")
                .into(),
            device_uid: first_hdawg.uid(),
            ports: vec!["SIGOUTS/0".to_string(), "SIGOUTS/1".to_string()], // TODO: Proper channel - port converter
            kind: SignalKind::Iq,
            calibration: SignalCalibration::default(),
        });
    }
    triggering_signal
}

fn analyze_setup(experiment: &ExperimentView) -> Result<()> {
    let all_devices = experiment
        .awg_devices
        .iter()
        .map(|d| d.kind())
        .collect::<HashSet<_>>();

    let has_sync_devices = has_sync_devices(experiment);

    if !has_sync_devices
        && all_devices.contains(&DeviceKind::Hdawg)
        && all_devices.contains(&DeviceKind::Uhfqa)
    {
        // Check that no internal reference clock is used for UHFQA+HDAWG.
        // TODO: Shall we move this to the Controller? This is the only place where
        // the reference clock is accessed.
        for device in experiment.awg_devices {
            if device.kind() == DeviceKind::Hdawg
                && let Some(ReferenceClock::Internal) = device.reference_clock()
            {
                return Err(laboneq_error!(
                    "HDAWG+UHFQA system can only be used with an external clock connected to HDAWG in order to prevent jitter.",
                ));
            }
        }
    }
    Ok(())
}

fn has_sync_devices(experiment: &ExperimentView) -> bool {
    experiment.auxiliary_devices.iter().any(|i| {
        matches!(
            i.kind(),
            AuxiliaryDeviceKind::Pqsc | AuxiliaryDeviceKind::Qhub
        )
    })
}

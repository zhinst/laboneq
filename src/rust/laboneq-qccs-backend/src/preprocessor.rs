// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::collections::HashSet;
use std::num::NonZeroU16;

use indexmap::IndexMap;

use crate::setup_description::ReferenceClock;
use laboneq_common::device_options::DeviceOptions;
use laboneq_common::device_traits::DeviceTraits;
use laboneq_common::named_id::resolve_ids;
use laboneq_common::types::AuxiliaryDevice;
use laboneq_common::types::AuxiliaryDeviceKind;
use laboneq_common::types::ChannelKey;
use laboneq_common::types::SignalKind;
use laboneq_compiler_py::compiler_backend::DeviceSignal;
use laboneq_compiler_py::compiler_backend::ExperimentView;
use laboneq_compiler_py::compiler_backend::PreprocessOutput;

use laboneq_dsl::signal_calibration::SignalCalibration;
use laboneq_dsl::types::{OscillatorKind, PhysicalChannelUid};
use laboneq_units::duration::Duration;
use laboneq_units::duration::Second;
use laboneq_units::duration::seconds;
use laboneq_units::duration::{Frequency, Hertz};
use smallvec::SmallVec;

use crate::device_kind::DeviceKind;
use laboneq_common::types::AwgKey;
use laboneq_compiler_py::compiler_backend::CompilerBackendResult;
use laboneq_compiler_py::compiler_backend::PreprocessedBackendData;
use laboneq_dsl::types::DeviceUid;
use laboneq_dsl::types::SignalUid;
use laboneq_error::{bail, laboneq_error};

use crate::Result;
use crate::experiment_view::ExperimentSignal;
use crate::experiment_view::ExperimentViewWrapper;
use crate::oscillator::{
    Device as OscillatorDevice, SignalOscillator, resolve_oscillator_modulation,
};
use crate::output_routing::process_output_routing;
use crate::ports::{IoDirection, Port, parse_port};
use crate::ppc_connections::{discard_unusable_amplifier_pumps, resolve_ppc_connections};
use crate::precompensation::{SignalPrecompensation, process_precompensation};
use crate::setup_description::InstrumentKind;
use crate::setup_processor::create_awg_devices;

pub struct QccsBackendPreprocessedData {
    auxiliary_devices: Vec<AuxiliaryDevice>,
    signals: Vec<BackendSignal>,
    signal_indices: HashMap<SignalUid, usize>,
    lead_delays: HashMap<DeviceUid, Duration<Second>>,
    routed_output_channel_map: HashMap<PhysicalChannelUid, u8>,
    /// Devices that form one half of an SHFQC, which appears as a virtual
    /// SHFQA and a virtual SHFSG.
    shfqc_devices: HashSet<DeviceUid>,
    device_kinds: IndexMap<DeviceUid, DeviceKind>,
    device_options: HashMap<DeviceUid, DeviceOptions>,
}

pub(crate) struct SignalProperties<'a> {
    pub uid: SignalUid,
    pub device_uid: DeviceUid,
    pub awg_key: AwgKey,
    pub kind: SignalKind,
    pub device_kind: DeviceKind,
    /// Whether the device has the "LRT" option.
    pub has_lrt: bool,
    pub calibration: &'a SignalCalibration,
}

impl SignalProperties<'_> {
    /// Traits of the device the signal plays on.
    pub(crate) fn device_traits(&self) -> &'static DeviceTraits {
        self.device_kind.traits()
    }

    /// Whether the signal is modulated by an oscillator on the device.
    pub(crate) fn is_hardware_modulated(&self) -> bool {
        self.calibration
            .oscillator
            .as_ref()
            .is_some_and(|osc| matches!(osc.kind, OscillatorKind::Hardware))
    }
}

impl QccsBackendPreprocessedData {
    fn new(
        signals: Vec<BackendSignal>,
        auxiliary_devices: Vec<AuxiliaryDevice>,
        lead_delays: HashMap<DeviceUid, Duration<Second>>,
        routed_output_channel_map: HashMap<PhysicalChannelUid, u8>,
        shfqc_devices: HashSet<DeviceUid>,
        device_kinds: IndexMap<DeviceUid, DeviceKind>,
        device_options: HashMap<DeviceUid, DeviceOptions>,
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
            shfqc_devices,
            device_kinds,
            device_options,
        }
    }

    /// What this backend made of each signal.
    pub(crate) fn signal_properties(&self) -> impl Iterator<Item = SignalProperties<'_>> {
        self.signals.iter().map(properties_of)
    }

    /// What this backend made of one signal.
    pub(crate) fn signal(&self, signal_uid: SignalUid) -> Option<SignalProperties<'_>> {
        self.get_signal(signal_uid).map(properties_of)
    }

    /// The model each device in the setup is, in setup order.
    pub(crate) fn device_kinds(&self) -> &IndexMap<DeviceUid, DeviceKind> {
        &self.device_kinds
    }

    /// The options ordered with the instrument that the device belongs to.
    pub(crate) fn device_options(&self, device_uid: DeviceUid) -> Option<&DeviceOptions> {
        self.device_options.get(&device_uid)
    }

    /// Whether the device is one half of an SHFQC.
    pub(crate) fn is_shfqc(&self, device_uid: DeviceUid) -> bool {
        self.shfqc_devices.contains(&device_uid)
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

    pub(crate) fn routed_output_channel_map(&self) -> &HashMap<PhysicalChannelUid, u8> {
        &self.routed_output_channel_map
    }

    /// Lead delays of the devices in use.
    pub(crate) fn lead_delays(&self) -> &HashMap<DeviceUid, Duration<Second>> {
        &self.lead_delays
    }
}

fn properties_of(signal: &BackendSignal) -> SignalProperties<'_> {
    SignalProperties {
        uid: signal.uid,
        device_uid: signal.device_uid,
        awg_key: signal.awg_key,
        kind: signal.kind.clone(),
        device_kind: signal.device_kind,
        has_lrt: signal.has_lrt,
        calibration: &signal.calibration,
    }
}

pub struct BackendSignal {
    pub uid: SignalUid,
    pub device_uid: DeviceUid,
    pub device_kind: DeviceKind,
    pub kind: SignalKind,
    pub has_lrt: bool,
    pub sampling_rate: Frequency<Hertz>,
    /// The calibration of the signal, as this backend handed it to the compiler.
    pub calibration: SignalCalibration,
    /// Device channels the signal occupies, sorted.
    pub channels: SmallVec<[u16; 4]>,
    pub awg_key: AwgKey,
    pub awg_index: u16,
    pub ppc_channel: Option<ChannelKey>,
}

impl PreprocessedBackendData for QccsBackendPreprocessedData {
    fn awg_key(&self, signal_uid: SignalUid) -> Result<AwgKey> {
        self.get_signal(signal_uid)
            .map(|s| s.awg_key)
            .ok_or_else(|| laboneq_error!("Expected AWG key for signal UID {}", signal_uid.0))
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
}

pub(crate) fn preprocess_experiment(
    experiment: ExperimentView,
) -> CompilerBackendResult<PreprocessOutput<QccsBackendPreprocessedData>> {
    let mut experiment = ExperimentViewWrapper::from_experiment_view(experiment)?;
    let setup_fingerprint = experiment.device_setup_fingerprint.clone();
    fill_missing_device_options(&mut experiment);
    validate_setup(&experiment)?;
    let routed_outputs = process_output_routing(&experiment)?;
    let ppc_connections = resolve_ppc_connections(&experiment)?;
    discard_unusable_amplifier_pumps(
        &mut experiment.signals,
        &ppc_connections,
        experiment.id_store,
    );

    let hdawg_triggering_signal = create_uhfqa_hdawg_triggering_signal(&mut experiment);
    let awg_devices = create_awg_devices(&mut experiment)?;
    let shfqc_devices = awg_devices.shfqc_devices;
    let device_kinds = awg_devices.kinds;
    let device_options = awg_devices.options;
    let physical_device_uids = awg_devices.physical_uids;

    for reassigned in awg_devices.reassigned_signals {
        experiment
            .get_signal_mut(reassigned.signal)
            .unwrap()
            .device_uid = reassigned.to_device_uid;
    }
    // Add triggering signal to the signal list so it is processed uniformly below.
    if let Some(trigger) = hdawg_triggering_signal {
        experiment.signals.push(trigger);
    }

    let lead_delays = calculate_lead_delay(
        &experiment
            .signals
            .iter()
            .filter_map(|s| Some((s.device_uid, *device_kinds.get(&s.device_uid)?)))
            .collect::<Vec<_>>(),
        &experiment.auxiliary_devices,
    );

    let setup_has_shf = device_kinds.values().any(DeviceKind::is_shf);

    let mut backend_signals = Vec::with_capacity(experiment.signals.len());
    let mut device_signals = Vec::with_capacity(experiment.signals.len());
    let mut awg_cores: HashMap<(u16, DeviceUid), AwgKey> = HashMap::new();

    for signal in &experiment.signals {
        let device_kind = *device_kinds.get(&signal.device_uid).ok_or_else(|| {
            laboneq_error!("Device with UID {} not found in setup", signal.device_uid.0)
        })?;
        let device_traits = device_kind.traits();
        let physical_device_uid = *physical_device_uids
            .get(&signal.device_uid)
            .expect("Every device in the setup has an instrument");

        validate_iq_port_order(device_kind, &signal.ports)
            .map_err(|e| laboneq_error!("Signal with UID {}: {}", signal.uid.0, e))?;

        let mut channels: SmallVec<[u16; 4]> =
            signal.ports.iter().map(|p| p.channel as u16).collect();
        if channels.is_empty() {
            bail!(
                "Signal with UID {} does not have any valid ports",
                signal.uid.0
            );
        }
        channels.sort();

        let awg_cores_used: HashSet<u16> = channels
            .iter()
            .map(|ch| eval_awg_number(*ch, device_traits, device_kind))
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

        let sampling_rate = device_kind.sampling_rate(setup_has_shf);

        check_local_oscillator(signal, device_kind)?;
        let signal_kind = signal_kind(&signal.ports, device_kind)?;

        backend_signals.push(BackendSignal {
            uid: signal.uid,
            device_uid: signal.device_uid,
            device_kind,
            kind: signal_kind.clone(),
            has_lrt: contains_option(&device_options, signal.device_uid, "LRT"),
            sampling_rate,
            channels,
            awg_key,
            awg_index,
            ppc_channel: ppc_connections.get(&signal.uid).copied(),
            // Filled in below, once the passes over the device signals have
            // settled the calibration.
            calibration: SignalCalibration::default(),
        });

        device_signals.push(DeviceSignal {
            uid: signal.uid,
            device_uid: signal.device_uid,
            physical_device_uid,
            traits: device_traits,
            kind: signal_kind,
            calibration: signal.calibration.clone(),
            delay_signal: routed_outputs
                .delay_signal
                .get(&signal.uid)
                .copied()
                .unwrap_or(0),
            sampling_rate,
            sample_multiple: NonZeroU16::new(device_traits.sample_multiple)
                .expect("device traits define a non-zero sequencer grid"),
        });
    }

    process_precompensations(&backend_signals, &mut device_signals, experiment.id_store)?;

    let mut oscillators =
        signal_oscillators(&mut device_signals, &backend_signals, &device_options);
    resolve_oscillator_modulation(
        &mut oscillators,
        experiment.root,
        experiment.pulses,
        experiment.id_store,
    )?;

    for (backend_signal, device_signal) in backend_signals.iter_mut().zip(&device_signals) {
        backend_signal.calibration = device_signal.calibration.clone();
    }

    Ok(PreprocessOutput::new(
        QccsBackendPreprocessedData::new(
            backend_signals,
            experiment.auxiliary_devices,
            lead_delays,
            routed_outputs.channel_map,
            shfqc_devices,
            device_kinds,
            device_options,
        ),
        device_signals,
        setup_fingerprint,
    ))
}

fn contains_option(
    device_options: &HashMap<DeviceUid, DeviceOptions>,
    device_uid: DeviceUid,
    option: &'static str,
) -> bool {
    device_options
        .get(&device_uid)
        .is_some_and(|options| options.contains(option))
}

/// Pair each device signal with the device and channel information needed by the
/// oscillator modulation resolution. `backend_signals` describes the same signals in the
/// same order as `device_signals`.
fn signal_oscillators<'a>(
    device_signals: &'a mut [DeviceSignal],
    backend_signals: &'a [BackendSignal],
    device_options: &HashMap<DeviceUid, DeviceOptions>,
) -> Vec<SignalOscillator<'a>> {
    debug_assert_eq!(device_signals.len(), backend_signals.len());
    device_signals
        .iter_mut()
        .zip(backend_signals)
        .map(|(signal, backend)| SignalOscillator {
            uid: signal.uid,
            kind: &signal.kind,
            sampling_rate: signal.sampling_rate.value(),
            device: OscillatorDevice {
                uid: backend.device_uid,
                kind: backend.device_kind,
                has_lrt: contains_option(device_options, backend.device_uid, "LRT"),
            },
            channels: &backend.channels,
            oscillator: signal.calibration.oscillator.as_mut(),
        })
        .collect()
}

fn process_precompensations(
    backend_signals: &[BackendSignal],
    device_signals: &mut [DeviceSignal],
    id_store: &laboneq_common::named_id::NamedIdStore,
) -> Result<()> {
    let precompensations: Vec<SignalPrecompensation> = backend_signals
        .iter()
        .zip(device_signals.iter_mut())
        .map(|(bs, ds)| SignalPrecompensation {
            signal_uid: bs.uid,
            awg_key: bs.awg_key,
            device_kind: bs.device_kind,
            sampling_rate: ds.sampling_rate.value(),
            precompensation: ds.calibration.precompensation.take(),
        })
        .collect();
    let result = process_precompensation(precompensations, id_store)?;
    for warning in result.warnings {
        laboneq_log::warn!("{}", resolve_ids(&warning.to_string(), id_store));
    }
    for (uid, precompensation) in result.adapted {
        if let Some(signal) = device_signals.iter_mut().find(|s| s.uid == uid) {
            signal.delay_signal += result.delays.get(&uid).copied().unwrap_or(0);
            signal.calibration.precompensation = precompensation;
        }
    }
    Ok(())
}

fn eval_awg_number(channel: u16, traits: &DeviceTraits, kind: DeviceKind) -> u16 {
    if kind == DeviceKind::Uhfqa {
        0
    } else {
        channel / traits.channels_per_awg
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

    let has_channel_0_on_hdawg = experiment
        .signals
        .iter()
        .any(|s| s.device_uid == hdawg.uid && s.ports.iter().any(|p| [0, 1].contains(&p.channel)));

    if has_channel_0_on_hdawg {
        return None;
    }
    Some(ExperimentSignal {
        uid: experiment
            .id_store
            .get_or_insert("__small_system_trigger__")
            .into(),
        device_uid: hdawg.uid,
        ports: vec![
            parse_port("SIGOUTS/0", hdawg.kind).unwrap(),
            parse_port("SIGOUTS/1", hdawg.kind).unwrap(),
        ],
        calibration: SignalCalibration::default(),
    })
}

fn validate_setup(experiment: &ExperimentViewWrapper) -> Result<()> {
    let all_devices = experiment
        .instruments
        .values()
        .map(|dev| dev.kind)
        .collect::<HashSet<_>>();

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
    devices_in_use: &[(DeviceUid, DeviceKind)],
    auxiliary_devices: &[AuxiliaryDevice],
) -> HashMap<DeviceUid, Duration<Second>> {
    use laboneq_common::device_traits::{
        DEFAULT_HDAWG_LEAD_DESKTOP_SETUP, DEFAULT_HDAWG_LEAD_DESKTOP_SETUP_2GHZ,
        DEFAULT_HDAWG_LEAD_PQSC, DEFAULT_HDAWG_LEAD_PQSC_2GHZ, DEFAULT_SHFQA_LEAD_PQSC,
        DEFAULT_SHFSG_LEAD_PQSC, DEFAULT_UHFQA_LEAD_PQSC,
    };
    let has_shf = devices_in_use
        .iter()
        .any(|(_, kind)| matches!(kind, DeviceKind::Shfqa | DeviceKind::Shfsg));
    let has_sync_devices = contains_sync_devices(auxiliary_devices);

    devices_in_use
        .iter()
        .map(|(uid, kind)| {
            let lead_delay = match kind {
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
            };
            (*uid, lead_delay)
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

fn check_local_oscillator(signal: &ExperimentSignal, instrument: DeviceKind) -> Result<()> {
    if signal.calibration.lo_frequency.is_some() && !instrument.is_shf() {
        bail!(
            "Signal '{}' sets a local oscillator, which is unsupported on this channel \
            type. Remove the 'local_oscillator' calibration from this signal.",
            signal.uid.0,
        );
    }
    Ok(())
}

fn signal_kind(ports: &[Port], instrument: DeviceKind) -> Result<SignalKind> {
    if ports.len() > 1
        && ports
            .iter()
            .map(|p| p.direction)
            .collect::<HashSet<_>>()
            .len()
            != 1
    {
        bail!("Signal ports must all have the same direction (input or output)");
    }

    if ports
        .iter()
        .all(|p| matches!(p.direction, IoDirection::Input))
    {
        return Ok(SignalKind::Integration);
    }
    if instrument == DeviceKind::Hdawg && ports.len() == 1 {
        return Ok(SignalKind::Rf);
    }
    Ok(SignalKind::Iq)
}

/// Checks the port order of an HDAWG or UHFQA IQ output signal.
///
/// Both device kinds pair two `SIGOUTS` ports into one IQ signal: the first
/// port carries the I component, the second the Q component, and the I
/// channel must have the lower channel number. Devices other than HDAWG and
/// UHFQA, and signals that are not an IQ pair of theirs, are always accepted.
fn validate_iq_port_order(
    device_kind: DeviceKind,
    ports: &[Port],
) -> std::result::Result<(), String> {
    if !matches!(device_kind, DeviceKind::Hdawg | DeviceKind::Uhfqa)
        || ports.len() != 2
        || ports.iter().any(|p| p.direction != IoDirection::Output)
    {
        return Ok(());
    }
    if ports[0].channel > ports[1].channel {
        return Err(format!(
            "has an IQ pair with reversed ports: the I channel (port {}) \
             must have a lower channel number than the Q channel (port {})",
            ports[0].channel, ports[1].channel
        ));
    }
    Ok(())
}

#[cfg(test)]
mod port_order_tests {
    use super::*;

    fn port(channel: u8, direction: IoDirection) -> Port {
        Port {
            path: format!("SIGOUTS/{channel}"),
            channel,
            direction,
        }
    }

    #[test]
    fn accepts_ascending_iq_ports() {
        let ports = [port(0, IoDirection::Output), port(1, IoDirection::Output)];
        assert!(validate_iq_port_order(DeviceKind::Hdawg, &ports).is_ok());
        assert!(validate_iq_port_order(DeviceKind::Uhfqa, &ports).is_ok());
    }

    #[test]
    fn rejects_descending_iq_ports() {
        let ports = [port(1, IoDirection::Output), port(0, IoDirection::Output)];
        assert!(validate_iq_port_order(DeviceKind::Hdawg, &ports).is_err());
        assert!(validate_iq_port_order(DeviceKind::Uhfqa, &ports).is_err());
    }

    #[test]
    fn ignores_single_rf_port() {
        let ports = [port(0, IoDirection::Output)];
        assert!(validate_iq_port_order(DeviceKind::Hdawg, &ports).is_ok());
    }

    #[test]
    fn ignores_acquire_input_ports() {
        let ports = [port(1, IoDirection::Input), port(0, IoDirection::Input)];
        assert!(validate_iq_port_order(DeviceKind::Hdawg, &ports).is_ok());
        assert!(validate_iq_port_order(DeviceKind::Uhfqa, &ports).is_ok());
    }

    #[test]
    fn ignores_devices_without_iq_output_pairs() {
        let ports = [port(1, IoDirection::Output), port(0, IoDirection::Output)];
        assert!(validate_iq_port_order(DeviceKind::Shfsg, &ports).is_ok());
        assert!(validate_iq_port_order(DeviceKind::Shfqa, &ports).is_ok());
    }
}

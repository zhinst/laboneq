// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

//! Resolution of `AUTO` oscillator modulation into a concrete modulation type.

use std::collections::{HashMap, HashSet};

use laboneq_common::named_id::NamedIdStore;
use laboneq_common::types::SignalKind;

use crate::device_kind::DeviceKind;
use laboneq_dsl::ExperimentNode;
use laboneq_dsl::operation::Operation;
use laboneq_dsl::types::{
    AcquisitionType, DeviceUid, Oscillator, OscillatorKind, PulseDef, PulseLength, PulseUid,
    SignalUid,
};
use laboneq_error::laboneq_error;
use laboneq_log::{info, warn};
use laboneq_units::duration::{Duration, Second};

use crate::Result;

/// Threshold above which AUTO modulation resolves to HARDWARE on QA devices
/// with LRT option. Below this, SOFTWARE is used for integration mode.
/// This corresponds to 4096 samples at the SHFQA's 2 GHz sampling rate.
const LRT_HW_MODULATION_THRESHOLD: f64 = 4096.0 / 2e9; // 2.048 us

/// A signal and its oscillator, as seen by the modulation resolution pass.
///
/// Signals without an oscillator take part as well: their acquisitions determine how
/// the other signals on the same hardware channel resolve.
pub(crate) struct SignalOscillator<'a> {
    pub uid: SignalUid,
    pub kind: &'a SignalKind,
    pub sampling_rate: f64,
    pub device: Device,
    /// Sorted device channels the signal occupies.
    pub channels: &'a [u16],
    pub oscillator: Option<&'a mut Oscillator>,
}

/// Device properties relevant to oscillator modulation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Device {
    pub uid: DeviceUid,
    pub kind: DeviceKind,
    pub has_lrt: bool,
}

/// Resolves the modulation type for oscillators with AUTO modulation based on the
/// device capabilities and the acquisitions performed by the experiment.
///
/// # Errors
/// Returns an error if the experiment has no real-time averaging loop to resolve against,
/// if the resolution is impossible for a signal, or if an acquisition refers to an unknown
/// kernel pulse.
pub(crate) fn resolve_oscillator_modulation(
    signals: &mut [SignalOscillator],
    root: &ExperimentNode,
    pulses: &HashMap<PulseUid, PulseDef>,
    id_store: &NamedIdStore,
) -> Result<()> {
    let acquisitions = Acquisitions::collect(root, pulses)?;
    let acquisition_type = acquisitions.acquisition_type.ok_or_else(|| {
        laboneq_error!("Experiment must have exactly one real time acquisition loop.")
    })?;
    let is_spectroscopy = acquisition_type.is_spectroscopy();

    // Pre-compute which HW channels have any SHFQA signal with a long acquire.
    // This ensures that all signals sharing a HW channel (e.g. measure + acquire)
    // get consistent modulation resolution.
    let channels_with_long_readout: HashSet<HwChannel> = signals
        .iter()
        .filter(|signal| signal.device.kind == DeviceKind::Shfqa)
        .filter(|signal| {
            acquisitions
                .max_length_seconds(signal.uid, signal.sampling_rate)
                .is_some_and(|length| length.value() > LRT_HW_MODULATION_THRESHOLD)
        })
        .map(SignalOscillator::hw_channel)
        .collect();

    for signal in signals {
        let has_long_readout = channels_with_long_readout.contains(&signal.hw_channel());
        let is_rf = signal.kind == &SignalKind::Rf;
        let (uid, device) = (signal.uid, signal.device);
        let Some(osc) = signal.oscillator.as_mut() else {
            continue;
        };
        if osc.kind != OscillatorKind::Auto {
            continue;
        }
        osc.kind = match device.kind {
            DeviceKind::Shfqa => {
                if !has_long_readout {
                    if is_spectroscopy {
                        OscillatorKind::Hardware
                    } else {
                        OscillatorKind::Software
                    }
                } else if !is_spectroscopy && !device.has_lrt {
                    return Err(laboneq_error!(
                        "Acquisition length on signal '{}' exceeds \
                        {} (4096 samples) and \
                        requires hardware modulation, but the device \
                        '{}' does not have the LRT option \
                        installed. Either reduce the acquisition length or \
                        set the oscillator modulation type explicitly.",
                        uid.0,
                        LRT_HW_MODULATION_THRESHOLD,
                        device.uid.0
                    ));
                } else {
                    // Long readout requires HW modulation. In RAW mode the oscillator
                    // phase advances between shots, averaging the signal out.
                    if acquisition_type == AcquisitionType::Raw {
                        warn!(
                            "Oscillator '{}' on signal \
                                 '{}' resolved to HARDWARE modulation \
                                 in RAW acquisition mode. Set \
                                reset_oscillator_phase=True on the \
                                acquire_loop_rt, or use \
                                ModulationType.SOFTWARE explicitly, to avoid \
                                the signal averaging out.",
                            id_store.resolve_unchecked(osc.uid),
                            id_store.resolve_unchecked(uid)
                        );
                    }
                    OscillatorKind::Hardware
                }
            }
            DeviceKind::Uhfqa => {
                if is_spectroscopy {
                    OscillatorKind::Hardware
                } else {
                    OscillatorKind::Software
                }
            }
            // For HDAWG RF signals, SW modulation tends to be more useful
            DeviceKind::Hdawg if is_rf => OscillatorKind::Software,
            _ => OscillatorKind::Hardware,
        };
        info!(
            "Resolved modulation type of oscillator on signal: '{}' to {}",
            id_store.resolve_unchecked(uid),
            osc.kind
        );
    }
    Ok(())
}

/// Hardware channel a signal occupies. Signals sharing one resolve consistently.
type HwChannel<'a> = (DeviceUid, &'a [u16]);

impl<'a> SignalOscillator<'a> {
    fn hw_channel(&self) -> HwChannel<'a> {
        (self.device.uid, self.channels)
    }
}

/// Acquisitions performed by the experiment, as far as oscillator resolution is concerned.
#[derive(Debug)]
struct Acquisitions {
    /// Acquisition type of the experiment's real-time averaging loop. The compiler accepts
    /// exactly one such loop, so the first found is the only one.
    ///
    /// Once several real-time executions are supported, the modulation will have to be
    /// resolved per execution rather than once for the whole experiment.
    acquisition_type: Option<AcquisitionType>,
    /// The longest acquisition requested on each signal.
    max_lengths: HashMap<SignalUid, MaxAcquisitionLength>,
}

impl Acquisitions {
    /// Collect the acquisition type and the longest acquisition per signal from the experiment.
    ///
    /// # Errors
    /// Returns an error if an acquisition refers to an unknown kernel pulse.
    fn collect(root: &ExperimentNode, pulses: &HashMap<PulseUid, PulseDef>) -> Result<Self> {
        let mut acquisitions = Acquisitions {
            acquisition_type: None,
            max_lengths: HashMap::new(),
        };
        acquisitions.visit(root, pulses)?;
        Ok(acquisitions)
    }

    /// The longest acquisition on `signal`, in seconds, for the given sampling rate in Hz.
    ///
    /// `None` if the experiment states no acquisition length for the signal.
    fn max_length_seconds(
        &self,
        signal: SignalUid,
        sampling_rate: f64,
    ) -> Option<Duration<Second>> {
        self.max_lengths
            .get(&signal)
            .map(|length| length.max_seconds(sampling_rate))
    }

    fn visit(&mut self, node: &ExperimentNode, pulses: &HashMap<PulseUid, PulseDef>) -> Result<()> {
        match &node.kind {
            Operation::AveragingLoop(loop_) if self.acquisition_type.is_none() => {
                self.acquisition_type = Some(loop_.acquisition_type);
            }
            // An explicit length takes precedence over the kernel pulse lengths.
            Operation::Acquire(acquire) => {
                if let Some(seconds) = acquire.length {
                    let length = self.max_lengths.entry(acquire.signal).or_default();
                    length.seconds = length.seconds.max(seconds);
                } else {
                    for kernel in &acquire.kernel {
                        let pulse = pulses.get(kernel).ok_or_else(|| {
                            laboneq_error!("Kernel pulse '{}' not found for acquisition.", kernel.0)
                        })?;
                        let length = self.max_lengths.entry(acquire.signal).or_default();
                        match pulse.length() {
                            PulseLength::Seconds(seconds) => {
                                length.seconds = length.seconds.max(seconds)
                            }
                            PulseLength::Samples(samples) => {
                                length.samples = length.samples.max(samples.value())
                            }
                        }
                    }
                }
            }
            _ => {}
        }
        for child in &node.children {
            self.visit(child, pulses)?;
        }
        Ok(())
    }
}

/// Longest acquisition requested on a signal.
///
/// Lengths may be given either in seconds or in samples; both are tracked, as
/// converting between them requires the signal's sampling rate.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct MaxAcquisitionLength {
    seconds: Duration<Second>,
    samples: usize,
}

impl MaxAcquisitionLength {
    /// The longer of the two lengths, in seconds, at the given sampling rate in Hz.
    fn max_seconds(&self, sampling_rate: f64) -> Duration<Second> {
        self.seconds
            .value()
            .max(self.samples as f64 / sampling_rate)
            .into()
    }
}
#[cfg(test)]
mod tests {
    use std::num::NonZeroU32;

    use laboneq_dsl::operation::{Acquire, AveragingLoop};
    use laboneq_dsl::types::{
        AveragingMode, HandleUid, NumericLiteral, PulseKind, RepetitionMode, SampledPulse,
        SectionAlignment, SectionTimingMode,
    };

    use super::*;

    const SHFQA_SAMPLING_RATE: f64 = 2e9;
    /// Above the 2.048 us (4096 samples) long readout threshold.
    const LONG_ACQUIRE: f64 = 3e-6;

    fn acquire_node(
        signal: SignalUid,
        length: Option<f64>,
        kernel: Vec<PulseUid>,
    ) -> ExperimentNode {
        ExperimentNode::new(Operation::Acquire(Acquire {
            signal,
            handle: HandleUid::from(0),
            length: length.map(Into::into),
            kernel,
            parameters: vec![],
            pulse_parameters: vec![],
        }))
    }

    fn averaging_loop_node(
        store: &mut NamedIdStore,
        acquisition_type: AcquisitionType,
        children: Vec<ExperimentNode>,
    ) -> ExperimentNode {
        let mut node = ExperimentNode::new(Operation::AveragingLoop(AveragingLoop {
            uid: store.get_or_insert("rt_loop").into(),
            count: NonZeroU32::new(1).unwrap(),
            acquisition_type,
            averaging_mode: AveragingMode::Cyclic,
            repetition_mode: RepetitionMode::Fastest,
            reset_oscillator_phase: false,
            alignment: SectionAlignment::Left,
            section_timing_mode: SectionTimingMode::default(),
        }));
        node.children = children;
        node
    }

    fn pulse(uid: PulseUid, kind: PulseKind) -> PulseDef {
        PulseDef {
            uid,
            kind,
            can_compress: false,
            amplitude: NumericLiteral::Float(1.0),
        }
    }

    /// A signal without an oscillator still contributes its acquisition length to the
    /// hardware channel it shares with other signals.
    #[test]
    fn test_long_readout_from_signal_without_oscillator() {
        let mut store = NamedIdStore::new();
        let acquire: SignalUid = store.get_or_insert("q0/acquire").into();
        let measure: SignalUid = store.get_or_insert("q0/measure").into();
        let device = Device {
            uid: store.get_or_insert("shfqa").into(),
            kind: DeviceKind::Shfqa,
            has_lrt: true,
        };
        let mut oscillator = Oscillator {
            uid: store.get_or_insert("q0/measure/osc").into(),
            frequency: 1e6.into(),
            kind: OscillatorKind::Auto,
        };
        let root = averaging_loop_node(
            &mut store,
            AcquisitionType::Integration,
            vec![acquire_node(acquire, Some(LONG_ACQUIRE), vec![])],
        );

        // The acquire signal carries the long acquisition but has no oscillator; the
        // measure signal shares its hardware channel.
        let (integration, iq) = (SignalKind::Integration, SignalKind::Iq);
        let mut signals = vec![
            SignalOscillator {
                uid: acquire,
                kind: &integration,
                sampling_rate: SHFQA_SAMPLING_RATE,
                device,
                channels: &[0],
                oscillator: None,
            },
            SignalOscillator {
                uid: measure,
                kind: &iq,
                sampling_rate: SHFQA_SAMPLING_RATE,
                device,
                channels: &[0],
                oscillator: Some(&mut oscillator),
            },
        ];
        resolve_oscillator_modulation(&mut signals, &root, &HashMap::new(), &store).unwrap();

        assert_eq!(oscillator.kind, OscillatorKind::Hardware);
    }

    #[test]
    fn test_acquisitions_explicit_length_wins_over_kernel() {
        let mut store = NamedIdStore::new();
        let signal: SignalUid = store.get_or_insert("q0/acquire").into();
        let kernel: PulseUid = store.get_or_insert("kernel").into();
        let pulses = HashMap::from([(
            kernel,
            pulse(
                kernel,
                PulseKind::LengthOnly {
                    length: 5e-6.into(),
                },
            ),
        )]);

        let mut root = acquire_node(signal, Some(1e-6), vec![kernel]);
        root.children = vec![acquire_node(signal, Some(2e-6), vec![])];

        let acquisitions = Acquisitions::collect(&root, &pulses).unwrap();
        // The longest explicit length wins; the kernel is not consulted.
        assert_eq!(
            acquisitions.max_length_seconds(signal, SHFQA_SAMPLING_RATE),
            Some(2e-6.into())
        );
        // Signals with no acquisition length to speak of are absent, rather than zero.
        assert_eq!(
            acquisitions.max_length_seconds(SignalUid::from(store.get_or_insert("other")), 2e9),
            None
        );
        let without_length: SignalUid = store.get_or_insert("q1/acquire").into();
        let acquisitions =
            Acquisitions::collect(&acquire_node(without_length, None, vec![]), &pulses).unwrap();
        assert_eq!(
            acquisitions.max_length_seconds(without_length, SHFQA_SAMPLING_RATE),
            None
        );
    }

    /// A sampled kernel defines the acquisition length in samples; converting it to
    /// seconds needs the signal's sampling rate.
    #[test]
    fn test_acquisitions_from_sampled_kernel() {
        let mut store = NamedIdStore::new();
        let signal: SignalUid = store.get_or_insert("q0/acquire").into();
        let kernel: PulseUid = store.get_or_insert("kernel").into();
        let pulses = HashMap::from([(
            kernel,
            pulse(
                kernel,
                PulseKind::Sampled(SampledPulse {
                    samples: vec![0.0; 8192].into(),
                }),
            ),
        )]);

        let acquisitions =
            Acquisitions::collect(&acquire_node(signal, None, vec![kernel]), &pulses).unwrap();
        assert_eq!(
            acquisitions.max_length_seconds(signal, SHFQA_SAMPLING_RATE),
            Some(4.096e-6.into())
        );
    }

    #[test]
    fn test_max_acquisition_length_in_samples() {
        let length = MaxAcquisitionLength {
            seconds: 100e-9.into(),
            samples: 512,
        };
        // 512 samples at 2 GHz is 256 ns, longer than the 100 ns entry.
        assert_eq!(length.max_seconds(2e9).value(), 256e-9);
        // 512 samples at 8 GHz is 64 ns, shorter than the 100 ns entry.
        assert_eq!(length.max_seconds(8e9).value(), 100e-9);
    }
}

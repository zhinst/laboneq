// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use crate::ir::compilation_job::{DeviceKind, Signal};
use crate::ir::{IrNode, NodeKind, Samples, SectionId, SectionInfo, SignalUid};
use crate::utils::{samples_to_grid, samples_to_length};
use crate::{Error, Result};
use anyhow::Context as AnyhowContext;
use laboneq_units::duration;
use laboneq_units::duration::{Duration, Second};
use laboneq_units::tinysample::{tiny_samples, tinysamples_to_samples};
use std::collections::{HashMap, HashSet};

struct SampleToSecondsConverter {
    sampling_rate: f64,
}

impl SampleToSecondsConverter {
    fn new(sampling_rate: f64) -> Self {
        SampleToSecondsConverter { sampling_rate }
    }

    fn to_seconds(&self, samples: Samples) -> Duration<Second> {
        duration::seconds(samples_to_length(samples, self.sampling_rate))
    }

    fn sampling_rate(&self) -> f64 {
        self.sampling_rate
    }
}

#[derive(Debug, Clone)]
struct PlayInfo {
    start: Samples,
    end: Samples,
    signals: HashSet<SignalUid>,
}

#[derive(Debug, Clone)]
struct AcquireInfo {
    start: Samples,
    end: Samples,
    signals: HashSet<SignalUid>,
}

#[derive(Debug, Clone)]
struct MeasurementInfo<'a> {
    section_info: &'a SectionInfo,
    section_start: Samples,
    play: PlayInfo,
    acquire: AcquireInfo,
}

impl MeasurementInfo<'_> {
    fn start(&self) -> Samples {
        self.section_start + self.play.start.min(self.acquire.start)
    }

    fn end(&self) -> Samples {
        self.section_start + self.play.end.max(self.acquire.end)
    }
}

#[derive(Debug, Clone)]
struct PlayOperation<'a> {
    start: Samples,
    length: Samples,
    signal: &'a Signal,
}

impl PlayOperation<'_> {
    fn end(&self) -> Samples {
        self.start + self.length
    }
}

#[derive(Debug, Clone)]
struct AcquireOperation<'a> {
    start: Samples,
    length: Samples,
    signal: &'a Signal,
}

impl AcquireOperation<'_> {
    fn end(&self) -> Samples {
        self.start + self.length
    }
}

struct SectionMeasurement<'a> {
    section_info: &'a SectionInfo,
    section_start: Samples,
    play_operations: Vec<PlayOperation<'a>>,
    acquire_operations: Vec<AcquireOperation<'a>>,
}

/// Collect measurement information for each section in the IR node.
fn collect_section_measurements<'a>(
    node: &'a IrNode,
    section_info: Option<&'a SectionInfo>,
    section_start: Samples,
    section_map: &mut HashMap<SectionId, SectionMeasurement<'a>>,
    sampling_rate: f64,
) -> Result<()> {
    match node.data() {
        NodeKind::AcquirePulse(ob) => {
            let section_info = section_info.expect("Internal error: Acquire must be in a section");
            let meas_info = section_map.get_mut(&section_info.id).unwrap();
            meas_info.acquire_operations.push(AcquireOperation {
                start: tinysamples_to_samples(tiny_samples(*node.offset()), sampling_rate)
                    + ob.signal.signal_delay,
                length: tinysamples_to_samples(tiny_samples(ob.length), sampling_rate),
                signal: &ob.signal,
            });
            return Ok(());
        }
        NodeKind::PlayPulse(ob) => {
            if ob.pulse_def.is_none() {
                return Ok(());
            }
            let section_info = section_info.expect("Internal error: Play must be in a section");
            let meas_info = section_map.get_mut(&section_info.id).unwrap();
            meas_info.play_operations.push(PlayOperation {
                start: tinysamples_to_samples(tiny_samples(*node.offset()), sampling_rate)
                    + ob.signal.signal_delay,
                length: tinysamples_to_samples(tiny_samples(ob.length), sampling_rate),
                signal: &ob.signal,
            });
            return Ok(());
        }
        _ => {}
    }
    if !node.has_children() {
        return Ok(());
    }
    // If the node is a section, we need to create a new section measurement.
    // Otherwise, we continue with the current section.
    let section_info = node.data().get_section_info().or(section_info);
    let section_start = section_start + node.offset();
    if let Some(section_info) = section_info {
        section_map
            .entry(section_info.id)
            .or_insert_with(|| SectionMeasurement {
                section_info,
                section_start: tinysamples_to_samples(tiny_samples(section_start), sampling_rate),
                play_operations: vec![],
                acquire_operations: vec![],
            });
    }
    // Check only the first iteration.
    // Measurements are equal between the iterations.
    let break_first = matches!(node.data(), NodeKind::Loop(_));
    for child in node.iter_children() {
        collect_section_measurements(
            child,
            section_info,
            section_start,
            section_map,
            sampling_rate,
        )?;
        if break_first {
            break;
        }
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct SignalDelay {
    /// The delay in samples that the signal should be delayed in the AWG by code.
    delay_sequencer: Samples,
    /// The additional delay that the instrument should apply (via node setting).
    delay_port: Duration<Second>,
}

impl SignalDelay {
    fn new(delay_sequencer: Samples, delay_port: f64) -> Self {
        SignalDelay {
            delay_sequencer,
            delay_port: duration::seconds(delay_port),
        }
    }

    pub(crate) fn delay_sequencer(&self) -> Samples {
        self.delay_sequencer
    }

    pub(crate) fn delay_port(&self) -> Duration<Second> {
        self.delay_port
    }
}

fn validate_measurements(
    measurements: &Vec<MeasurementInfo>,
    sample_converter: &SampleToSecondsConverter,
) -> Result<()> {
    if measurements.is_empty() {
        return Ok(());
    }
    let mut top = measurements.first().unwrap();
    for next in measurements.iter().skip(1) {
        // No overlap
        if top.end() <= next.start() {
            top = next;
        } else {
            if top.play.start != next.play.start {
                // Ensure error message is sorted by start time
                let (prev_play, prev, next_play, next) = if top.play.start < next.play.start {
                    (&top.play, top, &next.play, next)
                } else {
                    (&next.play, next, &top.play, top)
                };
                let msg = format!(
                    "Measurements in sections {} and {} \
                    overlap but their play operations start at {} and {}. \
                    The readout pulses of overlapping measurements on the same AWG must start at \
                    the same time.",
                    prev.section_info.name,
                    next.section_info.name,
                    sample_converter.to_seconds(prev_play.start + prev.section_start),
                    sample_converter.to_seconds(next_play.start + next.section_start)
                );
                return Err(Error::new(&msg));
            }
            if top.acquire.start != next.acquire.start {
                // Ensure error message is sorted by start time
                let (prev_acquire, prev, next_acquire, next) =
                    if top.acquire.start < next.acquire.start {
                        (&top.acquire, top, &next.acquire, next)
                    } else {
                        (&next.acquire, next, &top.acquire, top)
                    };
                let msg = format!(
                    "Measurements in sections {} and {} \
                    overlap but their acquire operations start at {} and {}. \
                    The acquire operations of overlapping measurements on the same AWG must start at \
                    the same time.",
                    prev.section_info.name,
                    next.section_info.name,
                    sample_converter.to_seconds(prev_acquire.start + prev.section_start),
                    sample_converter.to_seconds(next_acquire.start + next.section_start)
                );
                return Err(Error::new(&msg));
            }
        }
    }
    Ok(())
}

fn insert_ensure_unique_delays(
    delays: &mut HashMap<SignalUid, SignalDelay>,
    signal: SignalUid,
    delay: SignalDelay,
    sample_converter: &SampleToSecondsConverter,
) -> Result<()> {
    if let Some(existing) = delays.get(&signal) {
        if existing.delay_sequencer != delay.delay_sequencer {
            let mut delays = [existing.delay_sequencer, delay.delay_sequencer];
            delays.sort();
            return Err(Error::new(format!(
                "Cannot resolve measure timing on a signal {} \
                as it would result in two different delays: {} and {}",
                signal.0,
                sample_converter.to_seconds(delays[0]),
                sample_converter.to_seconds(delays[1])
            )));
        } else if existing.delay_port != delay.delay_port {
            let mut delays = [existing.delay_port, delay.delay_port];
            delays.sort();
            return Err(Error::new(format!(
                "Cannot resolve measure timing on a signal {} \
                as it would result in two different delays: {} and {}",
                signal.0, delays[0], delays[1]
            )));
        }
    } else {
        delays.insert(signal, delay);
    }
    Ok(())
}

fn calculate_measurement_delays(
    measurements: Vec<SectionMeasurement<'_>>,
    device: &DeviceKind,
    sample_converter: &SampleToSecondsConverter,
) -> Result<HashMap<SignalUid, SignalDelay>> {
    if measurements.is_empty() {
        return Ok(HashMap::new());
    }
    let measurement_infos = create_measurements(measurements, sample_converter)?;
    let mut delays = HashMap::new();
    for info in measurement_infos {
        let play_info = &info.play;
        let acquire_info = &info.acquire;
        // NOTE: Play sample remainder is handled later by the waveform generator.
        let (play_delay, _) =
            samples_to_grid(play_info.start, device.traits().sample_multiple.into());
        let (acquire_delay, remainder) =
            samples_to_grid(acquire_info.start, device.traits().sample_multiple.into());
        match device {
            DeviceKind::UHFQA => {
                for signal in &acquire_info.signals {
                    let delay = SignalDelay::new(
                        -acquire_delay,
                        samples_to_length(
                            acquire_delay + remainder,
                            sample_converter.sampling_rate(),
                        ),
                    );
                    insert_ensure_unique_delays(&mut delays, *signal, delay, sample_converter)
                        .context(format!(
                            "Failed to adjust measurement timing in signal {}",
                            info.section_info.name
                        ))?;
                }
                for signal in &play_info.signals {
                    let delay = SignalDelay::new(0, 0.0);
                    insert_ensure_unique_delays(&mut delays, *signal, delay, sample_converter)
                        .context(format!(
                            "Failed to adjust measurement timing in signal {}",
                            info.section_info.name
                        ))?;
                }
            }
            DeviceKind::SHFQA => {
                let (delay_play, delay_acquire) = if play_delay > acquire_delay {
                    (acquire_delay - play_delay, 0)
                } else {
                    (0, play_delay - acquire_delay)
                };
                for signal in &acquire_info.signals {
                    let delay = SignalDelay::new(
                        delay_acquire,
                        samples_to_length(
                            -delay_acquire + remainder,
                            sample_converter.sampling_rate(),
                        ),
                    );
                    insert_ensure_unique_delays(&mut delays, *signal, delay, sample_converter)
                        .context(format!(
                            "Failed to adjust measurement timing in signal {}",
                            info.section_info.name
                        ))?;
                }
                for signal in &play_info.signals {
                    let delay = SignalDelay::new(
                        delay_play,
                        samples_to_length(-delay_play, sample_converter.sampling_rate()),
                    );
                    insert_ensure_unique_delays(&mut delays, *signal, delay, sample_converter)
                        .context(format!(
                            "Failed to adjust measurement timing in signal {}",
                            info.section_info.name
                        ))?;
                }
            }
            _ => {}
        }
    }
    Ok(delays)
}

fn build_play_operations<'a>(
    section_info: &'a SectionInfo,
    play_operations: &[PlayOperation<'a>],
    has_acquires: bool,
    sample_converter: &SampleToSecondsConverter,
) -> Result<Option<PlayInfo>> {
    if play_operations.is_empty() || !has_acquires {
        return Ok(None);
    }
    const OPERATION: &str = "play";
    let mut signals: HashSet<SignalUid> = HashSet::new();
    let first_op = play_operations.first().unwrap();
    signals.insert(first_op.signal.uid);
    let first_start = first_op.start;
    let mut total_end = first_op.end();
    for op in play_operations.iter().skip(1) {
        if signals.contains(&op.signal.uid) {
            let msg = format!(
                "There are multiple '{OPERATION}' operations in section '{}' on signal '{}'. \
                A section with acquire signals may only contain a single '{OPERATION}' operation per signal.",
                section_info.name, &op.signal.uid.0
            );
            return Err(Error::new(&msg));
        }
        if first_start != op.start {
            let msg = format!(
                "There are multiple '{OPERATION}' operations in section '{}'. \
                In a section with an acquire, all play signals must start at the same time. \
                Signal '{}' starts at {}. \
                This conflicts with the signals '{}' that start at {}.",
                section_info.name,
                &op.signal.uid.0,
                sample_converter.to_seconds(op.start),
                signals
                    .iter()
                    .map(|s| s.0.to_string())
                    .collect::<Vec<_>>()
                    .join(", "),
                sample_converter.to_seconds(first_start)
            );
            return Err(Error::new(&msg));
        }
        total_end = total_end.max(op.end());
        signals.insert(op.signal.uid);
    }
    let op = PlayInfo {
        start: first_start,
        end: total_end,
        signals,
    };
    Ok(Some(op))
}

fn build_acquire_operations<'a>(
    section_info: &'a SectionInfo,
    operations: &[AcquireOperation<'a>],
    sample_converter: &SampleToSecondsConverter,
) -> Result<Option<AcquireInfo>> {
    if operations.is_empty() {
        return Ok(None);
    }
    const OPERATION: &str = "acquire";
    let mut signals: HashSet<SignalUid> = HashSet::new();
    let first_op = operations.first().unwrap();
    signals.insert(first_op.signal.uid);
    let first_start = first_op.start;
    let mut total_end = first_op.end();
    for op in operations.iter().skip(1) {
        if signals.contains(&op.signal.uid) {
            let msg = format!(
                "There are multiple '{OPERATION}' operations in section '{}' on signal '{}'. \
                A section with acquire signals may only contain a single '{OPERATION}' operation per signal.",
                section_info.name, &op.signal.uid.0
            );
            return Err(Error::new(&msg));
        }
        if first_start != op.start {
            let msg = format!(
                "There are multiple '{OPERATION}' operations in section '{}'. \
                In a section with an acquire, all acquire signals must start at the same time. \
                Signal '{}' starts at {}. \
                This conflicts with the signals '{}' that start at {}.",
                section_info.name,
                &op.signal.uid.0,
                sample_converter.to_seconds(op.start),
                signals
                    .iter()
                    .map(|s| s.0.to_string())
                    .collect::<Vec<_>>()
                    .join(", "),
                sample_converter.to_seconds(first_start)
            );
            return Err(Error::new(&msg));
        }
        total_end = total_end.max(op.end());
        signals.insert(op.signal.uid);
    }
    let op = AcquireInfo {
        start: first_start,
        end: total_end,
        signals,
    };
    Ok(Some(op))
}

fn create_measurements<'a>(
    measurements: Vec<SectionMeasurement<'a>>,
    sample_converter: &SampleToSecondsConverter,
) -> Result<Vec<MeasurementInfo<'a>>> {
    let mut infos = vec![];
    for measurement in measurements.iter() {
        let play_op = build_play_operations(
            measurement.section_info,
            &measurement.play_operations,
            !measurement.acquire_operations.is_empty(),
            sample_converter,
        )?;
        let acquire_op = build_acquire_operations(
            measurement.section_info,
            &measurement.acquire_operations,
            sample_converter,
        )?;
        if play_op.is_none() || acquire_op.is_none() {
            continue;
        }
        if let (Some(play), Some(acquire)) = (play_op, acquire_op) {
            let measurement_info = MeasurementInfo {
                section_info: measurement.section_info,
                section_start: measurement.section_start,
                play,
                acquire,
            };
            infos.push(measurement_info);
        };
    }
    validate_measurements(&infos, sample_converter)?;
    Ok(infos)
}

pub(crate) struct IntegrationLength {
    signal: SignalUid,
    duration: Samples,
    is_play: bool,
}

impl IntegrationLength {
    pub(crate) fn signal(&self) -> &SignalUid {
        &self.signal
    }

    pub(crate) fn duration(&self) -> Samples {
        self.duration
    }

    pub(crate) fn is_play(&self) -> bool {
        self.is_play
    }
}

fn calculate_integration_times(
    measurements: &Vec<SectionMeasurement<'_>>,
    sample_converter: &SampleToSecondsConverter,
) -> Result<Vec<IntegrationLength>> {
    let mut integration_lengths: HashMap<SignalUid, IntegrationLength> = HashMap::new();
    for measurement in measurements.iter() {
        if measurement.acquire_operations.is_empty() {
            continue;
        }
        for play_op in measurement.play_operations.iter() {
            let play_length = play_op.end() - play_op.start;
            if let Some(previous) = integration_lengths.get(&play_op.signal.uid)
                && previous.duration != play_length
            {
                let msg = format!(
                    "Signal '{}' has two different integration lengths: \
                        '{}' from section '{}' and '{}' from earlier section.",
                    &play_op.signal.uid.0,
                    play_length,
                    measurement.section_info.name,
                    previous.duration
                );
                return Err(Error::new(&msg));
            }
            let integration = IntegrationLength {
                signal: play_op.signal.uid,
                duration: play_length,
                is_play: true,
            };
            integration_lengths.insert(play_op.signal.uid, integration);
        }
        for acquire_op in measurement.acquire_operations.iter() {
            let length = acquire_op.end() - acquire_op.start;
            if let Some(previous) = integration_lengths.get(&acquire_op.signal.uid)
                && previous.duration != length
            {
                let msg = format!(
                    "Signal '{}' has two different integration lengths: \
                        '{}' from section '{}' and '{}' from earlier section.",
                    &acquire_op.signal.uid.0,
                    sample_converter.to_seconds(length),
                    measurement.section_info.name,
                    sample_converter.to_seconds(previous.duration),
                );
                return Err(Error::new(&msg));
            }
            let integration = IntegrationLength {
                signal: acquire_op.signal.uid,
                duration: length,
                is_play: false,
            };
            integration_lengths.insert(acquire_op.signal.uid, integration);
        }
    }
    Ok(integration_lengths.into_values().collect())
}

#[derive(Default)]
pub(crate) struct MeasurementAnalysis {
    /// A map of signal names to their respective delays, which include both the sequencer
    /// delay and the port delay
    pub delays: HashMap<SignalUid, SignalDelay>,
    /// A list of integration lengths for signals, which includes both play and acquire signals.
    pub integration_lengths: Vec<IntegrationLength>,
}

/// Analyze the measurements in the given IR node.
///
/// This function evaluates measurement operations in the IR node and calculates the
/// integration times and delays for each signal involved in the measurements.
///
/// # Returns
///
/// Analysis result of the measurements.
pub(crate) fn analyze_measurements(
    node: &IrNode,
    device: &DeviceKind,
    sampling_rate: f64,
) -> Result<MeasurementAnalysis> {
    if !matches!(device, DeviceKind::UHFQA | DeviceKind::SHFQA) {
        return Ok(MeasurementAnalysis::default());
    }
    let mut measurements: HashMap<u32, SectionMeasurement<'_>> = HashMap::new();
    collect_section_measurements(node, None, 0, &mut measurements, sampling_rate)?;
    let mut measurements: Vec<SectionMeasurement<'_>> = measurements.into_values().collect();
    measurements.sort_by(|a, b| a.section_start.cmp(&b.section_start));
    let sample_converter = SampleToSecondsConverter::new(sampling_rate);
    let integration_times = calculate_integration_times(&measurements, &sample_converter)?;
    let signal_delays = calculate_measurement_delays(measurements, device, &sample_converter)?;
    Ok(MeasurementAnalysis {
        delays: signal_delays,
        integration_lengths: integration_times,
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::ir::compilation_job::{DeviceKind, PulseDef, PulseDefKind, Signal, SignalKind};
    use crate::ir::{AcquirePulse, IrNode, NodeKind, PlayPulse, Section, SectionInfo};
    use laboneq_units::tinysample::samples_to_tinysamples;

    struct IrBuilder {
        node_stack: Vec<IrNode>,
        section_id_counter: HashMap<String, u32>,
    }

    impl IrBuilder {
        fn new() -> Self {
            Self {
                node_stack: vec![IrNode::new(NodeKind::Nop { length: 0 }, 0)],
                section_id_counter: HashMap::new(),
            }
        }

        pub(crate) fn with<F>(&mut self, f: F)
        where
            F: FnOnce(&mut Self),
        {
            f(self);
        }

        fn enter_stack<F>(&mut self, node: IrNode, f: F)
        where
            F: FnOnce(&mut Self),
        {
            self.node_stack.push(node);
            f(self);
            let parent = self.node_stack.pop();
            if let Some(parent) = parent {
                self.node_stack.last_mut().unwrap().add_child_node(parent);
            }
        }

        pub(crate) fn section<F>(&mut self, uid: &str, length: Samples, offset: Samples, f: F)
        where
            F: FnOnce(&mut Self),
        {
            let section = NodeKind::Section(Section {
                length,
                trigger_output: vec![],
                prng_setup: None,
                section_info: Arc::new(SectionInfo {
                    name: uid.to_string(),
                    id: {
                        let next_id = self.section_id_counter.len() as u32;
                        *self
                            .section_id_counter
                            .entry(uid.to_string())
                            .or_insert(next_id)
                    },
                }),
            });
            self.enter_stack(IrNode::new(section, offset), f);
        }

        pub(crate) fn play(&mut self, offset: Samples, signal: &Signal, length: Samples) {
            let play = NodeKind::PlayPulse(PlayPulse {
                signal: Arc::new(signal.clone()),
                set_oscillator_phase: None,
                increment_oscillator_phase: None,
                length,
                amp_param_name: None,
                id_pulse_params: None,
                amplitude: None,
                incr_phase_param_name: None,
                markers: vec![],
                pulse_def: Arc::new(PulseDef::test("".to_string(), PulseDefKind::Pulse)).into(),
                phase: 0.0,
            });
            let ir_node = IrNode::new(play, offset);
            self.node_stack.last_mut().unwrap().add_child_node(ir_node);
        }

        pub(crate) fn acquire(&mut self, offset: Samples, signal: &Signal, length: Samples) {
            let play = NodeKind::AcquirePulse(AcquirePulse {
                signal: Arc::new(signal.clone()),
                length,
                pulse_defs: vec![Arc::new(PulseDef::test(
                    "".to_string(),
                    PulseDefKind::Pulse,
                ))],
                id_pulse_params: vec![None],
                handle: "".into(),
            });
            let ir_node = IrNode::new(play, offset);
            self.node_stack.last_mut().unwrap().add_child_node(ir_node);
        }

        fn build(&mut self) -> IrNode {
            self.node_stack.pop().unwrap()
        }
    }

    fn create_signal(uid: u32, delay: Samples) -> Signal {
        Signal {
            uid: uid.into(),
            kind: SignalKind::IQ,
            signal_delay: delay,
            start_delay: 0,
            channels: vec![],
            oscillator: None,
            automute: false,
        }
    }

    /// Test that the acquire and measure signal delays are calculated to start at the same time.
    #[test]
    fn test_acquire_offset() {
        let device = DeviceKind::SHFQA;
        let srate = device.traits().sampling_rate;

        let signals = HashMap::from([
            ("acquire".to_string(), (create_signal(0, 0))),
            ("measure".to_string(), (create_signal(1, 0))),
        ]);

        let mut builder = IrBuilder::new();
        builder.with(|b| {
            b.section("s0", 0, 0, |b| {
                b.play(
                    samples_to_tinysamples(0, srate).value(),
                    signals.get("measure").unwrap(),
                    samples_to_tinysamples(320, srate).value(),
                );
                b.acquire(
                    samples_to_tinysamples(16, srate).value(),
                    signals.get("acquire").unwrap(),
                    samples_to_tinysamples(320, srate).value(),
                );
            });
        });

        let result = analyze_measurements(&builder.build(), &DeviceKind::SHFQA, srate)
            .unwrap()
            .delays;

        // Test that measure and acquire are adjusted to start at the same time for sequencer
        let measure = result.get(&1.into()).unwrap();
        let acquire = result.get(&0.into()).unwrap();
        assert_eq!(measure.delay_sequencer(), 0);
        assert_eq!(acquire.delay_sequencer(), -16);
        assert_eq!(Into::<f64>::into(measure.delay_port()), 0.0);
        assert_eq!(Into::<f64>::into(measure.delay_port()), 0.0);
    }

    /// Test that when delays cannot be fully adjusted for the sequencer,
    /// the delays are compensated with the port delay.
    #[test]
    fn test_port_delay() {
        let device = DeviceKind::SHFQA;
        let srate = device.traits().sampling_rate;

        // Offset + signal delay that does no match the granularity of the sequencer,
        // So it cannot be fully compensated by the sequencer.
        let offset: Samples = device.traits().sample_multiple.into();
        let signal_delay = 24;

        let signals = HashMap::from([
            ("acquire".to_string(), (create_signal(0, signal_delay))),
            ("measure".to_string(), (create_signal(1, 0))),
        ]);

        let mut builder = IrBuilder::new();
        builder.with(|b| {
            b.section("s0", 0, 0, |b| {
                b.play(
                    samples_to_tinysamples(0, srate).value(),
                    signals.get("measure").unwrap(),
                    samples_to_tinysamples(320, srate).value(),
                );
                b.acquire(
                    samples_to_tinysamples(offset, srate).value(),
                    signals.get("acquire").unwrap(),
                    samples_to_tinysamples(320, srate).value(),
                );
            });
        });

        let result = analyze_measurements(&builder.build(), &DeviceKind::SHFQA, srate)
            .unwrap()
            .delays;
        let measure = result.get(&1.into()).unwrap();
        let acquire = result.get(&0.into()).unwrap();
        assert_eq!(measure.delay_sequencer(), 0);
        assert_eq!(Into::<f64>::into(measure.delay_port()), 0.0);

        assert_eq!(acquire.delay_sequencer(), -32);
        assert_eq!(
            Into::<f64>::into(acquire.delay_port()),
            samples_to_length(offset, srate) + samples_to_length(signal_delay, srate)
        );
    }

    /// Test multiple sections delay compensation.
    #[test]
    fn test_delay_subtraction_rounding() {
        let device = DeviceKind::SHFQA;
        let srate = device.traits().sampling_rate;

        let signals = HashMap::from([
            ("acquire".to_string(), (create_signal(0, 0))),
            ("measure".to_string(), (create_signal(1, 0))),
        ]);
        let s0_offset = 0;
        let s0_offset_ts = samples_to_tinysamples(s0_offset, srate);
        let s0_play_offset = samples_to_tinysamples(212, srate);
        let s0_play_length = samples_to_tinysamples(600, srate);
        let s0_acquire_offset = samples_to_tinysamples(240, srate);
        let s0_acquire_length = samples_to_tinysamples(2000, srate);

        let s1_offset = 240;
        let s1_offset_ts = samples_to_tinysamples(s1_offset, srate);
        let s1_play_offset = samples_to_tinysamples(212, srate);
        let s1_play_length = samples_to_tinysamples(600, srate);
        let s1_acquire_offset = samples_to_tinysamples(240, srate);
        let s1_acquire_length = samples_to_tinysamples(2000, srate);

        let mut builder = IrBuilder::new();
        builder.with(|b| {
            b.section("s0", 0, s0_offset_ts.value(), |b| {
                b.play(
                    s0_play_offset.value(),
                    signals.get("measure").unwrap(),
                    s0_play_length.value(),
                );
                b.acquire(
                    s0_acquire_offset.value(),
                    signals.get("acquire").unwrap(),
                    s0_acquire_length.value(),
                );
            });
            b.section("s1", 0, s1_offset_ts.value(), |b| {
                b.play(
                    s1_play_offset.value(),
                    signals.get("measure").unwrap(),
                    s1_play_length.value(),
                );
                b.acquire(
                    s1_acquire_offset.value(),
                    signals.get("acquire").unwrap(),
                    s1_acquire_length.value(),
                );
            });
        });
        let result = analyze_measurements(&builder.build(), &DeviceKind::SHFQA, srate)
            .unwrap()
            .delays;

        let measure = result.get(&1.into()).unwrap();
        let acquire = result.get(&0.into()).unwrap();

        let expected_delay = 32;
        assert_eq!(measure.delay_sequencer(), 0);
        assert_eq!(Into::<f64>::into(measure.delay_port()), 0.0);
        assert_eq!(acquire.delay_sequencer(), -expected_delay);
        assert_eq!(
            Into::<f64>::into(acquire.delay_port()),
            samples_to_length(expected_delay, srate)
        );
    }
}

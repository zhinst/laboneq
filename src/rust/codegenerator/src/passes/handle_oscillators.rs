// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::cmp::max;
use std::collections::BTreeMap;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;
use std::sync::Arc;

use crate::ir::compilation_job::{self as cjob, AwgCore, DeviceKind};
use crate::ir::{
    IrNode, LinearParameterInfo, NodeKind, OscillatorFrequencySweepStep, Samples,
    SetOscillatorFrequencySweep,
};
use crate::tinysample::TINYSAMPLE;
use crate::{Error, Result};

struct SignalPhaseTracker {
    cumulative: f64,
    // Reference time of last phase set time
    reference_time: Samples,
}

struct PhaseTracker {
    trackers: HashMap<String, SignalPhaseTracker>,
    global_reset_time: Samples,
}

impl PhaseTracker {
    fn new(signals: &[Rc<cjob::Signal>]) -> Self {
        let mut trackers = HashMap::new();
        for sig in signals.iter() {
            trackers.insert(
                sig.uid.clone(),
                SignalPhaseTracker {
                    cumulative: 0.0,
                    reference_time: 0,
                },
            );
        }
        PhaseTracker {
            trackers,
            global_reset_time: 0,
        }
    }

    fn set(&mut self, signal: &cjob::Signal, ts: Samples, value: f64) {
        let tracker = self.trackers.get_mut(&signal.uid).expect("Unknown signal");
        tracker.cumulative = value;
        tracker.reference_time = ts;
    }

    fn increment(&mut self, signal: &cjob::Signal, value: f64) {
        let tracker = self.trackers.get_mut(&signal.uid).expect("Unknown signal");
        tracker.cumulative += value;
    }

    fn global_reset(&mut self, ts: Samples) {
        for tracker in self.trackers.values_mut() {
            tracker.cumulative = 0.0;
        }
        self.global_reset_time = ts;
    }

    pub fn phase_now(&self, signal: &cjob::Signal) -> (Samples, f64) {
        let tracker = self.trackers.get(&signal.uid).expect("Unknown signal");
        let time_ref = max(tracker.reference_time, self.global_reset_time);
        let phase = tracker.cumulative;
        (time_ref, phase)
    }

    pub fn calculate_phase_at(&self, signal: &cjob::Signal, freq: f64, ts: Samples) -> f64 {
        let (ref_time, phase_now) = self.phase_now(signal);
        let t = (ts as f64 - ref_time as f64) * TINYSAMPLE;
        t * 2.0 * std::f64::consts::PI * freq + phase_now
    }
}

pub struct SoftwareOscillatorParameters {
    active_osc_freq: HashMap<String, Vec<f64>>,
    pulse_osc_freq: HashMap<(String, Samples), f64>,
    sw_osc_phases: HashMap<(String, Samples), f64>,
}

impl SoftwareOscillatorParameters {
    fn set_osc_freq(&mut self, signal: &cjob::Signal, value: f64) {
        self.active_osc_freq
            .entry(signal.uid.to_owned())
            .or_default()
            .push(value);
    }

    fn timestamp_osc_freq(&mut self, signal: &cjob::Signal, time: Samples) {
        let freq = self
            .active_osc_freq
            .get(&signal.uid)
            .map_or(&0.0, |x| x.last().unwrap_or(&0.0));
        self.pulse_osc_freq
            .insert((signal.uid.to_owned(), time), *freq);
    }

    /// Frequency for selected signal at given timestamp
    pub fn freq_at(&self, signal: &cjob::Signal, ts: Samples) -> Option<f64> {
        self.pulse_osc_freq
            .get(&(signal.uid.to_owned(), ts))
            .copied()
    }

    /// Phase for selected signal at given timestamp
    pub fn phase_at(&self, signal: &cjob::Signal, ts: Samples) -> Option<f64> {
        self.sw_osc_phases
            .get(&(signal.uid.to_owned(), ts))
            .copied()
    }
}

fn collect_osc_parameters(
    node: &mut IrNode,
    state: &mut SoftwareOscillatorParameters,
    phase_tracker: &mut Option<PhaseTracker>,
    func: &impl Fn(&str, Samples) -> Samples,
    in_branch: bool,
    device: &DeviceKind,
) -> Result<()> {
    let node_offset = *node.offset();
    match node.data_mut() {
        NodeKind::InitialOscillatorFrequency(x) => {
            for signal_freq in x.iter() {
                if signal_freq.signal.is_sw_modulated() {
                    state.set_osc_freq(&signal_freq.signal, signal_freq.frequency);
                }
            }
            Ok(())
        }
        NodeKind::SetOscillatorFrequency(x) => {
            for signal_freq in x.iter() {
                if signal_freq.signal.is_sw_modulated() {
                    state.set_osc_freq(&signal_freq.signal, signal_freq.frequency);
                }
            }
            Ok(())
        }
        NodeKind::PlayPulse(ob) => {
            if ob.set_oscillator_phase.is_some() {
                if ob.signal.is_hw_modulated() {
                    let msg = format!(
                        "Cannot use 'set_oscillator_phase' of hardware modulated signal: {}",
                        &ob.signal.uid
                    );
                    return Err(Error::new(&msg));
                }
                if in_branch && ob.signal.is_sw_modulated() {
                    let msg = format!(
                        "Conditional 'set_oscillator_phase' of software modulated signal '{}' is not supported",
                        &ob.signal.uid
                    );
                    return Err(Error::new(&msg));
                }
            }
            if ob.increment_oscillator_phase.is_some() && in_branch {
                // SHFQA and UHFQA do not support phase registers.
                if device.is_qa_device() {
                    let msg = format!(
                        "Conditional 'increment_oscillator_phase' of signal '{}' is not supported on device type '{}'",
                        &ob.signal.uid,
                        device.as_str()
                    );
                    return Err(Error::new(&msg));
                }
                if ob.signal.is_sw_modulated() {
                    let msg = format!(
                        "Conditional 'increment_oscillator_phase' of software modulated signal '{}' is not supported",
                        &ob.signal.uid
                    );
                    return Err(Error::new(&msg));
                }
            }
            if ob.signal.is_hw_modulated() {
                return Ok(());
            }
            // TODO: More elegant way to map to the node than a timestamp
            let offset = func(&ob.signal.uid, node_offset);
            state.timestamp_osc_freq(&ob.signal, offset);
            if let Some(tracker) = phase_tracker {
                // Set oscillator phase priority over incrementing
                if let Some(set_oscillator_phase) = ob.set_oscillator_phase {
                    tracker.set(&ob.signal, node_offset, set_oscillator_phase);
                    // PlayIR nodes that have `set_oscillator_phase` and no `pulse_def` can be pruned.
                    // Perhaps it this could be a separate node altogether?
                    if ob.pulse_def.is_none() {
                        node.replace_data(NodeKind::Nop {
                            length: node.data().length(),
                        });
                        return Ok(());
                    }
                } else if let Some(increment_oscillator_phase) = ob.increment_oscillator_phase {
                    tracker.increment(&ob.signal, increment_oscillator_phase);
                }
                let phase = tracker.calculate_phase_at(
                    &ob.signal,
                    state.freq_at(&ob.signal, offset).unwrap_or(0.0),
                    node_offset,
                );
                state
                    .sw_osc_phases
                    .insert((ob.signal.uid.clone(), offset), phase);
            } else {
                state
                    .sw_osc_phases
                    .insert((ob.signal.uid.clone(), offset), 0.0);
            }
            Ok(())
        }
        NodeKind::AcquirePulse(ob) => {
            // TODO: More elegant way to map to the node than a timestamp
            let offset = func(&ob.signal.uid, node_offset);
            state.timestamp_osc_freq(&ob.signal, offset);
            Ok(())
        }
        NodeKind::PhaseReset(ob) => {
            if ob.reset_sw_oscillators {
                if let Some(tracker) = phase_tracker {
                    tracker.global_reset(node_offset);
                }
            }
            Ok(())
        }
        NodeKind::Case(_) => {
            for child in node.iter_children_mut() {
                collect_osc_parameters(child, state, phase_tracker, func, true, device)?;
            }
            Ok(())
        }
        _ => {
            for child in node.iter_children_mut() {
                collect_osc_parameters(child, state, phase_tracker, func, in_branch, device)?;
            }
            Ok(())
        }
    }
}

/// Pass to handle  and calculate software oscillator parameters.
///
/// Consumes [`NodeKind::PlayPulse`] which the are used only for phase increments.
/// Tiny sample is used to avoid potential rounding errors when calculating phase increments.
///
/// # Returns
///
/// Calculated oscillator frequency and phase values for each signal and pulse at given time in target device unit samples,
/// which is calculated by the `timestamp_shifting_function`.
pub fn handle_oscillator_parameters(
    node: &mut IrNode,
    signals: &[Rc<cjob::Signal>],
    device_kind: &DeviceKind,
    timestamp_shifting_function: impl Fn(&str, Samples) -> Samples,
) -> Result<SoftwareOscillatorParameters> {
    let mut state = SoftwareOscillatorParameters {
        active_osc_freq: HashMap::new(),
        pulse_osc_freq: HashMap::new(),
        sw_osc_phases: HashMap::new(),
    };
    // Phase on QA device is always 0
    let mut phase_tracker = match device_kind.traits().is_qa_device {
        true => None,
        false => Some(PhaseTracker::new(signals)),
    };
    collect_osc_parameters(
        node,
        &mut state,
        &mut phase_tracker,
        &timestamp_shifting_function,
        false,
        device_kind,
    )?;
    Ok(state)
}

struct OscillatorFrequencySweepInfo {
    start_frequency: f64,
    current_frequency: f64,
    step_frequency: f64,
    n_iterations: usize,
}

struct OscillatorIteration {
    osc_index: u16,
    iteration: usize,
}

struct SweepInfo {
    osc_sweep_info: HashMap<u16, OscillatorFrequencySweepInfo>,
    node_id_to_osc_index: HashMap<usize, Vec<OscillatorIteration>>,
}

fn collect_set_oscillator_frequency_nodes<'a>(node: &'a mut IrNode, out: &mut Vec<&'a mut IrNode>) {
    for child in node.iter_children_mut() {
        if let NodeKind::SetOscillatorFrequency(_) = child.data() {
            out.push(child);
        } else {
            collect_set_oscillator_frequency_nodes(child, out);
        }
    }
}

fn evaluate_sweep_properties(
    nodes: &Vec<&mut IrNode>,
    osc_allocation: &HashMap<&str, u16>,
) -> Result<SweepInfo> {
    let mut node_id_to_osc_index: HashMap<usize, Vec<OscillatorIteration>> = HashMap::new();
    let mut osc_sweep_info = HashMap::new();

    for (node_id, node) in nodes.iter().enumerate() {
        if let NodeKind::SetOscillatorFrequency(ob) = node.data() {
            let mut osc_frequencies = BTreeMap::new();
            for signal_freq in ob.iter() {
                if let Some(osc_index) = osc_allocation.get(signal_freq.signal.uid.as_str()) {
                    osc_frequencies
                        .insert(*osc_index, (signal_freq.frequency, &signal_freq.signal));
                }
            }
            if osc_frequencies.is_empty() {
                // No hardware oscillators, skip
                continue;
            }
            let iteration = ob.iteration();
            for (osc_index, (frequency, signal)) in osc_frequencies {
                // Keep track of the oscillator index and iteration number for this node
                node_id_to_osc_index
                    .entry(node_id)
                    .or_default()
                    .push(OscillatorIteration {
                        osc_index,
                        iteration,
                    });
                if iteration == 0 {
                    // First iteration, initialize the oscillator frequency info, overwriting any previous values
                    // in case of an nested loop
                    osc_sweep_info.insert(
                        osc_index,
                        OscillatorFrequencySweepInfo {
                            start_frequency: frequency,
                            current_frequency: frequency,
                            step_frequency: 0.0,
                            n_iterations: 1,
                        },
                    );
                    continue;
                }
                let sweep_info = osc_sweep_info.get_mut(&osc_index).expect(
                    "Internal error: Hardware oscillator frequencies not ordered by iteration number",
                );
                if iteration == 1 {
                    // Second iteration, set the step frequency
                    sweep_info.step_frequency = frequency - sweep_info.start_frequency;
                    sweep_info.current_frequency = frequency;
                } else {
                    // For subsequent iterations, check if the step frequency is linear
                    const STEP_ABS_TOLERANCE: f64 = 1e-3; // 1 mHz tolerance for hardware oscillator frequency sweeps
                    let next_step = frequency
                        - iteration as f64 * sweep_info.step_frequency
                        - sweep_info.start_frequency;
                    if next_step.abs() > STEP_ABS_TOLERANCE {
                        return Err(Error::new(
                            format!(
                                "Invalid oscillator frequency step on signal '{}': {}",
                                signal.uid, "Realtime oscillator frequency sweep must be linear."
                            )
                            .as_str(),
                        ));
                    }
                    sweep_info.current_frequency = frequency;
                }
                sweep_info.n_iterations += 1;
            }
        }
    }
    let out = SweepInfo {
        osc_sweep_info,
        node_id_to_osc_index,
    };
    Ok(out)
}

/// Replace the gathered oscillator sweep step nodes with [`SetOscillatorFrequencySweep`] nodes.
fn replace_hw_oscillator_sweep_nodes(
    mut nodes: Vec<&mut IrNode>,
    sweep_info: &SweepInfo,
    cut_points: &mut HashSet<Samples>,
) {
    let osc_sweep_infos = sweep_info
        .osc_sweep_info
        .iter()
        .map(|(osc_index, info)| {
            (
                *osc_index,
                Arc::new(LinearParameterInfo {
                    start: info.start_frequency,
                    step: info.step_frequency,
                    count: info.n_iterations,
                }),
            )
        })
        .collect::<HashMap<_, _>>();
    if osc_sweep_infos.is_empty() {
        // No hardware sweeps
        return;
    }
    for (node_id, node) in nodes.iter_mut().enumerate() {
        let node_iteration_info = sweep_info
            .node_id_to_osc_index
            .get(&node_id)
            .expect("Internal error: Expected node ID to be in osc_calculator");
        let mut osc_setups = vec![];
        for node_iteration in node_iteration_info {
            let osc_index = node_iteration.osc_index;
            let info = osc_sweep_infos
                .get(&osc_index)
                .expect("Internal error: Expected oscillator index to exists.");
            osc_setups.push(OscillatorFrequencySweepStep {
                iteration: node_iteration.iteration,
                osc_index,
                parameter: Arc::clone(info),
            });
        }
        let node_info = NodeKind::SetOscillatorFrequencySweep(SetOscillatorFrequencySweep {
            length: node.data().length(),
            oscillators: osc_setups,
        });
        node.replace_data(node_info);
        cut_points.insert(*node.offset());
    }
}

fn check_device_compatibility(device: &DeviceKind, sweep_info: &SweepInfo) -> Result<()> {
    const MAX_SWEEP_ITERATIONS_HDAWG: usize = 512; // HDAWG can only handle up to 512 iterations in a frequency sweep
    if !matches!(
        device,
        DeviceKind::SHFSG | DeviceKind::SHFQA | DeviceKind::HDAWG
    ) {
        return Err(Error::new(
            "Real-time frequency sweep only supported on SHF and HDAWG devices",
        ));
    }
    if device == &DeviceKind::HDAWG
        && sweep_info
            .osc_sweep_info
            .values()
            .any(|info| info.n_iterations > MAX_SWEEP_ITERATIONS_HDAWG)
    {
        return Err(Error::new(&format!(
            "HDAWG can only handle RT frequency sweeps up to {MAX_SWEEP_ITERATIONS_HDAWG} steps."
        )));
    }
    Ok(())
}

/// Transformation pass to handle realtime oscillator frequency sweeps in the IR tree.
///
/// This function searches for [`SetOscillatorFrequency`] nodes in the IR tree,
/// collects their frequency sweep information, and replaces them with [`SetOscillatorFrequencySweep`] nodes.
///
/// The pass does the following:
///
/// 1. Calculates and evaluates the oscillator realtime frequency sweep parameters.
/// 2. Checks if the device is compatible with realtime frequency sweeps.
/// 3. Replaces the collected nodes with a [`SetOscillatorFrequencySweep`] node
///    that contains the frequency sweep information for each oscillator.
/// 4. Applies a global delay to the offset of the nodes and updates cut points accordingly.
pub fn handle_oscillator_sweeps(
    node: &mut IrNode,
    awg: &AwgCore,
    cut_points: &mut HashSet<Samples>,
) -> Result<()> {
    let signal_osc_index_allocation = &awg.oscillator_index_by_signal_uid();
    if signal_osc_index_allocation.is_empty() {
        return Ok(());
    }
    let mut nodes = vec![];
    collect_set_oscillator_frequency_nodes(node, &mut nodes);
    if nodes.is_empty() {
        return Ok(());
    }
    let sweep_info = evaluate_sweep_properties(&nodes, signal_osc_index_allocation)?;
    check_device_compatibility(awg.device_kind(), &sweep_info)?;
    replace_hw_oscillator_sweep_nodes(nodes, &sweep_info, cut_points);
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::ir::{PhaseReset, PlayPulse};

    use super::*;

    fn make_signal(uid: &str, kind: cjob::OscillatorKind) -> Rc<cjob::Signal> {
        let sig = cjob::Signal {
            uid: uid.to_string(),
            kind: cjob::SignalKind::IQ,
            channels: vec![],
            oscillator: Some(cjob::Oscillator {
                uid: "osc".to_string(),
                kind,
            }),
            signal_delay: 0,
            start_delay: 0,
            mixer_type: None,
        };
        Rc::new(sig)
    }

    fn make_reset(reset_sw_oscillators: bool) -> NodeKind {
        NodeKind::PhaseReset(PhaseReset {
            reset_sw_oscillators,
            signals: vec![],
        })
    }

    fn make_pulse(
        signal: Rc<cjob::Signal>,
        set_oscillator_phase: Option<f64>,
        increment_oscillator_phase: Option<f64>,
    ) -> NodeKind {
        NodeKind::PlayPulse(PlayPulse {
            signal,
            set_oscillator_phase,
            increment_oscillator_phase,
            length: 32,
            amp_param_name: None,
            id_pulse_params: None,
            amplitude: None,
            incr_phase_param_name: None,
            markers: vec![],
            pulse_def: Some(Arc::new(cjob::PulseDef::test(
                "param".to_string(),
                cjob::PulseDefKind::Pulse,
            ))),
            phase: 0.0,
        })
    }

    #[test]
    fn test_phase_increment() {
        let signal = make_signal("test", cjob::OscillatorKind::SOFTWARE);
        let mut root = IrNode::new(NodeKind::Nop { length: 0 }, 0);
        root.add_child(0, make_reset(true));
        root.add_child(1, make_pulse(Rc::clone(&signal), None, Some(0.5)));
        root.add_child(3, make_pulse(Rc::clone(&signal), None, Some(0.5)));
        let mut nested_section = IrNode::new(NodeKind::Nop { length: 0 }, 5);
        nested_section.add_child(5, make_reset(true));
        nested_section.add_child(7, make_pulse(Rc::clone(&signal), None, Some(0.5)));
        root.add_child_node(nested_section);
        root.add_child(11, make_pulse(Rc::clone(&signal), None, Some(0.5)));

        let params = handle_oscillator_parameters(
            &mut root,
            &[Rc::clone(&signal)],
            &DeviceKind::SHFSG,
            |_, ts| ts,
        )
        .unwrap();
        assert_eq!(params.phase_at(&signal, 1).unwrap(), 0.5);
        assert_eq!(params.phase_at(&signal, 3).unwrap(), 1.0);
        assert_eq!(params.phase_at(&signal, 7).unwrap(), 0.5);
        assert_eq!(params.phase_at(&signal, 11).unwrap(), 1.0);
    }

    #[test]
    fn test_phase_increment_no_sw_reset() {
        let signal = make_signal("test", cjob::OscillatorKind::SOFTWARE);
        let mut root = IrNode::new(NodeKind::Nop { length: 0 }, 0);
        root.add_child(0, make_pulse(Rc::clone(&signal), None, Some(0.5)));
        root.add_child(1, make_reset(false));
        root.add_child(3, make_pulse(Rc::clone(&signal), None, Some(0.5)));

        let params = handle_oscillator_parameters(
            &mut root,
            &[Rc::clone(&signal)],
            &DeviceKind::SHFSG,
            |_, ts| ts,
        )
        .unwrap();
        assert_eq!(params.phase_at(&signal, 0).unwrap(), 0.5);
        assert_eq!(params.phase_at(&signal, 3).unwrap(), 1.0);
    }

    #[test]
    fn test_phase_increment_hw_osc() {
        let signal = make_signal("test", cjob::OscillatorKind::HARDWARE);
        let mut root = IrNode::new(NodeKind::Nop { length: 0 }, 0);
        root.add_child(0, make_pulse(Rc::clone(&signal), None, Some(0.5)));
        root.add_child(1, make_pulse(Rc::clone(&signal), None, Some(0.5)));

        let params = handle_oscillator_parameters(
            &mut root,
            &[Rc::clone(&signal)],
            &DeviceKind::SHFSG,
            |_, ts| ts,
        )
        .unwrap();
        assert!(params.phase_at(&signal, 0).is_none());
        assert!(params.phase_at(&signal, 1).is_none());
    }

    #[test]
    fn test_set_phase() {
        let signal = make_signal("test", cjob::OscillatorKind::SOFTWARE);
        let mut root = IrNode::new(NodeKind::Nop { length: 0 }, 0);
        root.add_child(0, make_pulse(Rc::clone(&signal), Some(1.0), Some(0.5)));
        root.add_child(1, make_pulse(Rc::clone(&signal), None, Some(0.5)));
        root.add_child(3, make_reset(true));
        root.add_child(5, make_pulse(Rc::clone(&signal), Some(1.0), None));

        let params = handle_oscillator_parameters(
            &mut root,
            &[Rc::clone(&signal)],
            &DeviceKind::SHFSG,
            |_, ts| ts,
        )
        .unwrap();
        // Set phase wins over phase increments
        assert_eq!(params.phase_at(&signal, 0).unwrap(), 1.0);
        assert_eq!(params.phase_at(&signal, 1).unwrap(), 1.5);
        assert_eq!(params.phase_at(&signal, 5).unwrap(), 1.0);
    }

    #[test]
    fn test_set_phase_no_osc() {
        let sig = cjob::Signal {
            uid: "test".to_string(),
            kind: cjob::SignalKind::IQ,
            channels: vec![],
            oscillator: None,
            signal_delay: 0,
            start_delay: 0,
            mixer_type: None,
        };
        let signal = Rc::new(sig);
        let mut root = IrNode::new(NodeKind::Nop { length: 0 }, 0);
        root.add_child(0, make_pulse(Rc::clone(&signal), Some(1.0), Some(0.5)));
        root.add_child(1, make_pulse(Rc::clone(&signal), None, Some(0.5)));
        root.add_child(3, make_reset(true));
        root.add_child(5, make_pulse(Rc::clone(&signal), Some(1.0), None));

        let params = handle_oscillator_parameters(
            &mut root,
            &[Rc::clone(&signal)],
            &DeviceKind::SHFSG,
            |_, ts| ts,
        )
        .unwrap();
        // Set phase wins over phase increments
        assert_eq!(params.phase_at(&signal, 0).unwrap(), 1.0);
        assert_eq!(params.phase_at(&signal, 1).unwrap(), 1.5);
        assert_eq!(params.phase_at(&signal, 5).unwrap(), 1.0);
    }

    mod handle_oscillator_sweeps {
        use super::*;
        use crate::ir::compilation_job::{
            AwgCore, Device, DeviceKind, Oscillator, OscillatorKind, Signal, SignalKind,
        };
        use crate::ir::{
            LinearParameterInfo, NodeKind, OscillatorFrequencySweepStep, SetOscillatorFrequency,
            SignalFrequency,
        };
        use std::rc::Rc;
        use std::sync::Arc;

        fn assert_set_oscillator_sweep(
            node_kind: &NodeKind,
            expected: &SetOscillatorFrequencySweep,
        ) {
            if let NodeKind::SetOscillatorFrequencySweep(node_kind) = node_kind {
                assert_eq!(node_kind, expected);
            } else {
                panic!("Expected SetOscillatorFrequencySweep node");
            }
        }

        fn make_set_oscillator_frequency(values: Vec<(String, f64)>, iteration: usize) -> NodeKind {
            let values = values
                .into_iter()
                .map(|(signal_uid, frequency)| SignalFrequency {
                    signal: Rc::new(Signal {
                        uid: signal_uid,
                        kind: SignalKind::IQ,
                        channels: vec![],
                        oscillator: Some(Oscillator {
                            uid: "osc".to_string(),
                            kind: OscillatorKind::HARDWARE,
                        }),
                        signal_delay: 0,
                        start_delay: 0,
                        mixer_type: None,
                    }),
                    frequency,
                })
                .collect();
            let obj = SetOscillatorFrequency::new(values, iteration);
            NodeKind::SetOscillatorFrequency(obj)
        }

        /// Test that oscillator frequency sweeps are correctly calculated and replaced in the IR tree.
        #[test]
        fn test_osc_freq_sweep_parameter_calculator() {
            let awg = AwgCore::new(
                0,
                cjob::AwgKind::IQ,
                vec![Rc::new(Signal {
                    uid: "test".to_string(),
                    kind: SignalKind::IQ,
                    channels: vec![],
                    oscillator: Some(Oscillator {
                        uid: "osc".to_string(),
                        kind: OscillatorKind::HARDWARE,
                    }),
                    signal_delay: 0,
                    start_delay: 0,
                    mixer_type: None,
                })],
                1e9,
                Arc::new(Device::new(
                    "test_device".to_string().into(),
                    DeviceKind::SHFSG,
                )),
                HashMap::from([("osc".to_string(), 0)]),
            );
            let n_iterations = 3;
            let start_freq = 1e9;
            let freq_step = 0.5e9;
            let osc_index = 0;

            let mut root = IrNode::new(NodeKind::Nop { length: 0 }, 0);
            for i in 0..n_iterations {
                root.add_child(
                    i * 32,
                    make_set_oscillator_frequency(
                        vec![("test".to_string(), start_freq + (i as f64) * freq_step)],
                        i.try_into().unwrap(),
                    ),
                );
            }
            handle_oscillator_sweeps(&mut root, &awg, &mut HashSet::default()).unwrap();

            let children = root.iter_children().collect::<Vec<_>>();
            // Check that the target nodes has been replaced with a SetOscillatorFrequencySweep node
            let target_first = children.first().unwrap();
            let expected_first = SetOscillatorFrequencySweep {
                length: 0,
                oscillators: vec![OscillatorFrequencySweepStep {
                    iteration: 0,
                    osc_index,
                    parameter: Arc::new(LinearParameterInfo {
                        start: start_freq,
                        step: freq_step,
                        count: n_iterations as usize,
                    }),
                }],
            };
            assert_set_oscillator_sweep(target_first.data(), &expected_first);

            // Check that the last node has the correct frequency sweep
            let target_last = children.last().unwrap();
            let expected_last = SetOscillatorFrequencySweep {
                length: 0,
                oscillators: vec![OscillatorFrequencySweepStep {
                    iteration: 2,
                    osc_index,
                    parameter: Arc::new(LinearParameterInfo {
                        start: start_freq,
                        step: freq_step,
                        count: n_iterations as usize,
                    }),
                }],
            };
            assert_set_oscillator_sweep(target_last.data(), &expected_last);
        }
    }
}

// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use anyhow::anyhow;
use std::cmp::max;
use std::collections::HashMap;
use std::rc::Rc;

use crate::Result;
use crate::ir::compilation_job::{self as cjob};
use crate::{ir, tinysample};

struct SignalPhaseTracker {
    cumulative: f64,
    // Reference time of last phase set time
    reference_time: ir::Samples,
}

struct PhaseTracker {
    trackers: HashMap<String, SignalPhaseTracker>,
    global_reset_time: ir::Samples,
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

    fn set(&mut self, signal: &cjob::Signal, ts: ir::Samples, value: f64) {
        let tracker = self.trackers.get_mut(&signal.uid).expect("Unknown signal");
        tracker.cumulative = value;
        tracker.reference_time = ts;
    }

    fn increment(&mut self, signal: &cjob::Signal, value: f64) {
        let tracker = self.trackers.get_mut(&signal.uid).expect("Unknown signal");
        tracker.cumulative += value;
    }

    fn global_reset(&mut self, ts: ir::Samples) {
        for tracker in self.trackers.values_mut() {
            tracker.cumulative = 0.0;
        }
        self.global_reset_time = ts;
    }

    pub fn phase_now(&self, signal: &cjob::Signal) -> (ir::Samples, f64) {
        let tracker = self.trackers.get(&signal.uid).expect("Unknown signal");
        let time_ref = max(tracker.reference_time, self.global_reset_time);
        let phase = tracker.cumulative;
        (time_ref, phase)
    }

    pub fn calculate_phase_at(&self, signal: &cjob::Signal, freq: f64, ts: ir::Samples) -> f64 {
        let (ref_time, phase_now) = self.phase_now(signal);
        let t = (ts as f64 - ref_time as f64) * tinysample::TINYSAMPLE;
        t * 2.0 * std::f64::consts::PI * freq + phase_now
    }
}

pub struct SoftwareOscillatorParameters {
    active_osc_freq: HashMap<String, Vec<f64>>,
    pulse_osc_freq: HashMap<(String, ir::Samples), f64>,
    sw_osc_phases: HashMap<(String, ir::Samples), f64>,
}

impl SoftwareOscillatorParameters {
    fn set_osc_freq(&mut self, signal: &cjob::Signal, value: f64) {
        self.active_osc_freq
            .entry(signal.uid.to_owned())
            .or_default()
            .push(value);
    }

    fn timestamp_osc_freq(&mut self, signal: &cjob::Signal, time: ir::Samples) {
        let freq = self
            .active_osc_freq
            .get(&signal.uid)
            .map_or(&0.0, |x| x.last().unwrap_or(&0.0));
        self.pulse_osc_freq
            .insert((signal.uid.to_owned(), time), *freq);
    }

    /// Frequency for selected signal at given timestamp
    pub fn freq_at(&self, signal: &cjob::Signal, ts: ir::Samples) -> Option<f64> {
        self.pulse_osc_freq
            .get(&(signal.uid.to_owned(), ts))
            .copied()
    }

    /// Phase for selected signal at given timestamp
    pub fn phase_at(&self, signal: &cjob::Signal, ts: ir::Samples) -> Option<f64> {
        self.sw_osc_phases
            .get(&(signal.uid.to_owned(), ts))
            .copied()
    }
}

fn collect_osc_parameters(
    node: &mut ir::IrNode,
    state: &mut SoftwareOscillatorParameters,
    phase_tracker: &mut Option<PhaseTracker>,
    sampling_rate: &f64,
) -> Result<()> {
    let node_offset = *node.offset();
    match node.data_mut() {
        ir::NodeKind::SetOscillatorFrequency(x) => {
            for (signal, freq) in x.values.iter() {
                if signal.is_sw_modulated() {
                    state.set_osc_freq(signal, *freq);
                }
            }
            node.replace_data(ir::NodeKind::Nop {
                length: node.data().length(),
            });
            Ok(())
        }
        ir::NodeKind::PlayPulse(ob) => {
            if let Some(osc) = &ob.signal.oscillator {
                if osc.kind == cjob::OscillatorKind::HARDWARE {
                    if ob.set_oscillator_phase.is_some() {
                        let msg = format!("Cannot set phase of hardware oscillator: {}", osc.uid);
                        return Err(anyhow!(msg).into());
                    }
                    return Ok(());
                }
            }
            // TODO: More elegant way to map to the node
            let offset = tinysample::tinysample_to_samples(node_offset, *sampling_rate);
            state.timestamp_osc_freq(&ob.signal, offset);
            if let Some(tracker) = phase_tracker {
                // Set oscillator phase priority over incrementing
                if let Some(set_oscillator_phase) = ob.set_oscillator_phase {
                    tracker.set(&ob.signal, node_offset, set_oscillator_phase);
                    // PlayIR nodes that have `set_oscillator_phase` and no `pulse_def` can be pruned.
                    // Perhaps it this could be a separate node altogether?
                    if ob.pulse_def.is_none() {
                        node.replace_data(ir::NodeKind::Nop {
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
        ir::NodeKind::AcquirePulse(ob) => {
            let offset = tinysample::tinysample_to_samples(node_offset, *sampling_rate);
            state.timestamp_osc_freq(&ob.signal, offset);
            Ok(())
        }
        ir::NodeKind::PhaseReset(ob) => {
            if ob.reset_sw_oscillators {
                if let Some(tracker) = phase_tracker {
                    tracker.global_reset(node_offset);
                }
            }
            node.replace_data(ir::NodeKind::Nop {
                length: node.data().length(),
            });
            Ok(())
        }
        _ => {
            for child in node.iter_children_mut() {
                collect_osc_parameters(child, state, phase_tracker, sampling_rate)?;
            }
            Ok(())
        }
    }
}

/// Pass to handle software oscillator parameters
///
/// Consumes oscillator frequency and phase nodes from the tree
/// and returns calculated oscillator frequency and phase values for each pulse at given time in target device unit samples.
/// Tiny sample is used to avoid potential rounding errors when calculating phase increments.
pub fn handle_oscillator_parameters(
    node: &mut ir::IrNode,
    signals: &[Rc<cjob::Signal>],
    device_kind: &cjob::DeviceKind,
    sampling_rate: &f64,
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
    collect_osc_parameters(node, &mut state, &mut phase_tracker, sampling_rate)?;
    Ok(state)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::tinysample::length_to_samples;

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
            delay: 0,
            mixer_type: None,
        };
        Rc::new(sig)
    }

    fn make_reset(reset_sw_oscillators: bool) -> ir::NodeKind {
        ir::NodeKind::PhaseReset(ir::PhaseReset {
            reset_sw_oscillators,
        })
    }

    fn make_pulse(
        signal: Rc<cjob::Signal>,
        set_oscillator_phase: Option<f64>,
        increment_oscillator_phase: Option<f64>,
    ) -> ir::NodeKind {
        ir::NodeKind::PlayPulse(ir::PlayPulse {
            signal,
            set_oscillator_phase,
            increment_oscillator_phase,
            length: 32,
            amp_param_name: None,
            id_pulse_params: None,
            amplitude: None,
            incr_phase_param_name: None,
            markers: vec![],
            pulse_def: Some(Arc::new(cjob::PulseDef {
                uid: "param".to_string(),
                kind: cjob::PulseDefKind::Pulse,
            })),
            phase: 0.0,
        })
    }

    #[test]
    fn test_phase_increment() {
        let sampling_rate = 2.4e9;
        let signal = make_signal("test", cjob::OscillatorKind::SOFTWARE);
        let mut root = ir::IrNode::new(ir::NodeKind::Nop { length: 0 }, 0);
        root.add_child(0, make_reset(true));
        root.add_child(
            tinysample::samples_to_tinysample(1),
            make_pulse(signal.clone(), None, Some(0.5)),
        );
        root.add_child(
            tinysample::samples_to_tinysample(3),
            make_pulse(signal.clone(), None, Some(0.5)),
        );
        let mut nested_section = ir::IrNode::new(
            ir::NodeKind::Nop { length: 0 },
            tinysample::samples_to_tinysample(5),
        );
        nested_section.add_child(tinysample::samples_to_tinysample(5), make_reset(true));
        nested_section.add_child(
            tinysample::samples_to_tinysample(7),
            make_pulse(signal.clone(), None, Some(0.5)),
        );
        root.add_child_node(nested_section);
        root.add_child(
            tinysample::samples_to_tinysample(11),
            make_pulse(signal.clone(), None, Some(0.5)),
        );

        let params = handle_oscillator_parameters(
            &mut root,
            &[signal.clone()],
            &cjob::DeviceKind::SHFSG,
            &sampling_rate,
        )
        .unwrap();
        assert_eq!(
            params
                .phase_at(&signal, length_to_samples(1.0, sampling_rate))
                .unwrap(),
            0.5
        );
        assert_eq!(
            params
                .phase_at(&signal, length_to_samples(3.0, sampling_rate))
                .unwrap(),
            1.0
        );
        assert_eq!(
            params
                .phase_at(&signal, length_to_samples(7.0, sampling_rate))
                .unwrap(),
            0.5
        );
        assert_eq!(
            params
                .phase_at(&signal, length_to_samples(11.0, sampling_rate))
                .unwrap(),
            1.0
        );
    }

    #[test]
    fn test_phase_increment_no_sw_reset() {
        let sampling_rate = 2.4e9;
        let signal = make_signal("test", cjob::OscillatorKind::SOFTWARE);
        let mut root = ir::IrNode::new(ir::NodeKind::Nop { length: 0 }, 0);
        root.add_child(
            tinysample::samples_to_tinysample(0),
            make_pulse(signal.clone(), None, Some(0.5)),
        );
        root.add_child(tinysample::samples_to_tinysample(1), make_reset(false));
        root.add_child(
            tinysample::samples_to_tinysample(3),
            make_pulse(signal.clone(), None, Some(0.5)),
        );

        let params = handle_oscillator_parameters(
            &mut root,
            &[signal.clone()],
            &cjob::DeviceKind::SHFSG,
            &sampling_rate,
        )
        .unwrap();
        assert_eq!(
            params
                .phase_at(&signal, length_to_samples(0.0, sampling_rate))
                .unwrap(),
            0.5
        );
        assert_eq!(
            params
                .phase_at(&signal, length_to_samples(3.0, sampling_rate))
                .unwrap(),
            1.0
        );
    }

    #[test]
    fn test_phase_increment_hw_osc() {
        let sampling_rate = 2.4e9;
        let signal = make_signal("test", cjob::OscillatorKind::HARDWARE);
        let mut root = ir::IrNode::new(ir::NodeKind::Nop { length: 0 }, 0);
        root.add_child(
            tinysample::samples_to_tinysample(0),
            make_pulse(signal.clone(), None, Some(0.5)),
        );
        root.add_child(
            tinysample::samples_to_tinysample(1),
            make_pulse(signal.clone(), None, Some(0.5)),
        );

        let params = handle_oscillator_parameters(
            &mut root,
            &[signal.clone()],
            &cjob::DeviceKind::SHFSG,
            &sampling_rate,
        )
        .unwrap();
        assert!(
            params
                .phase_at(&signal, length_to_samples(0.0, sampling_rate))
                .is_none()
        );
        assert!(
            params
                .phase_at(&signal, length_to_samples(1.0, sampling_rate))
                .is_none()
        );
    }

    #[test]
    fn test_set_phase() {
        let sampling_rate = 2.4e9;
        let signal = make_signal("test", cjob::OscillatorKind::SOFTWARE);
        let mut root = ir::IrNode::new(ir::NodeKind::Nop { length: 0 }, 0);
        root.add_child(
            tinysample::samples_to_tinysample(0),
            make_pulse(signal.clone(), Some(1.0), Some(0.5)),
        );
        root.add_child(
            tinysample::samples_to_tinysample(1),
            make_pulse(signal.clone(), None, Some(0.5)),
        );
        root.add_child(tinysample::samples_to_tinysample(3), make_reset(true));
        root.add_child(
            tinysample::samples_to_tinysample(5),
            make_pulse(signal.clone(), Some(1.0), None),
        );

        let params = handle_oscillator_parameters(
            &mut root,
            &[signal.clone()],
            &cjob::DeviceKind::SHFSG,
            &sampling_rate,
        )
        .unwrap();
        // Set phase wins over phase increments
        assert_eq!(
            params
                .phase_at(&signal, length_to_samples(0.0, sampling_rate))
                .unwrap(),
            1.0
        );
        assert_eq!(
            params
                .phase_at(&signal, length_to_samples(1.0, sampling_rate))
                .unwrap(),
            1.5
        );
        assert_eq!(
            params
                .phase_at(&signal, length_to_samples(5.0, sampling_rate))
                .unwrap(),
            1.0
        );
    }

    #[test]
    fn test_set_phase_no_osc() {
        let sampling_rate = 2.4e9;
        let sig = cjob::Signal {
            uid: "test".to_string(),
            kind: cjob::SignalKind::IQ,
            channels: vec![],
            oscillator: None,
            delay: 0,
            mixer_type: None,
        };
        let signal = Rc::new(sig);
        let mut root = ir::IrNode::new(ir::NodeKind::Nop { length: 0 }, 0);
        root.add_child(
            tinysample::samples_to_tinysample(0),
            make_pulse(signal.clone(), Some(1.0), Some(0.5)),
        );
        root.add_child(
            tinysample::samples_to_tinysample(1),
            make_pulse(signal.clone(), None, Some(0.5)),
        );
        root.add_child(tinysample::samples_to_tinysample(3), make_reset(true));
        root.add_child(
            tinysample::samples_to_tinysample(5),
            make_pulse(signal.clone(), Some(1.0), None),
        );

        let params = handle_oscillator_parameters(
            &mut root,
            &[signal.clone()],
            &cjob::DeviceKind::SHFSG,
            &sampling_rate,
        )
        .unwrap();
        // Set phase wins over phase increments
        assert_eq!(
            params
                .phase_at(&signal, length_to_samples(0.0, sampling_rate))
                .unwrap(),
            1.0
        );
        assert_eq!(
            params
                .phase_at(&signal, length_to_samples(1.0, sampling_rate))
                .unwrap(),
            1.5
        );
        assert_eq!(
            params
                .phase_at(&signal, length_to_samples(5.0, sampling_rate))
                .unwrap(),
            1.0
        );
    }
}

// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::collections::HashSet;

use laboneq_common::types::AwgKey;
use laboneq_dsl::types::OscillatorKind;
use laboneq_dsl::types::{AmplifierPump, ParameterUid, SectionUid, SignalUid, ValueOrParameter};
use laboneq_ir::SetOscillatorFrequency;
use laboneq_ir::TinySamples;
use laboneq_ir::{IrKind, PpcStep};
use laboneq_units::duration::seconds;
use laboneq_units::tinysample::seconds_to_tinysamples;
use laboneq_units::tinysample::tiny_samples;

use crate::error::{Error, Result};
use crate::experiment_context::ExperimentContext;
use crate::lower_experiment::handle_reset_oscillator_phase;
use crate::schedule_info::ScheduleInfoBuilder;
use crate::scheduled_node::NodeChild;
use crate::signal_grids::SignalGrids;
use crate::utils::ceil_to_grid;
use crate::{ScheduledNode, SignalInfo};

/// Gives every loop iteration its preamble.
///
/// The preamble of a loop iteration prepares the signals of the iteration's body, and its
/// PPC steps may add signals the body does not use. These signals are added to the
/// iteration and its ancestors. The preamble occupies all signals of its iteration.
///
/// Expects the signals of every node to be collected.
pub(crate) fn build_preambles<T: SignalInfo>(
    root: &mut ScheduledNode,
    ctx: &ExperimentContext<T>,
    grids: &SignalGrids,
) -> Result<()> {
    let builder = PreambleBuilder { ctx, grids };
    builder.visit(root, None)?;
    Ok(())
}

/// The loop data that the preamble of an iteration depends on.
struct EnclosingLoop<'a> {
    uid: SectionUid,
    parameters: &'a [ParameterUid],
    reset_oscillator_phase: bool,
}

struct PreambleBuilder<'a, T: SignalInfo> {
    ctx: &'a ExperimentContext<'a, T>,
    grids: &'a SignalGrids,
}

impl<T: SignalInfo> PreambleBuilder<'_, T> {
    /// Builds the preambles in `node` and below, and returns the signals that they add.
    fn visit(
        &self,
        node: &mut ScheduledNode,
        enclosing_loop: Option<&EnclosingLoop>,
    ) -> Result<HashSet<SignalUid>> {
        let ScheduledNode {
            kind,
            schedule,
            children,
            ..
        } = node;
        let mut added_signals = HashSet::new();
        let loop_ = match &*kind {
            IrKind::Loop(obj) => Some(EnclosingLoop {
                uid: obj.uid,
                parameters: obj.parameters(),
                reset_oscillator_phase: schedule.reset_oscillator_phase,
            }),
            _ => None,
        };
        for child in children.iter_mut() {
            if child.node.holds_children() {
                added_signals.extend(self.visit(&mut child.node, loop_.as_ref())?);
            }
        }
        schedule.signals.extend(&added_signals);

        if let IrKind::LoopIteration = kind {
            let loop_ = enclosing_loop.expect("Internal error: Loop iteration outside a loop");
            let mut preamble = self.build_preamble(&body_signals(children), loop_)?;
            for step in &preamble.children {
                for &signal in step.node.kind.signals() {
                    schedule.signals.insert(signal);
                    added_signals.insert(signal);
                }
            }
            preamble.schedule.signals = schedule.signals.clone();
            children.insert(
                0,
                NodeChild {
                    offset: tiny_samples(0),
                    node: preamble,
                },
            );
        }
        Ok(added_signals)
    }

    /// Builds the preamble of a loop iteration whose body occupies `body_signals`.
    fn build_preamble(
        &self,
        body_signals: &HashSet<SignalUid>,
        loop_: &EnclosingLoop,
    ) -> Result<ScheduledNode> {
        let mut preamble = ScheduledNode::new(
            IrKind::LoopIterationPreamble,
            ScheduleInfoBuilder::new().build(),
        );

        let signal_refs = body_signals
            .iter()
            .map(|s| self.ctx.get_signal(s))
            .collect::<Result<Vec<_>>>()?;

        // Add oscillator frequency setup
        if let Some(set_osc_freq) = handle_set_oscillator_frequency(
            &signal_refs,
            loop_.parameters.iter().collect(),
            self.grids.system_grid(),
        )? {
            preamble.add_child(tiny_samples(0), set_osc_freq);
        }

        // Add phase reset if needed
        if loop_.reset_oscillator_phase && !signal_refs.is_empty() {
            let reset_node = handle_reset_oscillator_phase(
                &signal_refs,
                self.ctx,
                self.grids.system_grid(),
                loop_.uid,
            )?;
            preamble.add_child(tiny_samples(0), reset_node);
        }

        // Add PPC sweep steps
        let all_signals: Vec<_> = self.ctx.signals().collect();
        let ppc_steps = handle_ppc_sweep_steps(&all_signals, loop_.parameters)?;
        for ppc_step in ppc_steps {
            preamble.add_child(tiny_samples(0), ppc_step);
        }
        Ok(preamble)
    }
}

/// Returns the signals of a loop iteration's body, without the iteration's own signals.
fn body_signals(children: &[NodeChild]) -> HashSet<SignalUid> {
    let mut signals = HashSet::new();
    for child in children {
        if child.node.holds_children() {
            signals.extend(&child.node.schedule.signals);
        } else {
            signals.extend(child.node.kind.signals());
        }
    }
    signals
}

/// Create PPC sweep step nodes for all `signals` that have an amplifier pump defined
/// and are swept by any of the given `parameters`.
///
/// The created PPC step nodes are grouped by AWG and the PPC steps
/// are assigned to a single signal on that AWG.
fn handle_ppc_sweep_steps(
    signals: &[&impl SignalInfo],
    parameters: &[ParameterUid],
) -> Result<Vec<ScheduledNode>> {
    let ppc_steps = signals
        .iter()
        .filter_map(|signal| {
            signal.amplifier_pump().map(|amplifier_pump| {
                create_ppc_step_for_parameter(signal.uid(), amplifier_pump, parameters)
            })
        })
        .flatten();
    let nodes = group_ppc_steps_by_awg(ppc_steps, signals)?
        .map(|ppc_step| {
            ScheduledNode::new(
                IrKind::PpcStep(ppc_step),
                ScheduleInfoBuilder::new().build(),
            )
        })
        .collect();
    Ok(nodes)
}

fn create_ppc_step_for_parameter(
    signal: SignalUid,
    pump: &AmplifierPump,
    parameters: &[ParameterUid],
) -> Option<PpcStep> {
    let pump_power = if let Some(ValueOrParameter::Parameter(param_uid)) = &pump.pump_power
        && parameters.contains(param_uid)
    {
        pump.pump_power
    } else {
        None
    };
    let pump_frequency = if let Some(ValueOrParameter::Parameter(param_uid)) = &pump.pump_frequency
        && parameters.contains(param_uid)
    {
        pump.pump_frequency
    } else {
        None
    };
    let probe_power = if let Some(ValueOrParameter::Parameter(param_uid)) = &pump.probe_power
        && parameters.contains(param_uid)
    {
        pump.probe_power
    } else {
        None
    };
    let probe_frequency = if let Some(ValueOrParameter::Parameter(param_uid)) =
        &pump.probe_frequency
        && parameters.contains(param_uid)
    {
        pump.probe_frequency
    } else {
        None
    };
    let cancellation_phase = if let Some(ValueOrParameter::Parameter(param_uid)) =
        &pump.cancellation_phase
        && parameters.contains(param_uid)
    {
        pump.cancellation_phase
    } else {
        None
    };
    let cancellation_attenuation = if let Some(ValueOrParameter::Parameter(param_uid)) =
        &pump.cancellation_attenuation
        && parameters.contains(param_uid)
    {
        pump.cancellation_attenuation
    } else {
        None
    };
    if pump_power.is_none()
        && pump_frequency.is_none()
        && probe_power.is_none()
        && probe_frequency.is_none()
        && cancellation_phase.is_none()
        && cancellation_attenuation.is_none()
    {
        // Fields are not swept by any of the given parameters
        return None;
    }
    let ppc_step = PpcStep {
        signal,
        trigger_duration: tiny_samples(0), // Resolved later
        pump_power,
        pump_frequency,
        probe_power,
        probe_frequency,
        cancellation_phase,
        cancellation_attenuation,
    };
    Some(ppc_step)
}

fn group_ppc_steps_by_awg<T: SignalInfo>(
    ppc_steps: impl Iterator<Item = PpcStep>,
    signals: &[&T],
) -> Result<impl Iterator<Item = PpcStep>> {
    let signals = signals
        .iter()
        .map(|s| (s.uid(), s))
        .collect::<HashMap<_, _>>();

    let mut step_per_awg: HashMap<AwgKey, PpcStep> = HashMap::new();
    for step in ppc_steps {
        let awg = signals.get(&step.signal).unwrap().awg_key();
        if let Some(existing_steps) = step_per_awg.get_mut(&awg) {
            merge_ppc_step(existing_steps, step)?;
        } else {
            step_per_awg.insert(awg, step);
        }
    }
    Ok(step_per_awg.into_values())
}

fn merge_ppc_step(one: &mut PpcStep, other: PpcStep) -> Result<()> {
    // TODO: Check for conflicts?
    if let Some(pump_power) = other.pump_power {
        one.pump_power = pump_power.into();
    }
    if let Some(pump_frequency) = other.pump_frequency {
        one.pump_frequency = pump_frequency.into();
    }
    if let Some(probe_power) = other.probe_power {
        one.probe_power = probe_power.into();
    }
    if let Some(probe_frequency) = other.probe_frequency {
        one.probe_frequency = probe_frequency.into();
    }
    if let Some(cancellation_phase) = other.cancellation_phase {
        one.cancellation_phase = cancellation_phase.into();
    }
    if let Some(cancellation_attenuation) = other.cancellation_attenuation {
        one.cancellation_attenuation = cancellation_attenuation.into();
    }
    Ok(())
}

/// Creates IR node to set oscillator frequencies for `signals` that have their oscillator frequency
/// controlled by one of the given `parameters`.
pub(crate) fn handle_set_oscillator_frequency<T: SignalInfo + Sized>(
    signals: &[&T],
    parameters: HashSet<&ParameterUid>,
    system_grid: TinySamples,
) -> Result<Option<ScheduledNode>> {
    if parameters.is_empty() {
        return Ok(None);
    }
    let mut swept_hw_osc_signals: Vec<&T> = Vec::new();
    let mut signal_osc_value = vec![];
    let mut oscillator_type: Option<OscillatorKind> = None;
    for signal in signals {
        if let Some(osc) = signal.oscillator()
            && let ValueOrParameter::Parameter(param_uid) = osc.frequency
            && parameters.contains(&param_uid)
        {
            match oscillator_type.as_mut() {
                None => oscillator_type = Some(osc.kind),
                Some(existing_type) => {
                    if *existing_type != osc.kind {
                        return Err(Error::new(
                            "Cannot sweep mixed hardware and software oscillators.",
                        ));
                    }
                }
            }
            swept_hw_osc_signals.push(signal);
            signal_osc_value.push((signal.uid(), osc.frequency));
        }
    }
    if signal_osc_value.is_empty() {
        return Ok(None);
    }
    let length = if oscillator_type == Some(OscillatorKind::Hardware) {
        swept_hw_osc_signals
            .iter()
            .fold(seconds(0.0), |acc, signal| {
                acc.max(signal.oscillator_set_latency())
            })
    } else {
        seconds(0.0)
    };
    let node = ScheduledNode::new(
        IrKind::SetOscillatorFrequency(SetOscillatorFrequency {
            values: signal_osc_value,
        }),
        ScheduleInfoBuilder::new()
            .grid(system_grid)
            .length(tiny_samples(ceil_to_grid(
                seconds_to_tinysamples(length).value(),
                system_grid.value(),
            )))
            .build(),
    );
    Ok(Some(node))
}

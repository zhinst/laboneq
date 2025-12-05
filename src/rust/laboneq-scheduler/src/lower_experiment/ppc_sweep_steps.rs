// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use laboneq_common::types::AwgKey;
use laboneq_units::duration::{Duration, Second, seconds};
use laboneq_units::tinysample::{TinySamples, seconds_to_tinysamples, tiny_samples};

use crate::error::Result;
use crate::experiment::types::{AmplifierPump, ParameterUid, SignalUid, ValueOrParameter};

use crate::ir::{IrKind, PpcStep};
use crate::schedule_info::ScheduleInfoBuilder;
use crate::utils::round_to_grid;
use crate::{ScheduledNode, SignalInfo};

const PPC_TRIGGER_ASSERT_DURATION: Duration<Second> = seconds(25e-3);
const PPC_TRIGGER_DEASSERT_DURATION: Duration<Second> = seconds(0.1e-3);

/// Create PPC sweep step nodes for all `signals` that have an amplifier pump defined
/// and are swept by any of the given `parameters`.
///
/// The created PPC step nodes are grouped by AWG and the PPC steps
/// are assigned to a single signal on that AWG.
pub(super) fn handle_ppc_sweep_steps(
    signals: &[&impl SignalInfo],
    parameters: &[ParameterUid],
    grid: TinySamples,
) -> Result<Vec<ScheduledNode>> {
    let ppc_steps = signals
        .iter()
        .filter_map(|signal| {
            signal.amplifier_pump().map(|amplifier_pump| {
                create_ppc_step_for_parameter(signal.uid(), amplifier_pump, parameters)
            })
        })
        .flatten();
    group_ppc_steps_by_awg(ppc_steps, signals)?
        .map(|ppc_step| schedule_ppc_step(ppc_step, grid))
        .collect::<Result<Vec<ScheduledNode>>>()
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
        device: pump.device,
        channel: pump.channel,
        trigger_duration: seconds_to_tinysamples(PPC_TRIGGER_ASSERT_DURATION),
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

fn schedule_ppc_step(mut ppc_step: PpcStep, grid: TinySamples) -> Result<ScheduledNode> {
    // We currently assert the trigger for 25 ms. We then deassert
    // it for 0.1 ms. This guarantees that, if the next sweep step
    // were to occur shortly after, there is sufficient time for
    // PPC to register the falling edge.
    ppc_step.trigger_duration = tiny_samples(round_to_grid(
        ppc_step.trigger_duration.value(),
        grid.value(),
    ));
    let length = tiny_samples(round_to_grid(
        seconds_to_tinysamples(PPC_TRIGGER_DEASSERT_DURATION).value(),
        grid.value(),
    )) + ppc_step.trigger_duration;
    let scheduled_ppc_step = ScheduledNode::new(
        IrKind::PpcStep(ppc_step),
        ScheduleInfoBuilder::new().grid(grid).length(length).build(),
    );
    Ok(scheduled_ppc_step)
}

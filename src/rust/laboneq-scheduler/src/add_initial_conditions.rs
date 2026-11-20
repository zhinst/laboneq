// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use anyhow::Context as _;
use laboneq_dsl::types::{OscillatorKind, ValueOrParameter};
use laboneq_ir::{InitialOscillatorFrequency, IrKind, ParameterLookup, TinySamples};
use laboneq_units::tinysample::tiny_samples;

use crate::error::{Error, Result};
use crate::experiment_context::ExperimentContext;
use crate::schedule_info::ScheduleInfoBuilder;
use crate::scheduled_node::NodeChild;
use crate::{ScheduledNode, SignalInfo};

/// Puts the initial oscillator frequencies of the signals first among the children of `root`.
///
/// The signals are taken in the order of their names, so the output is deterministic.
pub(crate) fn add_initial_conditions<T: SignalInfo>(
    root: &mut ScheduledNode,
    ctx: &ExperimentContext<T>,
    nt_parameters: &dyn ParameterLookup,
    system_grid: TinySamples,
) -> Result<()> {
    let mut signals: Vec<_> = ctx.signals().collect();
    signals.sort_by_key(|sig| {
        ctx.id_store
            .resolve(sig.uid().0)
            .unwrap_or("unknown")
            .to_string()
    });

    let initial_osc_freq =
        handle_initial_oscillator_frequency(&signals, nt_parameters, system_grid)?;
    root.children.insert(
        0,
        NodeChild {
            offset: tiny_samples(0),
            node: initial_osc_freq,
        },
    );
    Ok(())
}

pub(crate) fn handle_initial_oscillator_frequency<T: SignalInfo + Sized>(
    signals: &[&T],
    parameters: &dyn ParameterLookup,
    system_grid: TinySamples,
) -> Result<ScheduledNode> {
    let mut values = Vec::new();
    for signal in signals {
        if let Some(osc) = signal.oscillator() {
            let freq_unwrapped = match &osc.frequency {
                ValueOrParameter::Parameter(obj) => {
                    if osc.kind == OscillatorKind::Hardware
                        && !signal.supports_initial_oscillator_frequency()
                    {
                        // HW oscillators are either set by the controller, or swept in a RT
                        // loop. Either way, no need to set the initial frequency here. If we
                        // did include it in the schedule this would count as a dependency on
                        // the parameter, and force needless recompilation of each NT step.
                        continue;
                    }
                    if let Some(freq) = parameters.value(*obj) {
                        freq.try_into().map_err(Error::new).context(
                            "Oscillator frequency must be a real number (integer or float).",
                        )?
                    } else {
                        continue;
                    }
                }
                value => *value,
            };
            values.push((signal.uid(), freq_unwrapped));
        }
    }
    let node = ScheduledNode::new(
        IrKind::InitialOscillatorFrequency(InitialOscillatorFrequency { values }),
        ScheduleInfoBuilder::new()
            .length(0)
            .grid(system_grid)
            .build(),
    );
    Ok(node)
}

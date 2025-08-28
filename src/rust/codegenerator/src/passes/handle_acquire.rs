// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use crate::ir::{AcquirePulse, IrNode, NodeKind, PlayAcquire, Samples};
use crate::passes::handle_oscillators::SoftwareOscillatorParameters;
use crate::tinysample::floor_to_grid;
use crate::{Error, Result};
use std::rc::Rc;

fn validate(pulse: &AcquirePulse) -> Result<()> {
    assert_ne!(pulse.length, 0, "Acquire length must be non-zero");
    if pulse.pulse_defs.len() != pulse.id_pulse_params.len() {
        return Err(Error::new(
            "Pulse definitions and ID pulse parameters must match in length",
        ));
    }
    Ok(())
}

fn lower_acquire_pulse(
    pulse: &AcquirePulse,
    offset: Samples,
    sample_multiple: u16,
    oscillator_frequency: f64,
) -> Result<(Samples, PlayAcquire)> {
    validate(pulse)?;
    // Timing adjustment for grid
    // TODO: The adjustment should not be necessary, the scheduler should
    // handle this. The current functionality matches the old implementation for now.
    let start = floor_to_grid(offset, sample_multiple.into());
    let end = floor_to_grid(offset + pulse.length, sample_multiple.into());
    let acquire = PlayAcquire::new(
        Rc::clone(&pulse.signal),
        end - start,
        pulse.pulse_defs.clone(),
        oscillator_frequency,
        pulse.id_pulse_params.clone(),
    );
    Ok((start, acquire))
}

struct PassContext<'a> {
    sample_multiple: u16,
    osc_params: &'a SoftwareOscillatorParameters,
}

/// Recursively transform `AcquirePulse` nodes to `PlayAcquire` nodes.
fn transform_acquires(node: &mut IrNode, ctx: &PassContext) -> Result<()> {
    if let NodeKind::AcquirePulse(pulse) = node.data() {
        // Oscillator parameters are calculated before timing adjustments, so
        // we must use the unadjusted offset to query the frequency
        let oscillator_frequency = ctx
            .osc_params
            .freq_at(&pulse.signal, *node.offset())
            .unwrap_or(0.0);
        let (offset_adjusted, acquire_code) = lower_acquire_pulse(
            pulse,
            *node.offset(),
            ctx.sample_multiple,
            oscillator_frequency,
        )?;
        *node.offset_mut() = offset_adjusted;
        node.replace_data(NodeKind::Acquire(acquire_code));
    } else {
        for child in node.iter_children_mut() {
            transform_acquires(child, ctx)?;
        }
    }
    Ok(())
}

/// Transforms [`AcquirePulse`]s nodes in the IR into [`PlayAcquire`]s.
///
/// The pass modifies the IR in place and adjusts the nodes offset with relevant delay.
///
/// * Applies delays to the acquisition nodes.
/// * Oscillator frequency is applied to the acquisition nodes.
pub fn handle_acquisitions(
    node: &mut IrNode,
    sample_multiple: u16,
    osc_params: &SoftwareOscillatorParameters,
) -> Result<()> {
    let ctx = PassContext {
        osc_params,
        sample_multiple,
    };
    transform_acquires(node, &ctx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::compilation_job::{Signal, SignalKind};

    /// Test acquire grid alignment
    #[test]
    fn test_lower_acquire_timing() {
        let length: Samples = 1020;
        let grid: Samples = 16;

        let pulse = AcquirePulse {
            signal: Signal {
                uid: "".to_string(),
                kind: SignalKind::INTEGRATION,
                signal_delay: 0,
                start_delay: 0,
                channels: vec![0],
                oscillator: None,
                mixer_type: None,
            }
            .into(),
            pulse_defs: vec![],
            id_pulse_params: vec![],
            length,
            handle: "".into(),
        };
        let offset = 0;
        let (start, play_acquire) = lower_acquire_pulse(&pulse, offset, grid as u16, 0.0).unwrap();
        assert_eq!(start, offset);
        assert_eq!(play_acquire.length(), length - length % grid);

        let offset = 360;
        let (start, play_acquire) = lower_acquire_pulse(&pulse, offset, grid as u16, 0.0).unwrap();
        assert_eq!(start, offset - offset % grid);
        assert_eq!(
            play_acquire.length(),
            length + (grid - (length % grid)) % grid
        );
    }
}

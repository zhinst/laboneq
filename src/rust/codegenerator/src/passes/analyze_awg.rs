// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::{collections::HashMap, sync::Arc};

use laboneq_dsl::types::SignalUid;

use crate::ir::compilation_job::{AwgCore, ChannelIndex, DeviceKind, SignalKind};
use crate::ir::experiment::Handle;
use crate::ir::{IrNode, NodeKind, PlayPulse, PpcDevice, TriggerBitData};
use crate::result::MarkerMode;
use crate::{Error, Result};

/// Maps a channel to marker.
fn marker_to_channel_hdawg(marker_selector: &str) -> u8 {
    match marker_selector {
        "marker1" => 0,
        "marker2" => 1,
        _ => panic!("Invalid marker selector: {}", marker_selector),
    }
}

/// Maps a channel to trigger bit on HDAWG.
fn channel_to_trigger_bit_hdawg(channel: u8) -> u8 {
    2u8.pow((channel % 2) as u32)
}

pub(crate) struct AwgCompilationInfo {
    device_kind: DeviceKind,
    ppc_device: Option<Arc<PpcDevice>>,
    feedback_handles: Vec<Handle>,
    pub marker_modes: HashMap<ChannelIndex, MarkerMode>,
}

impl AwgCompilationInfo {
    fn new(device_kind: DeviceKind) -> Self {
        Self {
            device_kind,
            ppc_device: None,
            feedback_handles: Vec::new(),
            marker_modes: HashMap::new(),
        }
    }

    pub(crate) fn has_readout_feedback(&self) -> bool {
        !self.feedback_handles.is_empty()
    }

    pub(crate) fn ppc_device(&self) -> Option<&Arc<PpcDevice>> {
        self.ppc_device.as_ref()
    }

    pub(crate) fn feedback_handles(&self) -> &Vec<Handle> {
        &self.feedback_handles
    }

    fn add_ppc_device(&mut self, ppc_device: &Arc<PpcDevice>) {
        if self.ppc_device.is_none() {
            self.ppc_device = Some(Arc::clone(ppc_device));
        } else if let Some(unique_ppc) = &self.ppc_device {
            assert_eq!(
                unique_ppc, ppc_device,
                "Internal error: Multiple SHFPPC devices found in the same AWG. \
                Only a single device and a single channel is supported."
            )
        }
    }

    fn visit_play_pulse(&mut self, play: &PlayPulse) -> Result<()> {
        let signal = &play.signal;
        // Process markers
        for marker in &play.markers {
            if signal.kind == SignalKind::IQ {
                if self.device_kind == DeviceKind::HDAWG {
                    let marker_channel = marker_to_channel_hdawg(&marker.marker_selector);
                    self.check_trigger_marker_conflict(
                        marker_channel,
                        signal.uid,
                        MarkerMode::Marker,
                    )?;
                    self.marker_modes.insert(marker_channel, MarkerMode::Marker);
                } else if self.device_kind == DeviceKind::SHFSG {
                    if marker.marker_selector != "marker1" {
                        return Err(Error::new("Only marker1 supported on SHFSG"));
                    }
                    for channel in &signal.channels {
                        self.check_trigger_marker_conflict(
                            *channel,
                            signal.uid,
                            MarkerMode::Marker,
                        )?;
                        self.marker_modes.insert(*channel, MarkerMode::Marker);
                    }
                }
            } else if signal.kind == SignalKind::SINGLE {
                self.marker_modes
                    .extend(signal.channels.iter().map(|&ch| (ch, MarkerMode::Marker)));
            }
        }
        Ok(())
    }

    fn visit_trigger_set(&mut self, trigger: &TriggerBitData) -> Result<()> {
        if !trigger.set || trigger.signal.kind == SignalKind::IQ && trigger.bits == 0 {
            return Ok(());
        }
        if self.device_kind == DeviceKind::HDAWG {
            for channel in &trigger.signal.channels {
                let trigger_bit = channel_to_trigger_bit_hdawg(*channel);
                if trigger.bits & trigger_bit != 0 {
                    self.check_trigger_marker_conflict(
                        *channel,
                        trigger.signal.uid,
                        MarkerMode::Trigger,
                    )?;
                    self.marker_modes.insert(*channel, MarkerMode::Trigger);
                }
            }
        } else if self.device_kind == DeviceKind::SHFSG {
            for channel in &trigger.signal.channels {
                self.check_trigger_marker_conflict(
                    *channel,
                    trigger.signal.uid,
                    MarkerMode::Trigger,
                )?;
                self.marker_modes.insert(*channel, MarkerMode::Trigger);
            }
        }
        Ok(())
    }

    fn check_trigger_marker_conflict(
        &self,
        channel: ChannelIndex,
        signal: SignalUid,
        marker_mode: MarkerMode,
    ) -> Result<()> {
        if let Some(mode) = self.marker_modes.get(&channel)
            && *mode != marker_mode
        {
            let msg = format!(
                "Using triggers and markers on the same channel '{}' on signal '{}' on '{}' is not allowed.",
                channel,
                signal.0,
                self.device_kind.as_str()
            );
            return Err(Error::new(msg));
        }
        Ok(())
    }
}

fn traverse_awg_ir(node: &IrNode, info: &mut AwgCompilationInfo) -> Result<()> {
    match node.data() {
        NodeKind::Match(ob) => {
            if let Some(handle) = ob.handle.as_ref() {
                info.feedback_handles.push(handle.clone());
            }
        }
        NodeKind::PpcSweepStep(ob) => {
            info.add_ppc_device(&ob.ppc_device);
        }
        NodeKind::PlayPulse(obj) => {
            info.visit_play_pulse(obj)?;
        }
        NodeKind::TriggerSet(obj) => {
            info.visit_trigger_set(obj)?;
        }
        _ => {}
    }
    for child in node.iter_children() {
        traverse_awg_ir(child, info)?;
    }
    Ok(())
}

pub(crate) fn analyze_awg_ir(node: &IrNode, awg: &AwgCore) -> Result<AwgCompilationInfo> {
    let mut info = AwgCompilationInfo::new(awg.device_kind().clone());
    traverse_awg_ir(node, &mut info)?;
    Ok(info)
}

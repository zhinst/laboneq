// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::sync::Arc;

use codegenerator_utils::pulse_parameters::PulseParameterDeduplicator;
use laboneq_common::named_id::NamedIdStore;
use laboneq_dsl::types::{
    ComplexOrFloat, DeviceUid, MarkerSelector, ParameterUid, PulseUid, SectionUid, SignalUid,
    ValueOrParameter,
};
use laboneq_ir as hir;
use laboneq_ir::node::IrNode as HirNode;
use laboneq_units::tinysample::{TinySamples, tiny_samples};
use num_complex::Complex;

use crate::Result;
use crate::ir::compilation_job::{Marker, PulseDef, Signal, SweepParameter};
use crate::ir::experiment::{PulseParametersId, SweepCommand};
use crate::ir::{
    AcquirePulse, Case, InitialOscillatorFrequency, IrNode, Loop, LoopIteration, Match, NodeKind,
    PhaseReset, PlayPulse, PpcDevice, PpcSweepStep, PrngSetup, Section, SectionId, SectionInfo,
    SetOscillatorFrequency, SignalFrequency,
};

pub struct CodegenIr {
    pub root: IrNode,
    pub pulse_parameters: PulseParameterDeduplicator,
}

/// Lower a IR into a codegenerator IR.
pub fn ir_to_codegen_ir(
    node: &HirNode,
    id_store: &NamedIdStore,
    signals: Vec<&Arc<Signal>>,
    pulse_defs: Vec<&Arc<PulseDef>>,
    parameters: Vec<&Arc<SweepParameter>>,
) -> Result<CodegenIr> {
    let mut lowerer = IrToCodeIrLowerer::new(id_store, signals, pulse_defs, parameters);
    let codegen_ir = transform_ir(&mut lowerer, node, tiny_samples(0))?;
    let result = CodegenIr {
        root: codegen_ir,
        pulse_parameters: lowerer.pulse_parameter_deduplicator,
    };
    Ok(result)
}

fn transform_ir(
    ctx: &mut IrToCodeIrLowerer,
    node: &HirNode,
    offset: TinySamples,
) -> Result<IrNode> {
    let kind = match &node.kind {
        hir::IrKind::PlayPulse(obj) => {
            let code_ir = ctx.play_pulse_to_code(obj, node.length)?;
            NodeKind::PlayPulse(code_ir)
        }
        hir::IrKind::Delay { signal } => {
            let code_ir = ctx.delay_to_code(signal, node.length)?;
            NodeKind::PlayPulse(code_ir)
        }
        hir::IrKind::ChangeOscillatorPhase(obj) => {
            let code_ir = ctx.change_oscillator_phase_to_code(obj, node.length)?;
            NodeKind::PlayPulse(code_ir)
        }
        hir::IrKind::Acquire(obj) => {
            let code_ir = ctx.acquire_to_code(obj)?;
            NodeKind::AcquirePulse(code_ir)
        }
        hir::IrKind::Section(obj) => {
            let code_ir = ctx.section_to_code(obj, node.length)?;
            NodeKind::Section(code_ir)
        }
        hir::IrKind::LoopIteration => NodeKind::LoopIteration(LoopIteration {
            length: node.length.value(),
        }),
        hir::IrKind::Loop(obj) => {
            let code_ir = ctx.loop_to_code(obj, node.length);
            NodeKind::Loop(code_ir)
        }
        hir::IrKind::LoopIterationPreamble => NodeKind::Nop {
            length: node.length.value(),
        },
        hir::IrKind::Match(obj) => {
            let code_ir = ctx.lower_match_to_code(obj, node.length);
            NodeKind::Match(code_ir)
        }
        hir::IrKind::Case(obj) => {
            let code_ir = ctx.case_to_code(obj, node.length);
            NodeKind::Case(code_ir)
        }
        hir::IrKind::SetOscillatorFrequency(obj) => {
            let code_ir = ctx.set_osc_frequency_to_code(obj);
            NodeKind::SetOscillatorFrequency(code_ir)
        }
        hir::IrKind::InitialOscillatorFrequency(obj) => {
            let code_ir = ctx.initial_osc_frequency_to_code(obj);
            NodeKind::InitialOscillatorFrequency(code_ir)
        }
        hir::IrKind::ResetOscillatorPhase { signals } => {
            let code_ir = ctx.reset_osc_phase_to_code(signals);
            NodeKind::PhaseReset(code_ir)
        }
        hir::IrKind::PpcStep(obj) => {
            let code_ir = ctx.ppc_step_to_code(obj);
            NodeKind::PpcSweepStep(code_ir)
        }
        hir::IrKind::ClearPrecompensation { signal } => NodeKind::PrecompensationFilterReset {
            signal: ctx.get_signal(signal),
        },
        hir::IrKind::Root => NodeKind::Nop {
            length: node.length.value(),
        },
        _ => panic!("Codegenerator encountered unexpected node: {:?}", node.kind),
    };

    // Construct the code IR node and recursively process children
    let mut code_node = IrNode::new(kind, offset.value());
    for child in node.children.iter() {
        let child_code_ir = transform_ir(ctx, &child.node, child.offset)?;
        code_node.add_child_node(child_code_ir);
    }
    Ok(code_node)
}

struct IrToCodeIrLowerer<'a> {
    // Reference structures
    id_store: &'a NamedIdStore,
    signals: HashMap<SignalUid, &'a Arc<Signal>>,
    pulse_defs: HashMap<PulseUid, &'a Arc<PulseDef>>,
    parameters: HashMap<ParameterUid, &'a Arc<SweepParameter>>,
    // Build time structures
    // These are populated during lowering for deduplication and cross-referencing
    pulse_parameter_deduplicator: PulseParameterDeduplicator,
    section_info: HashMap<SectionUid, Arc<SectionInfo>>,
    ppc_devices: HashMap<SignalUid, Arc<PpcDevice>>,
}

impl<'a> IrToCodeIrLowerer<'a> {
    fn new(
        id_store: &'a NamedIdStore,
        signals: Vec<&'a Arc<Signal>>,
        pulse_defs: Vec<&'a Arc<PulseDef>>,
        parameters: Vec<&'a Arc<SweepParameter>>,
    ) -> Self {
        let signals_map = signals
            .into_iter()
            .map(|signal| (signal.uid, signal))
            .collect();
        let pulse_defs_map = pulse_defs
            .into_iter()
            .map(|pulse_def| (id_store.get(&pulse_def.uid).unwrap().into(), pulse_def))
            .collect();
        let parameter_map = parameters
            .into_iter()
            .map(|param| (id_store.get(&param.uid).unwrap().into(), param))
            .collect();
        IrToCodeIrLowerer {
            id_store,
            signals: signals_map,
            pulse_defs: pulse_defs_map,
            parameters: parameter_map,
            pulse_parameter_deduplicator: PulseParameterDeduplicator::new(),
            section_info: HashMap::new(),
            ppc_devices: HashMap::new(),
        }
    }

    fn get_signal(&self, uid: &SignalUid) -> Arc<Signal> {
        Arc::clone(self.signals[uid])
    }

    fn get_pulse_def(&self, uid: &PulseUid) -> Arc<PulseDef> {
        Arc::clone(self.pulse_defs[uid])
    }

    fn get_or_create_section_info(&mut self, uid: &SectionUid) -> Arc<SectionInfo> {
        if let Some(info) = self.section_info.get(uid) {
            Arc::clone(info)
        } else {
            let info = Arc::new(SectionInfo {
                id: self.section_info.len() as SectionId + 1,
                name: self.id_store.resolve_unchecked(*uid).to_string(),
            });
            self.section_info.insert(*uid, Arc::clone(&info));
            info
        }
    }

    fn get_sweep_parameter(&self, uid: &ParameterUid) -> Arc<SweepParameter> {
        Arc::clone(self.parameters[uid])
    }

    fn get_or_create_ppc_device(
        &mut self,
        signal: &SignalUid,
        device: DeviceUid,
        channel: u16,
    ) -> Arc<PpcDevice> {
        if let Some(ppc_device) = self.ppc_devices.get(signal) {
            Arc::clone(ppc_device)
        } else {
            let device = self.id_store.resolve_unchecked(device).to_string();
            let ppc = Arc::new(PpcDevice { device, channel });
            self.ppc_devices.insert(*signal, Arc::clone(&ppc));
            ppc
        }
    }

    fn resolve_parametrized_phase(&self, param: &ValueOrParameter<f64>) -> f64 {
        match param {
            ValueOrParameter::Value(value) => *value,
            ValueOrParameter::ResolvedParameter { value, .. } => *value,
            ValueOrParameter::Parameter(_) => unimplemented!("Phase parameter not resolved."),
        }
    }

    fn resolve_parametrized_set_oscillator_phase(&self, param: &ValueOrParameter<f64>) -> f64 {
        match param {
            ValueOrParameter::Value(value) => *value,
            ValueOrParameter::ResolvedParameter { value, .. } => *value,
            ValueOrParameter::Parameter(_) => unimplemented!("Phase parameter not resolved."),
        }
    }

    fn resolve_parametrized_amplitude(
        &self,
        param: &ValueOrParameter<ComplexOrFloat>,
    ) -> (Option<Complex<f64>>, Option<ParameterUid>) {
        match param {
            ValueOrParameter::Value(value) => match value {
                ComplexOrFloat::Float(f) => (Some(Complex::new(*f, 0.0)), None),
                ComplexOrFloat::Complex(c) => (Some(*c), None),
            },
            ValueOrParameter::ResolvedParameter { value, uid } => match value {
                ComplexOrFloat::Float(f) => (Some(Complex::new(*f, 0.0)), Some(*uid)),
                ComplexOrFloat::Complex(c) => (Some(*c), Some(*uid)),
            },
            ValueOrParameter::Parameter(_) => unimplemented!("Amplitude parameter not resolved."),
        }
    }

    fn resolve_parametrized_increment_oscillator_phase(
        &self,
        param: &ValueOrParameter<f64>,
    ) -> (f64, Option<ParameterUid>) {
        match param {
            ValueOrParameter::Value(value) => (*value, None),
            ValueOrParameter::ResolvedParameter { value, uid } => (*value, Some(*uid)),
            ValueOrParameter::Parameter(_) => unimplemented!("Amplitude parameter not resolved."),
        }
    }

    fn play_pulse_to_code(
        &mut self,
        obj: &hir::PlayPulse,
        length: TinySamples,
    ) -> Result<PlayPulse> {
        let (amplitude, amplitude_param) = self.resolve_parametrized_amplitude(&obj.amplitude);
        let phase = obj
            .phase
            .map(|value| self.resolve_parametrized_phase(&value))
            .unwrap_or(0.0);
        let (increment_oscillator_phase, incr_phase_param_name) =
            if let Some(param) = &obj.increment_oscillator_phase {
                let (phase, param) = self.resolve_parametrized_increment_oscillator_phase(param);
                (Some(phase), param)
            } else {
                (None, None)
            };
        let id_pulse_params = self
            .pulse_parameter_deduplicator
            .intern(&obj.pulse_parameters, &obj.parameters);
        let data = PlayPulse {
            length: length.value(),
            signal: self.get_signal(&obj.signal),
            amplitude,
            amp_param_name: amplitude_param
                .map(|param| self.id_store.resolve_unchecked(param).to_string()),
            phase,
            pulse_def: Some(self.get_pulse_def(&obj.pulse)),
            set_oscillator_phase: obj
                .set_oscillator_phase
                .as_ref()
                .map(|param| self.resolve_parametrized_set_oscillator_phase(param)),
            increment_oscillator_phase,
            incr_phase_param_name: incr_phase_param_name
                .map(|param| self.id_store.resolve_unchecked(param).to_string()),
            id_pulse_params: Some(PulseParametersId(id_pulse_params)),
            markers: obj
                .markers
                .iter()
                .map(|marker| Marker {
                    marker_selector: match marker.marker_selector {
                        MarkerSelector::M1 => "marker1".to_string(),
                        MarkerSelector::M2 => "marker2".to_string(),
                    },
                    enable: marker.enable,
                    start: marker.start.map(|v| v.value()),
                    length: marker.length.map(|v| v.value()),
                    pulse_id: marker
                        .pulse_id
                        .map(|uid| self.id_store.resolve_unchecked(uid).to_string()),
                })
                .collect(),
        };
        Ok(data)
    }

    fn change_oscillator_phase_to_code(
        &self,
        obj: &hir::ChangeOscillatorPhase,
        length: TinySamples,
    ) -> Result<PlayPulse> {
        let (increment_oscillator_phase, incr_phase_param_name) =
            if let Some(param) = &obj.increment {
                let (phase, param) = self.resolve_parametrized_increment_oscillator_phase(param);
                (Some(phase), param)
            } else {
                (None, None)
            };

        let data = PlayPulse {
            length: length.value(),
            signal: self.get_signal(&obj.signal),
            amplitude: None,
            amp_param_name: None,
            phase: 0.0,
            pulse_def: None,
            set_oscillator_phase: obj
                .set
                .as_ref()
                .map(|param| self.resolve_parametrized_set_oscillator_phase(param)),
            increment_oscillator_phase,
            incr_phase_param_name: incr_phase_param_name
                .map(|param| self.id_store.resolve_unchecked(param).to_string()),
            id_pulse_params: None,
            markers: vec![],
        };
        Ok(data)
    }

    fn acquire_to_code(&mut self, obj: &hir::Acquire) -> Result<AcquirePulse> {
        let id_pulse_parameters = obj
            .pulse_parameters
            .iter()
            .zip(obj.parameters.iter())
            .map(|(pulse_params, play_params)| {
                let uid = self
                    .pulse_parameter_deduplicator
                    .intern(pulse_params, play_params);
                Some(PulseParametersId(uid))
            })
            .collect();
        let mut data = AcquirePulse {
            signal: self.get_signal(&obj.signal),
            length: obj.integration_length.value(),
            handle: self.id_store.resolve_unchecked(obj.handle).into(),
            pulse_defs: obj
                .kernels
                .iter()
                .map(|uid| self.get_pulse_def(uid))
                .collect(),
            id_pulse_params: id_pulse_parameters,
        };
        if data.id_pulse_params.is_empty() {
            data.id_pulse_params = vec![None; obj.kernels.len()];
        }
        Ok(data)
    }

    fn section_to_code(&mut self, obj: &hir::Section, length: TinySamples) -> Result<Section> {
        let section_info = self.get_or_create_section_info(&obj.uid);
        let prng_setup = if let Some(prng_setup) = obj.prng_setup.as_ref() {
            PrngSetup {
                range: prng_setup.range as u16,
                seed: prng_setup.seed,
                section_info: Arc::clone(&section_info),
            }
            .into()
        } else {
            None
        };
        let trigger_output = obj
            .triggers
            .iter()
            .map(|trig| (self.get_signal(&trig.signal), trig.state))
            .collect();

        let section = Section {
            length: length.value(),
            prng_setup,
            trigger_output,
            section_info,
        };
        Ok(section)
    }

    fn loop_to_code(&mut self, obj: &hir::Loop, length: TinySamples) -> Loop {
        let (prng_sample, parameters) = match &obj.kind {
            hir::LoopKind::Sweeping { parameters } => (
                None,
                parameters
                    .iter()
                    .map(|param| self.get_sweep_parameter(param))
                    .collect(),
            ),
            hir::LoopKind::Prng { sample_uid } => (
                Some(self.id_store.resolve_unchecked(*sample_uid).to_string()),
                vec![],
            ),
            _ => (None, vec![]),
        };
        Loop {
            section_info: self.get_or_create_section_info(&obj.uid),
            length: length.value(),
            count: obj.iterations.get() as u64,
            compressed: obj.compressed(),
            prng_sample,
            parameters,
        }
    }

    fn lower_match_to_code(&mut self, obj: &hir::Match, length: TinySamples) -> Match {
        let section_info = self.get_or_create_section_info(&obj.uid);
        let length = length.value();
        match &obj.target {
            hir::MatchTarget::Handle(handle) => {
                let handle = Some(self.id_store.resolve_unchecked(*handle).into());
                let prng_sample = None;
                Match {
                    section_info,
                    length,
                    handle,
                    user_register: None,
                    local: obj.local,
                    prng_sample,
                }
            }
            hir::MatchTarget::SweepParameter(_) => {
                panic!(
                    "Match sections with SweepParameter target should have been erased before code generation."
                )
            }
            hir::MatchTarget::UserRegister(register) => {
                let handle = None;
                let prng_sample = None;
                Match {
                    section_info,
                    length,
                    handle,
                    user_register: Some(*register),
                    local: false,
                    prng_sample,
                }
            }
            hir::MatchTarget::PrngSample(sample) => {
                let prng_sample = Some(self.id_store.resolve_unchecked(*sample).to_string());
                Match {
                    section_info,
                    length,
                    handle: None,
                    user_register: None,
                    local: false,
                    prng_sample,
                }
            }
        }
    }

    fn case_to_code(&mut self, obj: &hir::Case, length: TinySamples) -> Case {
        Case {
            section_info: self.get_or_create_section_info(&obj.uid),
            length: length.value(),
            state: obj.state as u16,
        }
    }

    fn set_osc_frequency_to_code(
        &mut self,
        obj: &hir::SetOscillatorFrequency,
    ) -> SetOscillatorFrequency {
        let out = obj
            .values
            .iter()
            .map(|(sig_uid, freq)| {
                let signal = self.get_signal(sig_uid);
                match freq {
                    ValueOrParameter::Value(v) => SignalFrequency {
                        signal,
                        frequency: *v,
                    },
                    ValueOrParameter::ResolvedParameter { value, .. } => SignalFrequency {
                        signal,
                        frequency: *value,
                    },
                    ValueOrParameter::Parameter(_) => {
                        unimplemented!("Frequency parameter not resolved.")
                    }
                }
            })
            .collect();
        SetOscillatorFrequency::new(out)
    }

    fn initial_osc_frequency_to_code(
        &mut self,
        obj: &hir::InitialOscillatorFrequency,
    ) -> InitialOscillatorFrequency {
        let out = obj
            .values
            .iter()
            .map(|(sig_uid, freq)| {
                let signal = self.get_signal(sig_uid);
                match freq {
                    ValueOrParameter::Value(v) => SignalFrequency {
                        signal,
                        frequency: *v,
                    },
                    ValueOrParameter::ResolvedParameter { value, .. } => SignalFrequency {
                        signal,
                        frequency: *value,
                    },
                    ValueOrParameter::Parameter(_) => {
                        unimplemented!("Frequency parameter not resolved.")
                    }
                }
            })
            .collect();
        InitialOscillatorFrequency::new(out)
    }

    fn reset_osc_phase_to_code(&mut self, signals: &[SignalUid]) -> PhaseReset {
        PhaseReset {
            signals: signals
                .iter()
                .map(|sig_uid| self.get_signal(sig_uid))
                .collect(),
        }
    }

    fn ppc_step_to_code(&mut self, obj: &hir::PpcStep) -> PpcSweepStep {
        let command: SweepCommand = SweepCommand {
            pump_power: obj.pump_power.map(|v| expect_resolved_f64_value(&v)),
            pump_frequency: obj.pump_frequency.map(|v| expect_resolved_f64_value(&v)),
            probe_power: obj.probe_power.map(|v| expect_resolved_f64_value(&v)),
            probe_frequency: obj.probe_frequency.map(|v| expect_resolved_f64_value(&v)),
            cancellation_phase: obj
                .cancellation_phase
                .map(|v| expect_resolved_f64_value(&v)),
            cancellation_attenuation: obj
                .cancellation_attenuation
                .map(|v| expect_resolved_f64_value(&v)),
        };
        PpcSweepStep {
            signal: self.get_signal(&obj.signal),
            length: obj.trigger_duration.value(),
            sweep_command: command,
            ppc_device: self.get_or_create_ppc_device(&obj.signal, obj.device, obj.channel),
        }
    }

    fn delay_to_code(&self, signal: &SignalUid, length: TinySamples) -> Result<PlayPulse> {
        let data = PlayPulse {
            length: length.value(),
            signal: self.get_signal(signal),
            amplitude: None,
            amp_param_name: None,
            phase: 0.0,
            pulse_def: None,
            set_oscillator_phase: None,
            increment_oscillator_phase: None,
            incr_phase_param_name: None,
            id_pulse_params: None,
            markers: vec![],
        };
        Ok(data)
    }
}

fn expect_resolved_f64_value(value: &ValueOrParameter<f64>) -> f64 {
    match value {
        ValueOrParameter::Value(v) => *v,
        ValueOrParameter::ResolvedParameter { value, .. } => *value,
        ValueOrParameter::Parameter(_) => {
            panic!("Expected resolved f64 value, but found unresolved parameter.")
        }
    }
}

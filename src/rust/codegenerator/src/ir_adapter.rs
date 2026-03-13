// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use laboneq_common::device_options::DeviceOptions;
use num_complex::Complex;

use codegenerator_utils::pulse_parameters::PulseParameterDeduplicator;
use laboneq_common::named_id::NamedIdStore;
use laboneq_dsl::types::{
    AcquisitionType as AcquisitionTypeCommon, ComplexOrFloat, DeviceUid, MarkerSelector,
    OscillatorKind as OscillatorKindCommon, ParameterUid, PulseUid, SectionUid, SignalUid,
    SweepParameter as SweepParameterCommon, ValueOrParameter,
};
use laboneq_ir::node::IrNode as HirNode;
use laboneq_ir::signal::{Signal as SignalCommon, SignalKind as SignalKindCommon};
use laboneq_ir::{self as hir, ExperimentIr};

use laboneq_units::tinysample::{TinySamples, tiny_samples};

use crate::Result;
use crate::awg_processor::process_awgs;
use crate::ir::compilation_job::{
    AwgCore, ChannelIndex, Device, Marker, Oscillator, OscillatorKind, PulseDef, Signal,
    SignalKind, SweepParameter, TriggerMode,
};
use crate::ir::experiment::{AcquisitionType, PulseParametersId, SweepCommand};
use crate::ir::{
    AcquirePulse, Case, InitialOscillatorFrequency, IrNode, Loop, LoopIteration, Match, NodeKind,
    PhaseReset, PlayPulse, PpcDevice, PpcSweepStep, PrngSetup, Section, SectionId, SectionInfo,
    SetOscillatorFrequency, SignalFrequency,
};
use crate::utils::length_to_samples;

pub struct CodegenIr {
    pub root: IrNode,
    pub sweep_parameters: Vec<Arc<SweepParameter>>,
    pub pulse_parameters: PulseParameterDeduplicator,
    pub acquisition_type: AcquisitionType,
    pub awg_cores: Vec<AwgCore>,
}

pub struct AwgInfo {
    pub uid: u16,
    pub device: Arc<Device>,
    pub trigger_mode: TriggerMode,
    pub signals: HashSet<SignalUid>,
}

/// Lower a IR into a codegenerator IR.
pub fn ir_to_codegen_ir(
    experiment: &ExperimentIr,
    awg_cores: Vec<AwgInfo>,
    pulse_defs: Vec<&Arc<PulseDef>>,
) -> Result<CodegenIr> {
    let ir_signals = experiment.device_setup.signals();
    let id_store = &experiment.id_store;

    // Device UID - DeviceOptions, sampling_rate, device is SHFQC
    let settings_per_device: HashMap<String, (DeviceOptions, f64, bool)> = experiment
        .device_setup
        .signals()
        .map(|signal| {
            let device = experiment
                .device_setup
                .device_by_uid(&signal.device_uid)
                .unwrap();
            (
                id_store.resolve(signal.device_uid).unwrap().to_string(),
                (
                    device
                        .options()
                        .expect("Expected device options to be set")
                        .clone(),
                    signal.sampling_rate,
                    device.is_shfqc(),
                ),
            )
        })
        .collect();

    let mut signals_by_awg: HashMap<SignalUid, Signal> = ir_signals
        .map(|signal| (signal.uid, signal_to_codegen_signal(signal, id_store)))
        .collect();

    let mut awgs = vec![];
    for awg_info in awg_cores {
        let signals: Vec<Arc<Signal>> = awg_info
            .signals
            .iter()
            .map(|uid| signals_by_awg.remove(uid).unwrap().into())
            .collect();
        let (options, sampling_rate, is_shfqc) = settings_per_device
            .get(&awg_info.device.uid().to_string())
            .unwrap();

        // Store the original channel map for this AWG.
        // This is needed due to the fact that signal channels are modified during the code generation,
        // but we need the original channels for output.
        // TODO: Reverse this and do not modify the signal channels and store QA generator channels in a separate map.
        // The signal channels for QA are modified in `allocate_shfqa_generator_channels()`, which should be refactor
        // to not modify the original signals.
        let channel_map = signals
            .iter()
            .map(|s| {
                (
                    s.uid,
                    s.channels.iter().map(|c| *c as ChannelIndex).collect(),
                )
            })
            .collect();

        let awg = AwgCore::new(
            awg_info.uid,
            signals,
            *sampling_rate,
            awg_info.device,
            Some(awg_info.trigger_mode),
            options.clone(),
            channel_map,
            *is_shfqc,
        );
        awgs.push(awg);
    }
    // We must process the AWGs before building the tree, since
    // the signals are shared in the nodes.
    // TODO: Refactor to avoid this somewhat awkward ordering.
    process_awgs(&mut awgs)?;

    let sweep_parameters = experiment
        .parameters
        .iter()
        .map(|param| Arc::new(transform_parameter_to_code_parameter(param, id_store)))
        .collect::<Vec<Arc<SweepParameter>>>();

    let mut lowerer = IrToCodeIrLowerer::new(
        id_store,
        awgs.iter().flat_map(|awg| &awg.signals).collect(),
        pulse_defs,
        &sweep_parameters,
    );
    let codegen_ir = transform_ir(&mut lowerer, &experiment.root, tiny_samples(0))?;
    let result = CodegenIr {
        root: codegen_ir,
        sweep_parameters,
        pulse_parameters: lowerer.pulse_parameter_deduplicator,
        acquisition_type: convert_acquisition_type(&experiment.acquisition_type),
        awg_cores: awgs,
    };
    Ok(result)
}

fn signal_to_codegen_signal(signal: &SignalCommon, id_store: &NamedIdStore) -> Signal {
    Signal {
        uid: signal.uid,
        kind: match signal.kind {
            SignalKindCommon::Rf => SignalKind::SINGLE,
            SignalKindCommon::Integration => SignalKind::INTEGRATION,
            SignalKindCommon::Iq => SignalKind::IQ,
        },
        channels: signal.channels.iter().map(|c| *c as u8).collect(),
        oscillator: signal.oscillator.as_ref().map(|osc| Oscillator {
            uid: id_store.resolve_unchecked(osc.uid).to_string(),
            kind: match osc.kind {
                OscillatorKindCommon::Software => OscillatorKind::SOFTWARE,
                OscillatorKindCommon::Hardware => OscillatorKind::HARDWARE,
            },
            frequency: osc.frequency.fixed_value(),
        }),
        start_delay: length_to_samples(signal.start_delay.value(), signal.sampling_rate),
        signal_delay: length_to_samples(signal.signal_delay.value(), signal.sampling_rate),
        automute: signal.automute,
    }
}

fn transform_parameter_to_code_parameter(
    parameter: &SweepParameterCommon,
    id_store: &NamedIdStore,
) -> SweepParameter {
    let uid = id_store.resolve_unchecked(parameter.uid);
    SweepParameter {
        uid: uid.to_string(),
        values: Arc::clone(&parameter.values),
    }
}

fn convert_acquisition_type(acquisition_type: &AcquisitionTypeCommon) -> AcquisitionType {
    match acquisition_type {
        AcquisitionTypeCommon::Raw => AcquisitionType::RAW,
        AcquisitionTypeCommon::Integration => AcquisitionType::INTEGRATION,
        AcquisitionTypeCommon::SpectroscopyIq => AcquisitionType::SPECTROSCOPY_IQ,
        AcquisitionTypeCommon::SpectroscopyPsd => AcquisitionType::SPECTROSCOPY_PSD,
        AcquisitionTypeCommon::Spectroscopy => AcquisitionType::SPECTROSCOPY_IQ,
        AcquisitionTypeCommon::Discrimination => AcquisitionType::DISCRIMINATION,
    }
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
    parameters: HashMap<ParameterUid, Arc<SweepParameter>>,
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
        parameters: &[Arc<SweepParameter>],
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
            .iter()
            .map(|param| (id_store.get(&param.uid).unwrap().into(), Arc::clone(param)))
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
        Arc::clone(&self.parameters[uid])
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

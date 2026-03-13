// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

//! Semantic equivalence checking for `Experiment` structs.
//!
//! Two experiments built via different paths (legacy vs Cap'n Proto) will have
//! the same string names but different `SymbolU32` values in their `NamedIdStore`.
//! This module resolves every `NamedId`-typed field to its string before comparing,
//! so structural equivalence is checked by name rather than by interned integer.

use std::collections::{HashMap, HashSet};

use laboneq_common::named_id::NamedIdStore;
use laboneq_dsl::ExperimentNode;
use laboneq_dsl::operation::{Operation, PulseParameterValue};
use laboneq_dsl::types::{
    ComplexOrFloat, ExternalParameterUid, HandleUid, Marker, MatchTarget, NumericLiteral,
    ParameterUid, PulseParameterUid, PulseUid, SectionUid, SignalUid, SweepParameter, Trigger,
    ValueOrParameter,
};
use laboneq_py_utils::pulse::{PulseDef, PulseKind};
use laboneq_units::duration::{Duration, Second};

use numeric_array::NumericArray;

use crate::experiment::Experiment;

/// Compare two `NumericArray` values, treating Integer64 and Float64 as
/// equivalent when all integer values are exactly representable as f64.
fn assert_numeric_array_eq(l: &NumericArray, r: &NumericArray, ctx: &str, label: &str) {
    fn to_f64s(a: &NumericArray) -> Option<Vec<f64>> {
        match a {
            NumericArray::Float64(v) => Some(v.clone()),
            NumericArray::Integer64(v) => Some(v.iter().map(|&x| x as f64).collect()),
            _ => None,
        }
    }
    match (to_f64s(l), to_f64s(r)) {
        (Some(lf), Some(rf)) => {
            assert_eq!(lf, rf, "{label} mismatch in {ctx}");
        }
        _ => {
            assert_eq!(l, r, "{label} mismatch in {ctx}");
        }
    }
}

struct ExperimentComparator<'a> {
    lhs_store: &'a NamedIdStore,
    rhs_store: &'a NamedIdStore,
}

impl ExperimentComparator<'_> {
    fn str_lhs(&self, id: impl Into<laboneq_common::named_id::NamedId>) -> &str {
        self.lhs_store.resolve(id).unwrap_or("<unresolved-lhs>")
    }

    fn str_rhs(&self, id: impl Into<laboneq_common::named_id::NamedId>) -> &str {
        self.rhs_store.resolve(id).unwrap_or("<unresolved-rhs>")
    }

    fn cmp_signal(&self, lhs: SignalUid, rhs: SignalUid, ctx: &str) {
        let l = self.str_lhs(lhs);
        let r = self.str_rhs(rhs);
        assert_eq!(l, r, "signal mismatch in {ctx}: {l:?} != {r:?}");
    }

    fn cmp_pulse_uid(&self, lhs: PulseUid, rhs: PulseUid, ctx: &str) {
        let l = self.str_lhs(lhs);
        let r = self.str_rhs(rhs);
        assert_eq!(l, r, "pulse UID mismatch in {ctx}: {l:?} != {r:?}");
    }

    fn cmp_opt_pulse_uid(&self, lhs: Option<PulseUid>, rhs: Option<PulseUid>, ctx: &str) {
        match (lhs, rhs) {
            (None, None) => {}
            (Some(l), Some(r)) => self.cmp_pulse_uid(l, r, ctx),
            _ => panic!("pulse UID optionality mismatch in {ctx}"),
        }
    }

    fn cmp_section_uid(&self, lhs: SectionUid, rhs: SectionUid, ctx: &str) {
        let l = self.str_lhs(lhs);
        let r = self.str_rhs(rhs);
        assert_eq!(l, r, "section UID mismatch in {ctx}: {l:?} != {r:?}");
    }

    fn cmp_param_uid(&self, lhs: ParameterUid, rhs: ParameterUid, ctx: &str) {
        let l = self.str_lhs(lhs);
        let r = self.str_rhs(rhs);
        assert_eq!(l, r, "parameter UID mismatch in {ctx}: {l:?} != {r:?}");
    }

    fn cmp_handle_uid(&self, lhs: HandleUid, rhs: HandleUid, ctx: &str) {
        let l = self.str_lhs(lhs);
        let r = self.str_rhs(rhs);
        assert_eq!(l, r, "handle UID mismatch in {ctx}: {l:?} != {r:?}");
    }

    fn cmp_vop_f64(&self, lhs: &ValueOrParameter<f64>, rhs: &ValueOrParameter<f64>, ctx: &str) {
        match (lhs, rhs) {
            (ValueOrParameter::Value(l), ValueOrParameter::Value(r)) => {
                assert_eq!(l, r, "f64 value mismatch in {ctx}");
            }
            (ValueOrParameter::Parameter(l), ValueOrParameter::Parameter(r)) => {
                self.cmp_param_uid(*l, *r, ctx);
            }
            (
                ValueOrParameter::ResolvedParameter { value: lv, uid: lu },
                ValueOrParameter::ResolvedParameter { value: rv, uid: ru },
            ) => {
                assert_eq!(lv, rv, "resolved f64 value mismatch in {ctx}");
                self.cmp_param_uid(*lu, *ru, ctx);
            }
            _ => panic!("ValueOrParameter<f64> variant mismatch in {ctx}"),
        }
    }

    fn cmp_vop_numeric(
        &self,
        lhs: &ValueOrParameter<NumericLiteral>,
        rhs: &ValueOrParameter<NumericLiteral>,
        ctx: &str,
    ) {
        match (lhs, rhs) {
            (ValueOrParameter::Value(l), ValueOrParameter::Value(r)) => {
                assert_eq!(l, r, "NumericLiteral value mismatch in {ctx}");
            }
            (ValueOrParameter::Parameter(l), ValueOrParameter::Parameter(r)) => {
                self.cmp_param_uid(*l, *r, ctx);
            }
            (
                ValueOrParameter::ResolvedParameter { value: lv, uid: lu },
                ValueOrParameter::ResolvedParameter { value: rv, uid: ru },
            ) => {
                assert_eq!(lv, rv, "resolved NumericLiteral mismatch in {ctx}");
                self.cmp_param_uid(*lu, *ru, ctx);
            }
            _ => panic!("ValueOrParameter<NumericLiteral> variant mismatch in {ctx}"),
        }
    }

    fn cmp_vop_duration(
        &self,
        lhs: &ValueOrParameter<Duration<Second>>,
        rhs: &ValueOrParameter<Duration<Second>>,
        ctx: &str,
    ) {
        match (lhs, rhs) {
            (ValueOrParameter::Value(l), ValueOrParameter::Value(r)) => {
                assert_eq!(l, r, "Duration value mismatch in {ctx}");
            }
            (ValueOrParameter::Parameter(l), ValueOrParameter::Parameter(r)) => {
                self.cmp_param_uid(*l, *r, ctx);
            }
            (
                ValueOrParameter::ResolvedParameter { value: lv, uid: lu },
                ValueOrParameter::ResolvedParameter { value: rv, uid: ru },
            ) => {
                assert_eq!(lv, rv, "resolved Duration mismatch in {ctx}");
                self.cmp_param_uid(*lu, *ru, ctx);
            }
            _ => panic!("ValueOrParameter<Duration> variant mismatch in {ctx}"),
        }
    }

    fn cmp_vop_complex_or_float(
        &self,
        lhs: &ValueOrParameter<ComplexOrFloat>,
        rhs: &ValueOrParameter<ComplexOrFloat>,
        ctx: &str,
    ) {
        match (lhs, rhs) {
            (ValueOrParameter::Value(l), ValueOrParameter::Value(r)) => {
                assert_eq!(l, r, "ComplexOrFloat value mismatch in {ctx}");
            }
            (ValueOrParameter::Parameter(l), ValueOrParameter::Parameter(r)) => {
                self.cmp_param_uid(*l, *r, ctx);
            }
            (
                ValueOrParameter::ResolvedParameter { value: lv, uid: lu },
                ValueOrParameter::ResolvedParameter { value: rv, uid: ru },
            ) => {
                assert_eq!(lv, rv, "resolved ComplexOrFloat mismatch in {ctx}");
                self.cmp_param_uid(*lu, *ru, ctx);
            }
            _ => panic!("ValueOrParameter<ComplexOrFloat> variant mismatch in {ctx}"),
        }
    }

    fn cmp_pulse_param_value(
        &self,
        lhs: &PulseParameterValue,
        rhs: &PulseParameterValue,
        ctx: &str,
    ) {
        match (lhs, rhs) {
            (
                PulseParameterValue::ExternalParameter(l),
                PulseParameterValue::ExternalParameter(r),
            ) => {
                assert_eq!(l, r, "external parameter UID mismatch in {ctx}");
            }
            (
                PulseParameterValue::ValueOrParameter(l),
                PulseParameterValue::ValueOrParameter(r),
            ) => {
                self.cmp_vop_numeric(l, r, ctx);
            }
            _ => panic!("PulseParameterValue variant mismatch in {ctx}"),
        }
    }

    fn cmp_pulse_param_map(
        &self,
        lhs: &HashMap<PulseParameterUid, PulseParameterValue>,
        rhs: &HashMap<PulseParameterUid, PulseParameterValue>,
        ctx: &str,
    ) {
        assert_eq!(
            lhs.len(),
            rhs.len(),
            "pulse parameter map length mismatch in {ctx}: {} vs {}",
            lhs.len(),
            rhs.len()
        );
        let mut lhs_sorted: Vec<(&str, &PulseParameterValue)> =
            lhs.iter().map(|(k, v)| (self.str_lhs(*k), v)).collect();
        let mut rhs_sorted: Vec<(&str, &PulseParameterValue)> =
            rhs.iter().map(|(k, v)| (self.str_rhs(*k), v)).collect();
        lhs_sorted.sort_by_key(|(k, _)| *k);
        rhs_sorted.sort_by_key(|(k, _)| *k);
        for ((lk, lv), (rk, rv)) in lhs_sorted.into_iter().zip(rhs_sorted) {
            assert_eq!(lk, rk, "pulse parameter key mismatch in {ctx}");
            self.cmp_pulse_param_value(lv, rv, &format!("{ctx}/{lk}"));
        }
    }

    fn cmp_pulse_param_sections(
        &self,
        lhs: &[HashMap<PulseParameterUid, PulseParameterValue>],
        rhs: &[HashMap<PulseParameterUid, PulseParameterValue>],
        ctx: &str,
    ) {
        assert_eq!(
            lhs.len(),
            rhs.len(),
            "pulse parameter sections length mismatch in {ctx}"
        );
        for (i, (l, r)) in lhs.iter().zip(rhs.iter()).enumerate() {
            self.cmp_pulse_param_map(l, r, &format!("{ctx}[{i}]"));
        }
    }

    fn cmp_markers(&self, lhs: &[Marker], rhs: &[Marker], ctx: &str) {
        // Filter out no-op markers (disabled with no start, length, or waveform).
        // The capnp path drops these because they are indistinguishable from absent
        // markers in the Cap'n Proto encoding.
        let is_effective = |m: &&Marker| {
            m.enable || m.start.is_some() || m.length.is_some() || m.pulse_id.is_some()
        };
        let lhs_eff: Vec<_> = lhs.iter().filter(is_effective).collect();
        let rhs_eff: Vec<_> = rhs.iter().filter(is_effective).collect();
        assert_eq!(
            lhs_eff.len(),
            rhs_eff.len(),
            "markers length mismatch in {ctx}"
        );
        for (i, (l, r)) in lhs_eff.iter().zip(rhs_eff.iter()).enumerate() {
            let ctx = format!("{ctx}/marker[{i}]");
            assert_eq!(
                l.marker_selector, r.marker_selector,
                "marker selector mismatch in {ctx}"
            );
            assert_eq!(l.enable, r.enable, "marker enable mismatch in {ctx}");
            assert_eq!(l.start, r.start, "marker start mismatch in {ctx}");
            assert_eq!(l.length, r.length, "marker length mismatch in {ctx}");
            self.cmp_opt_pulse_uid(l.pulse_id, r.pulse_id, &ctx);
        }
    }

    fn cmp_triggers(&self, lhs: &[Trigger], rhs: &[Trigger], ctx: &str) {
        assert_eq!(lhs.len(), rhs.len(), "triggers length mismatch in {ctx}");
        for (i, (l, r)) in lhs.iter().zip(rhs.iter()).enumerate() {
            let ctx = format!("{ctx}/trigger[{i}]");
            self.cmp_signal(l.signal, r.signal, &ctx);
            assert_eq!(l.state, r.state, "trigger state mismatch in {ctx}");
        }
    }

    fn cmp_play_after(&self, lhs: &[SectionUid], rhs: &[SectionUid], ctx: &str) {
        assert_eq!(lhs.len(), rhs.len(), "play_after length mismatch in {ctx}");
        let mut ls: Vec<&str> = lhs.iter().map(|s| self.str_lhs(*s)).collect();
        let mut rs: Vec<&str> = rhs.iter().map(|s| self.str_rhs(*s)).collect();
        ls.sort();
        rs.sort();
        assert_eq!(ls, rs, "play_after mismatch in {ctx}");
    }

    fn cmp_match_target(&self, lhs: &MatchTarget, rhs: &MatchTarget, ctx: &str) {
        match (lhs, rhs) {
            (MatchTarget::Handle(l), MatchTarget::Handle(r)) => {
                self.cmp_handle_uid(*l, *r, ctx);
            }
            (MatchTarget::UserRegister(l), MatchTarget::UserRegister(r)) => {
                assert_eq!(l, r, "user register mismatch in {ctx}");
            }
            (MatchTarget::PrngSample(l), MatchTarget::PrngSample(r)) => {
                self.cmp_section_uid(*l, *r, ctx);
            }
            (MatchTarget::SweepParameter(l), MatchTarget::SweepParameter(r)) => {
                self.cmp_param_uid(*l, *r, ctx);
            }
            _ => panic!("MatchTarget variant mismatch in {ctx}"),
        }
    }

    fn compare_operations(&self, lhs: &Operation, rhs: &Operation, ctx: &str) {
        match (lhs, rhs) {
            (Operation::Root, Operation::Root) => {}
            (Operation::RealTimeBoundary, Operation::RealTimeBoundary) => {}
            (Operation::NearTimeCallback, Operation::NearTimeCallback) => {}
            (Operation::SetNode, Operation::SetNode) => {}

            (Operation::Section(l), Operation::Section(r)) => {
                self.cmp_section_uid(l.uid, r.uid, ctx);
                assert_eq!(l.alignment, r.alignment, "alignment mismatch in {ctx}");
                assert_eq!(l.length, r.length, "length mismatch in {ctx}");
                assert_eq!(
                    l.on_system_grid, r.on_system_grid,
                    "on_system_grid mismatch in {ctx}"
                );
                self.cmp_play_after(&l.play_after, &r.play_after, ctx);
                self.cmp_triggers(&l.triggers, &r.triggers, ctx);
            }

            (Operation::PrngSetup(l), Operation::PrngSetup(r)) => {
                self.cmp_section_uid(l.uid, r.uid, ctx);
                assert_eq!(l.range, r.range, "PRNG range mismatch in {ctx}");
                assert_eq!(l.seed, r.seed, "PRNG seed mismatch in {ctx}");
            }

            (Operation::PrngLoop(l), Operation::PrngLoop(r)) => {
                self.cmp_section_uid(l.uid, r.uid, ctx);
                assert_eq!(l.count, r.count, "PRNG loop count mismatch in {ctx}");
                let ls = self.str_lhs(l.sample_uid);
                let rs = self.str_rhs(r.sample_uid);
                assert_eq!(
                    ls, rs,
                    "PRNG sample UID mismatch in {ctx}: {ls:?} != {rs:?}"
                );
            }

            (Operation::Reserve(l), Operation::Reserve(r)) => {
                self.cmp_signal(l.signal, r.signal, ctx);
            }

            (Operation::ResetOscillatorPhase(l), Operation::ResetOscillatorPhase(r)) => {
                let mut ls: Vec<&str> = l.signals.iter().map(|s| self.str_lhs(*s)).collect();
                let mut rs: Vec<&str> = r.signals.iter().map(|s| self.str_rhs(*s)).collect();
                ls.sort();
                rs.sort();
                assert_eq!(ls, rs, "reset oscillator phase signals mismatch in {ctx}");
            }

            (Operation::Sweep(l), Operation::Sweep(r)) => {
                self.cmp_section_uid(l.uid, r.uid, ctx);
                assert_eq!(l.count, r.count, "sweep count mismatch in {ctx}");
                assert_eq!(
                    l.alignment, r.alignment,
                    "sweep alignment mismatch in {ctx}"
                );
                assert_eq!(
                    l.reset_oscillator_phase, r.reset_oscillator_phase,
                    "sweep reset_oscillator_phase mismatch in {ctx}"
                );
                assert_eq!(l.chunking, r.chunking, "sweep chunking mismatch in {ctx}");
                let mut lp: Vec<&str> = l.parameters.iter().map(|p| self.str_lhs(*p)).collect();
                let mut rp: Vec<&str> = r.parameters.iter().map(|p| self.str_rhs(*p)).collect();
                lp.sort();
                rp.sort();
                assert_eq!(lp, rp, "sweep parameters mismatch in {ctx}");
            }

            (Operation::AveragingLoop(l), Operation::AveragingLoop(r)) => {
                self.cmp_section_uid(l.uid, r.uid, ctx);
                assert_eq!(l.count, r.count, "averaging loop count mismatch in {ctx}");
                assert_eq!(
                    l.acquisition_type, r.acquisition_type,
                    "acquisition type mismatch in {ctx}"
                );
                assert_eq!(
                    l.averaging_mode, r.averaging_mode,
                    "averaging mode mismatch in {ctx}"
                );
                assert_eq!(
                    l.repetition_mode, r.repetition_mode,
                    "repetition mode mismatch in {ctx}"
                );
                assert_eq!(
                    l.reset_oscillator_phase, r.reset_oscillator_phase,
                    "averaging loop reset_oscillator_phase mismatch in {ctx}"
                );
                assert_eq!(
                    l.alignment, r.alignment,
                    "averaging loop alignment mismatch in {ctx}"
                );
            }

            (Operation::Match(l), Operation::Match(r)) => {
                self.cmp_section_uid(l.uid, r.uid, ctx);
                self.cmp_match_target(&l.target, &r.target, ctx);
                assert_eq!(l.local, r.local, "match local mismatch in {ctx}");
                self.cmp_play_after(&l.play_after, &r.play_after, ctx);
            }

            (Operation::Case(l), Operation::Case(r)) => {
                self.cmp_section_uid(l.uid, r.uid, ctx);
                assert_eq!(l.state, r.state, "case state mismatch in {ctx}");
            }

            (Operation::Delay(l), Operation::Delay(r)) => {
                self.cmp_signal(l.signal, r.signal, ctx);
                self.cmp_vop_duration(&l.time, &r.time, ctx);
                assert_eq!(
                    l.precompensation_clear, r.precompensation_clear,
                    "precompensation_clear mismatch in {ctx}"
                );
            }

            (Operation::PlayPulse(l), Operation::PlayPulse(r)) => {
                self.cmp_signal(l.signal, r.signal, ctx);
                self.cmp_opt_pulse_uid(l.pulse, r.pulse, ctx);
                self.cmp_vop_complex_or_float(&l.amplitude, &r.amplitude, ctx);
                match (&l.phase, &r.phase) {
                    (None, None) => {}
                    (Some(lp), Some(rp)) => self.cmp_vop_f64(lp, rp, ctx),
                    _ => panic!("phase optionality mismatch in {ctx}"),
                }
                match (&l.increment_oscillator_phase, &r.increment_oscillator_phase) {
                    (None, None) => {}
                    (Some(lp), Some(rp)) => self.cmp_vop_f64(lp, rp, ctx),
                    _ => panic!("increment_oscillator_phase optionality mismatch in {ctx}"),
                }
                match (&l.set_oscillator_phase, &r.set_oscillator_phase) {
                    (None, None) => {}
                    (Some(lp), Some(rp)) => self.cmp_vop_f64(lp, rp, ctx),
                    _ => panic!("set_oscillator_phase optionality mismatch in {ctx}"),
                }
                match (&l.length, &r.length) {
                    (None, None) => {}
                    (Some(lp), Some(rp)) => self.cmp_vop_duration(lp, rp, ctx),
                    _ => panic!("play_pulse length optionality mismatch in {ctx}"),
                }
                self.cmp_pulse_param_map(&l.parameters, &r.parameters, ctx);
                self.cmp_pulse_param_map(&l.pulse_parameters, &r.pulse_parameters, ctx);
                self.cmp_markers(&l.markers, &r.markers, ctx);
            }

            (Operation::Acquire(l), Operation::Acquire(r)) => {
                self.cmp_signal(l.signal, r.signal, ctx);
                self.cmp_handle_uid(l.handle, r.handle, ctx);
                assert_eq!(l.length, r.length, "acquire length mismatch in {ctx}");
                assert_eq!(
                    l.kernel.len(),
                    r.kernel.len(),
                    "acquire kernel length mismatch in {ctx}"
                );
                for (i, (lk, rk)) in l.kernel.iter().zip(r.kernel.iter()).enumerate() {
                    self.cmp_pulse_uid(*lk, *rk, &format!("{ctx}/kernel[{i}]"));
                }
                self.cmp_pulse_param_sections(&l.parameters, &r.parameters, ctx);
                self.cmp_pulse_param_sections(&l.pulse_parameters, &r.pulse_parameters, ctx);
            }

            _ => panic!(
                "Operation variant mismatch in {ctx}: lhs={} rhs={}",
                operation_variant_name(lhs),
                operation_variant_name(rhs),
            ),
        }
    }

    fn compare_nodes(&self, lhs: &ExperimentNode, rhs: &ExperimentNode) {
        let ctx = operation_section_name(&lhs.kind, self.lhs_store).to_string();
        self.compare_operations(&lhs.kind, &rhs.kind, &ctx);
        assert_eq!(
            lhs.children.len(),
            rhs.children.len(),
            "children count mismatch at {ctx}: {} vs {}",
            lhs.children.len(),
            rhs.children.len()
        );
        for (lc, rc) in lhs.children.iter().zip(rhs.children.iter()) {
            self.compare_nodes(lc, rc);
        }
    }

    fn compare_parameters_map(
        &self,
        lhs: &HashMap<ParameterUid, SweepParameter>,
        rhs: &HashMap<ParameterUid, SweepParameter>,
    ) {
        assert_eq!(
            lhs.len(),
            rhs.len(),
            "parameters map length mismatch: {} vs {}",
            lhs.len(),
            rhs.len()
        );
        let mut lhs_sorted: Vec<(&str, &SweepParameter)> =
            lhs.iter().map(|(k, v)| (self.str_lhs(*k), v)).collect();
        let mut rhs_sorted: Vec<(&str, &SweepParameter)> =
            rhs.iter().map(|(k, v)| (self.str_rhs(*k), v)).collect();
        lhs_sorted.sort_by_key(|(k, _)| *k);
        rhs_sorted.sort_by_key(|(k, _)| *k);
        for ((lk, lv), (rk, rv)) in lhs_sorted.into_iter().zip(rhs_sorted) {
            assert_eq!(lk, rk, "parameter key mismatch: {lk:?} vs {rk:?}");
            assert_eq!(lv.values, rv.values, "parameter values mismatch for {lk:?}");
        }
    }

    fn compare_pulse_kind(&self, lhs: &PulseKind, rhs: &PulseKind, ctx: &str) {
        match (lhs, rhs) {
            (PulseKind::Functional(l), PulseKind::Functional(r)) => {
                assert_eq!(
                    l.length, r.length,
                    "functional pulse length mismatch in {ctx}"
                );
                assert_eq!(
                    l.function, r.function,
                    "functional pulse function mismatch in {ctx}"
                );
            }
            (PulseKind::Sampled(l), PulseKind::Sampled(r)) => {
                assert_numeric_array_eq(&l.samples, &r.samples, ctx, "sampled pulse samples");
            }
            (PulseKind::LengthOnly { length: l }, PulseKind::LengthOnly { length: r }) => {
                assert_eq!(l, r, "length-only pulse length mismatch in {ctx}");
            }
            (PulseKind::MarkerPulse { length: l }, PulseKind::MarkerPulse { length: r }) => {
                assert_eq!(l, r, "marker pulse length mismatch in {ctx}");
            }
            _ => panic!("PulseKind variant mismatch in {ctx}"),
        }
    }

    fn compare_pulses_map(
        &self,
        lhs: &HashMap<PulseUid, PulseDef>,
        rhs: &HashMap<PulseUid, PulseDef>,
    ) {
        assert_eq!(
            lhs.len(),
            rhs.len(),
            "pulses map length mismatch: {} vs {}",
            lhs.len(),
            rhs.len()
        );
        let mut lhs_sorted: Vec<(&str, &PulseDef)> =
            lhs.iter().map(|(k, v)| (self.str_lhs(*k), v)).collect();
        let mut rhs_sorted: Vec<(&str, &PulseDef)> =
            rhs.iter().map(|(k, v)| (self.str_rhs(*k), v)).collect();
        lhs_sorted.sort_by_key(|(k, _)| *k);
        rhs_sorted.sort_by_key(|(k, _)| *k);
        for ((lk, lv), (rk, rv)) in lhs_sorted.into_iter().zip(rhs_sorted) {
            assert_eq!(lk, rk, "pulse key mismatch: {lk:?} vs {rk:?}");
            self.cmp_pulse_uid(lv.uid, rv.uid, lk);
            assert_eq!(
                lv.can_compress, rv.can_compress,
                "pulse can_compress mismatch for {lk}"
            );
            assert_eq!(
                lv.amplitude, rv.amplitude,
                "pulse amplitude mismatch for {lk}"
            );
            self.compare_pulse_kind(&lv.kind, &rv.kind, lk);
        }
    }
}

fn operation_variant_name(op: &Operation) -> &'static str {
    match op {
        Operation::Root => "Root",
        Operation::Section(_) => "Section",
        Operation::PrngSetup(_) => "PrngSetup",
        Operation::PrngLoop(_) => "PrngLoop",
        Operation::Reserve(_) => "Reserve",
        Operation::Sweep(_) => "Sweep",
        Operation::PlayPulse(_) => "PlayPulse",
        Operation::Acquire(_) => "Acquire",
        Operation::Delay(_) => "Delay",
        Operation::AveragingLoop(_) => "AveragingLoop",
        Operation::RealTimeBoundary => "RealTimeBoundary",
        Operation::Match(_) => "Match",
        Operation::ResetOscillatorPhase(_) => "ResetOscillatorPhase",
        Operation::Case(_) => "Case",
        Operation::NearTimeCallback => "NearTimeCallback",
        Operation::SetNode => "SetNode",
    }
}

fn operation_section_name(op: &Operation, store: &NamedIdStore) -> String {
    let resolve = |uid: laboneq_common::named_id::NamedId| {
        store.resolve(uid).unwrap_or("<unknown>").to_string()
    };
    match op {
        Operation::Root => "Root".to_string(),
        Operation::Section(s) => format!("Section({})", resolve(s.uid.0)),
        Operation::PrngSetup(s) => format!("PrngSetup({})", resolve(s.uid.0)),
        Operation::PrngLoop(s) => format!("PrngLoop({})", resolve(s.uid.0)),
        Operation::Sweep(s) => format!("Sweep({})", resolve(s.uid.0)),
        Operation::AveragingLoop(s) => format!("AveragingLoop({})", resolve(s.uid.0)),
        Operation::Match(s) => format!("Match({})", resolve(s.uid.0)),
        Operation::Case(s) => format!("Case({})", resolve(s.uid.0)),
        other => operation_variant_name(other).to_string(),
    }
}

/// Assert that two built experiments are semantically equivalent.
///
/// Resolves all `NamedId`-typed fields to strings via each experiment's own
/// `NamedIdStore` before comparing, so different interning orders don't cause
/// false failures.
///
/// Panics with a descriptive message on the first detected mismatch.
pub(crate) fn assert_experiment_equivalent(lhs: &Experiment, rhs: &Experiment) {
    let cmp = ExperimentComparator {
        lhs_store: &lhs.id_store,
        rhs_store: &rhs.id_store,
    };
    cmp.compare_nodes(&lhs.root, &rhs.root);
    cmp.compare_parameters_map(&lhs.parameters, &rhs.parameters);
    cmp.compare_pulses_map(&lhs.pulses, &rhs.pulses);

    // For py_object_store: compare only the set of ExternalParameterUid keys.
    // ExternalParameterUid is content-addressed (sha256 of pickled object), not position-based,
    // so keys are directly comparable across the two experiments.
    let lhs_keys: HashSet<ExternalParameterUid> = lhs.py_object_store.keys().collect();
    let rhs_keys: HashSet<ExternalParameterUid> = rhs.py_object_store.keys().collect();
    assert_eq!(
        lhs_keys,
        rhs_keys,
        "py_object_store keys mismatch: lhs has {} entries, rhs has {} entries",
        lhs_keys.len(),
        rhs_keys.len()
    );
}

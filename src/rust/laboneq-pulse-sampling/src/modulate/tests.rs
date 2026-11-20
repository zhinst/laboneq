// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::f64::consts::PI;

use num_complex::Complex64;

use crate::samples::Samples;

use super::*;

#[test]
fn test_pulse_is_added_into_the_samples() {
    let mut samples = [1.0, 1.0, 1.0];
    modulate_envelope(
        Samples::Real(&mut samples),
        Samples::Real(&[0.5, -0.5, 0.25]),
        Complex64::new(0.8, 0.0),
        &Carriers::new(2e9).modulation(0.0, 0.0),
        &mut Recorder::default(),
    );
    assert_eq!(samples, [1.4, 0.6, 1.2]);
}

#[test]
fn test_envelope_longer_than_the_samples_is_cut_short() {
    let mut samples = [Complex64::ZERO; 2];
    modulate_envelope(
        Samples::Complex(&mut samples),
        Samples::Real(&[1.0; 4]),
        Complex64::new(0.0, 0.5),
        &Carriers::new(2e9).modulation(0.0, 0.0),
        &mut Recorder::default(),
    );
    assert_eq!(samples, [Complex64::new(0.0, 0.5); 2]);
}

#[test]
fn test_phase_rotates_clockwise() {
    let mut samples = [Complex64::ZERO];
    modulate_envelope(
        Samples::Complex(&mut samples),
        Samples::Real(&[1.0]),
        Complex64::new(1.0, 0.0),
        &Carriers::new(2e9).modulation(PI / 2.0, 0.0),
        &mut Recorder::default(),
    );
    assert!((samples[0] - Complex64::new(0.0, -1.0)).norm() < 1e-15);
}

#[test]
fn test_modulation_advances_the_phase_per_sample() {
    // A quarter turn per sample.
    let mut samples = [Complex64::ZERO; 3];
    modulate_envelope(
        Samples::Complex(&mut samples),
        Samples::Real(&[1.0; 3]),
        Complex64::new(1.0, 0.0),
        &Carriers::new(2e9).modulation(0.0, 0.5e9),
        &mut Recorder::default(),
    );
    let expected = [
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, -1.0),
        Complex64::new(-1.0, 0.0),
    ];
    for (sample, expected) in samples.iter().zip(expected) {
        assert!((sample - expected).norm() < 1e-15, "{sample} != {expected}");
    }
}

#[test]
fn test_real_samples_report_a_dropped_imaginary_part() {
    let mut samples = [0.0];
    let dropped = fold::fold_pulse(
        Samples::Real(&mut samples),
        Samples::Complex(&[Complex64::new(0.6, 0.8)]),
        Complex64::new(1.0, 0.0),
        &Carriers::new(2e9).modulation(0.0, 0.0),
        &mut Recorder::default(),
    );
    assert_eq!(samples, [0.6]);
    assert!(dropped);

    let dropped = fold::fold_pulse(
        Samples::Real(&mut samples),
        Samples::Complex(&[Complex64::new(0.6, 1e-9)]),
        Complex64::new(1.0, 0.0),
        &Carriers::new(2e9).modulation(0.0, 0.0),
        &mut Recorder::default(),
    );
    assert!(!dropped);
}

#[test]
fn test_real_samples_keep_the_real_part_of_a_modulated_pulse() {
    // A quarter turn per sample.
    let mut samples = [0.0; 2];
    let dropped = fold::fold_pulse(
        Samples::Real(&mut samples),
        Samples::Real(&[1.0; 2]),
        Complex64::new(1.0, 0.0),
        &Carriers::new(2e9).modulation(0.0, 0.5e9),
        &mut Recorder::default(),
    );
    for (sample, expected) in samples.iter().zip([1.0, 0.0]) {
        assert!((sample - expected).abs() < 1e-15, "{sample} != {expected}");
    }
    assert!(dropped);
}

#[test]
fn test_real_samples_keep_the_real_part_of_a_complex_amplitude() {
    let mut samples = [0.0];
    let dropped = fold::fold_pulse(
        Samples::Real(&mut samples),
        Samples::Real(&[0.5]),
        Complex64::new(0.6, 0.8),
        &Carriers::new(2e9).modulation(0.0, 0.0),
        &mut Recorder::default(),
    );
    assert_eq!(samples, [0.3]);
    assert!(dropped);
}

#[test]
fn test_complex_pulse_samples_are_checked_before_accumulation() {
    let mut samples = [Complex64::new(0.9, 0.0)];
    let mut recorder = Recorder::default();
    modulate_envelope(
        Samples::Complex(&mut samples),
        Samples::Real(&[1.0]),
        Complex64::new(0.6, 0.8),
        &Carriers::new(2e9).modulation(0.0, 0.0),
        &mut recorder,
    );
    assert_eq!(recorder.complex, [Complex64::new(0.6, 0.8)]);
    assert_eq!(samples, [Complex64::new(1.5, 0.8)]);
}

#[test]
fn test_real_pulse_samples_are_checked_before_accumulation() {
    let mut samples = [0.25];
    let mut recorder = Recorder::default();
    // Real amplitude without modulation, then a complex amplitude whose imaginary part is dropped.
    for amplitude in [Complex64::new(0.5, 0.0), Complex64::new(0.5, 0.25)] {
        modulate_envelope(
            Samples::Real(&mut samples),
            Samples::Real(&[1.0]),
            amplitude,
            &Carriers::new(2e9).modulation(0.0, 0.0),
            &mut recorder,
        );
    }
    assert_eq!(recorder.real, [0.5, 0.5]);
    assert!(recorder.complex.is_empty());
    assert_eq!(samples, [1.25]);
}

#[test]
#[should_panic(expected = "Found a pulse envelope with 1 samples, but 2 are played")]
fn test_envelope_shorter_than_the_samples_panics() {
    modulate_envelope(
        Samples::Real(&mut [0.0; 2]),
        Samples::Real(&[1.0]),
        Complex64::new(1.0, 0.0),
        &Carriers::new(2e9).modulation(0.0, 0.0),
        &mut Recorder::default(),
    );
}

#[derive(Default)]
struct Recorder {
    real: Vec<f64>,
    complex: Vec<Complex64>,
}

impl SampleCheck for Recorder {
    fn real(&mut self, value: f64) {
        self.real.push(value);
    }

    fn complex(&mut self, value: Complex64) {
        self.complex.push(value);
    }
}

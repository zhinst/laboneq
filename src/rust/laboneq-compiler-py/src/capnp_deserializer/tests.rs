// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use super::*;

const PARAM: u32 = 0;
const DERIVED_PARAM: u32 = 1;

#[test]
fn test_sweep_contains_derived_parameters() {
    let parameters = [
        Parameter::new("param", &[1.0, 2.0]),
        Parameter::new("derived_param", &[2.0, 4.0]).driven_by(&[PARAM]),
    ];
    let experiment = deserialize_sweep(&parameters, |operation| {
        let mut delay = operation.init_delay();
        delay.set_signal(0);
        delay.init_time().set_parameter_ref(DERIVED_PARAM);
    });

    let Operation::Sweep(sweep) = &sweep_node(&experiment).kind else {
        unreachable!()
    };
    let param = parameter_uid(&experiment, "param");
    let derived = parameter_uid(&experiment, "derived_param");
    assert_eq!(sweep.parameters, [param, derived]);
    assert_eq!(sweep.direct_parameters, [param]);
}

#[test]
fn test_play_keeps_pulse_parameter_references_and_values() {
    let parameters = [Parameter::new("param", &[1.0, 2.0])];
    let experiment = deserialize_sweep(&parameters, |operation| {
        let mut play = operation.init_play();
        play.set_signal(0);
        let mut entries = play.init_pulse_parameters(2);
        let mut sigma = entries.reborrow().get(0);
        sigma.set_key("sigma");
        sigma.init_value().set_parameter_ref(PARAM);
        let mut beta = entries.get(1);
        beta.set_key("beta");
        beta.init_value().init_constant().set_real(0.5);
    });

    let [play_node] = sweep_node(&experiment).children.as_slice() else {
        panic!("expected a single operation in the sweep");
    };
    let Operation::PlayPulse(play) = &play_node.kind else {
        panic!("expected a play, got {:?}", play_node.kind);
    };
    let pulse_parameter = |name| {
        let uid = PulseParameterUid(experiment.id_store.get(name).unwrap());
        &play.parameters[&uid]
    };
    assert_eq!(
        pulse_parameter("sigma"),
        &ExternalOrValue::ValueOrParameter(ValueOrParameter::Parameter(parameter_uid(
            &experiment,
            "param"
        )))
    );
    assert_eq!(
        pulse_parameter("beta"),
        &ExternalOrValue::ValueOrParameter(ValueOrParameter::Value(NumericLiteral::Float(0.5)))
    );
}

/// Deserializes an experiment laid out as:
///
/// ```text
/// signals:          [q0/drive]
/// sweep parameters: `parameters`
/// root
/// └── sweep over the first sweep parameter
///     └── the operation `write_operation` writes
/// ```
fn deserialize_sweep(
    parameters: &[Parameter<'_>],
    write_operation: impl FnOnce(operation_capnp::operation::Builder<'_>),
) -> DeserializedExperiment {
    let mut message = capnp::message::Builder::new_default();
    let mut experiment = message.init_root::<experiment_capnp::experiment::Builder<'_>>();
    experiment
        .reborrow()
        .init_metadata()
        .set_schema_version(experiment_capnp::SCHEMA_VERSION);
    experiment
        .reborrow()
        .init_signals(1)
        .get(0)
        .set_uid("q0/drive");

    let mut parameter_list = experiment
        .reborrow()
        .init_sweep_parameters(parameters.len() as u32);
    for (i, parameter) in parameters.iter().enumerate() {
        parameter.write(parameter_list.reborrow().get(i as u32));
    }

    let mut section = experiment
        .init_root_section()
        .init_content_items(1)
        .get(0)
        .init_section();
    section.set_name("sweep");
    let mut sweep = section.reborrow().init_sweep();
    sweep.set_chunk_count(1);
    sweep.init_parameters(1).set(0, 0);
    write_operation(section.init_content_items(1).get(0).init_operation());

    let bytes = capnp::serialize::write_message_to_words(&message);
    deserialize_experiment(&bytes, false).unwrap()
}

struct Parameter<'a> {
    uid: &'a str,
    values: &'a [f64],
    driven_by: &'a [u32],
}

impl<'a> Parameter<'a> {
    fn new(uid: &'a str, values: &'a [f64]) -> Self {
        Self {
            uid,
            values,
            driven_by: &[],
        }
    }

    fn driven_by(self, driven_by: &'a [u32]) -> Self {
        Self { driven_by, ..self }
    }

    fn write(&self, mut builder: sweep_capnp::sweep_parameter::Builder<'_>) {
        builder.set_uid(self.uid);
        let mut explicit = builder.init_explicit_values();
        explicit.set_real_values(self.values).unwrap();
        explicit.set_driven_by(self.driven_by).unwrap();
    }
}

fn sweep_node(experiment: &DeserializedExperiment) -> &ExperimentNode {
    let [sweep] = experiment.root.children.as_slice() else {
        panic!("expected a single section at the root");
    };
    sweep
}

fn parameter_uid(experiment: &DeserializedExperiment, name: &str) -> ParameterUid {
    ParameterUid(experiment.id_store.get(name).unwrap())
}

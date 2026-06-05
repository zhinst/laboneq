// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::collections::HashSet;

use laboneq_compiler_py::compiler_backend::ParameterValues;
use laboneq_dsl::device_setup::InstrumentKind;
use laboneq_dsl::types::{DeviceUid, ParameterUid, SignalUid};
use laboneq_error::WithContext;
use laboneq_error::{bail, laboneq_error};

use crate::Result;
use crate::experiment_view::{ExperimentSignal, ExperimentViewWrapper};
use crate::ports::{is_shfsg_port, parse_port};

/// Delay in samples introduced by output routing on SHFSG devices
const OUTPUT_ROUTING_DELAY_SAMPLES: i64 = 52;
/// Maximum number of output routes that can be routed to a single channel on SHFSG devices (RTR option)
const MAX_INBOUND_ROUTES_PER_CHANNEL: u8 = 3;
const OUTPUT_ROUTE_AMPLITUDE_RANGE: std::ops::RangeInclusive<f64> = 0.0..=1.0;

#[derive(Debug)]
pub(crate) struct ProcessedOutputRouting {
    /// Maps source channel name to target channel number for all channels involved in output routing.
    pub channel_map: HashMap<String, u8>,
    /// Maps signal uid to the delay in samples that should be applied to the signal to compensate for the delay introduced by output routing.
    pub delay_signal: HashMap<SignalUid, i64>,
}

/// Process the output routing configuration for the given experiment, validating the configuration and calculating the necessary delays for signals with output routing applied.
pub(crate) fn process_output_routing(
    experiment: &ExperimentViewWrapper,
) -> Result<ProcessedOutputRouting> {
    process_output_routing_impl(&experiment.signals, |uid| {
        experiment.get_parameter_values(uid)
    })
}

fn process_output_routing_impl<'s, F>(
    signals: &'s [ExperimentSignal],
    get_params: F,
) -> Result<ProcessedOutputRouting>
where
    F: Fn(ParameterUid) -> Result<&'s ParameterValues>,
{
    let mut inbound_route_count_per_channel: HashMap<(DeviceUid, &str), u8> = HashMap::new();
    let mut channel_requires_routing_delay: HashSet<(DeviceUid, &str)> = HashSet::new();
    let mut routed_output_channel_map = HashMap::new();

    for signal in signals
        .iter()
        .filter(|s| !s.calibration.added_outputs.is_empty())
    {
        let result: Result<()> = (|| {
            let added_outputs = &signal.calibration.added_outputs;

            if !signal.ports.iter().any(is_shfsg_port) {
                bail!("Output routing can only be applied to SGCHANNELS.");
            }

            let target_port = signal
                .ports
                .first()
                .expect("Expected device to have at least one port")
                .path
                .as_str();

            let channel_key_target = (signal.device_uid, target_port);
            let mut source_channels = HashSet::new();
            for source in added_outputs {
                {
                    let source_channel = source.source_channel.as_str();
                    let source_port =
                        parse_port(source_channel, InstrumentKind::Shfsg).map_err(|e| {
                            laboneq_error!(
                                "Output routing can only be applied to SGCHANNELS: {}",
                                e
                            )
                        })?;

                    if !source_channels.insert(source_channel) {
                        bail!(
                            "Duplicate output routing entries for signal '{}'",
                            signal.uid.0
                        );
                    }
                    if source_channel == target_port {
                        bail!(
                            "Output routing source is the same as the target channel on signal '{}'",
                            signal.uid.0
                        );
                    }

                    routed_output_channel_map
                        .insert(source_channel.to_string(), source_port.channel);
                    let channel_key_source = (signal.device_uid, source_channel);
                    channel_requires_routing_delay.insert(channel_key_source);
                    channel_requires_routing_delay.insert(channel_key_target);
                    // Count the number of output routes per source and target channel to enforce the inbound limits
                    let routes_per_target = inbound_route_count_per_channel
                        .entry(channel_key_target)
                        .or_default();
                    *routes_per_target += 1;
                    if *routes_per_target > MAX_INBOUND_ROUTES_PER_CHANNEL {
                        bail!("Maximum of three signals can be routed per SGCHANNEL");
                    }
                }

                for output in added_outputs {
                    if let Some(amplitude_scaling) = output.amplitude_scaling {
                        if let Some(parameter) = amplitude_scaling.parameter_uid().map(&get_params)
                        {
                            let amplitudes = parameter?.as_f64_slice().ok_or_else(|| {
                            laboneq_error!(
                                "Output route amplitude scaling must be a floating point number for signal '{}'.",
                                signal.uid.0
                            )
                        })?;
                            let (min, max) = find_min_max_f64(amplitudes).ok_or_else(|| {
                                laboneq_error!(
                                    "Amplitude scaling parameter must not be empty for signal '{}'",
                                    signal.uid.0
                                )
                            })?;
                            if !OUTPUT_ROUTE_AMPLITUDE_RANGE.contains(&max) {
                                bail!(
                                    "Output route amplitude scaling expects values in the range [0.0, 1.0] for signal '{}'",
                                    signal.uid.0
                                );
                            }
                            if !OUTPUT_ROUTE_AMPLITUDE_RANGE.contains(&min) {
                                bail!(
                                    "Output route amplitude scaling expects values in the range [0.0, 1.0] for signal '{}'",
                                    signal.uid.0
                                );
                            }
                        } else if let Some(value) = amplitude_scaling.fixed_value()
                            && !OUTPUT_ROUTE_AMPLITUDE_RANGE.contains(&value)
                        {
                            bail!(
                                "Output route amplitude scaling expects values in the range [0.0, 1.0] for signal '{}'",
                                signal.uid.0
                            );
                        }
                    }
                }
            }
            Ok(())
        })();
        result.with_context(|| {
            format!(
                "While processing output routing for signal '{}'",
                signal.uid.0
            )
        })?;
    }

    // Calculate the delays introduced by output routing on SHFSG device.
    //
    // Using output routing will introduce delay on both source and target channels,
    // where the both channels must be on the same device.
    let delay_signals = signals
        .iter()
        .filter_map(|s| {
            let key = (
                s.device_uid,
                s.ports
                    .first()
                    .expect("signal has at least one port")
                    .path
                    .as_str(),
            );
            if !channel_requires_routing_delay.contains(&key) {
                return None;
            }
            Some((s.uid, OUTPUT_ROUTING_DELAY_SAMPLES))
        })
        .collect();

    let result = ProcessedOutputRouting {
        channel_map: routed_output_channel_map,
        delay_signal: delay_signals,
    };
    Ok(result)
}

fn find_min_max_f64(vec: &[f64]) -> Option<(f64, f64)> {
    vec.iter().fold(None, |acc, &x| match acc {
        None => Some((x, x)),
        Some((min, max)) => Some((min.min(x), max.max(x))),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use laboneq_compiler_py::compiler_backend::ParameterValues;
    use laboneq_dsl::signal_calibration::{OutputRoute, SignalCalibration};
    use laboneq_dsl::types::ValueOrParameter;

    struct TestFixture {
        signals: Vec<ExperimentSignal>,
        params: HashMap<ParameterUid, ParameterValues>,
    }

    impl TestFixture {
        fn new() -> Self {
            Self {
                signals: Vec::new(),
                params: HashMap::new(),
            }
        }

        fn parameter(&mut self, uid: u32, values: ParameterValues) {
            self.params.insert(uid.into(), values);
        }

        fn signal(&mut self, uid: u32, device_uid: u32, port: &str, routes: Vec<OutputRoute>) {
            let signal = ExperimentSignal {
                uid: uid.into(),
                device_uid: device_uid.into(),
                ports: vec![parse_port(port, InstrumentKind::Shfsg).unwrap()],
                calibration: SignalCalibration {
                    added_outputs: routes,
                    ..Default::default()
                },
            };
            self.signals.push(signal);
        }

        fn run(&self) -> Result<ProcessedOutputRouting> {
            process_output_routing_impl(&self.signals, |uid| {
                self.params
                    .get(&uid)
                    .ok_or_else(|| laboneq_error!("parameter not found"))
            })
        }
    }

    fn route(source: &str) -> OutputRoute {
        OutputRoute {
            source_channel: source.to_string(),
            amplitude_scaling: None,
            phase_shift: None,
        }
    }

    fn route_with_fixed_amplitude(source: &str, amplitude: f64) -> OutputRoute {
        OutputRoute {
            source_channel: source.to_string(),
            amplitude_scaling: Some(ValueOrParameter::Value(amplitude)),
            phase_shift: None,
        }
    }

    fn route_with_param_amplitude(source: &str, param_uid: ParameterUid) -> OutputRoute {
        OutputRoute {
            source_channel: source.to_string(),
            amplitude_scaling: Some(ValueOrParameter::Parameter(param_uid)),
            phase_shift: None,
        }
    }

    #[test]
    fn no_output_routing_returns_empty_maps() {
        let mut fx = TestFixture::new();
        fx.signal(0, 0, "SGCHANNELS/0/OUTPUT", vec![]);

        let result = fx.run().unwrap();
        assert!(result.channel_map.is_empty());
        assert!(result.delay_signal.is_empty());
    }

    #[test]
    fn single_route_populates_channel_map_and_delays() {
        let mut fx = TestFixture::new();
        fx.signal(
            0,
            0,
            "SGCHANNELS/1/OUTPUT",
            vec![route("SGCHANNELS/0/OUTPUT")],
        );
        fx.signal(1, 0, "SGCHANNELS/0/OUTPUT", vec![]);
        let result = fx.run().unwrap();

        assert_eq!(result.channel_map["SGCHANNELS/0/OUTPUT"], 0);
        assert_eq!(result.delay_signal.len(), 2);
        assert_eq!(
            result.delay_signal[&fx.signals[0].uid],
            OUTPUT_ROUTING_DELAY_SAMPLES
        );
        assert_eq!(
            result.delay_signal[&fx.signals[1].uid],
            OUTPUT_ROUTING_DELAY_SAMPLES
        );
    }

    #[test]
    fn unrelated_signal_does_not_get_delay() {
        let mut fx = TestFixture::new();
        fx.signal(
            0,
            0,
            "SGCHANNELS/1/OUTPUT",
            vec![route("SGCHANNELS/0/OUTPUT")],
        );
        fx.signal(1, 0, "SGCHANNELS/0/OUTPUT", vec![]);
        fx.signal(2, 0, "SGCHANNELS/2/OUTPUT", vec![]);
        let result = fx.run().unwrap();
        assert!(!result.delay_signal.contains_key(&fx.signals[2].uid));
    }

    #[test]
    fn three_routes_to_same_target_is_valid() {
        let mut fx = TestFixture::new();
        fx.signal(
            0,
            0,
            "SGCHANNELS/3/OUTPUT",
            vec![
                route("SGCHANNELS/0/OUTPUT"),
                route("SGCHANNELS/1/OUTPUT"),
                route("SGCHANNELS/2/OUTPUT"),
            ],
        );
        assert!(fx.run().is_ok());
    }

    #[test]
    fn four_routes_to_same_target_is_an_error() {
        let mut fx = TestFixture::new();
        fx.signal(
            0,
            0,
            "SGCHANNELS/4/OUTPUT",
            vec![
                route("SGCHANNELS/0/OUTPUT"),
                route("SGCHANNELS/1/OUTPUT"),
                route("SGCHANNELS/2/OUTPUT"),
                route("SGCHANNELS/3/OUTPUT"),
            ],
        );
        let err = fx.run().unwrap_err().to_string();
        assert!(err.contains("Maximum of three"));
    }

    #[test]
    fn invalid_source_port_is_an_error() {
        let mut fx = TestFixture::new();
        fx.signal(0, 0, "SGCHANNELS/0/OUTPUT", vec![route("NOT_A_PORT")]);
        assert!(fx.run().is_err());
    }

    #[test]
    fn duplicate_source_channel_is_an_error() {
        let mut fx = TestFixture::new();
        fx.signal(
            0,
            0,
            "SGCHANNELS/1/OUTPUT",
            vec![route("SGCHANNELS/0/OUTPUT"), route("SGCHANNELS/0/OUTPUT")],
        );
        let err = fx.run().unwrap_err().to_string();
        assert!(err.contains("Duplicate"));
    }

    #[test]
    fn source_same_as_target_is_an_error() {
        let mut fx = TestFixture::new();
        fx.signal(
            0,
            0,
            "SGCHANNELS/0/OUTPUT",
            vec![route("SGCHANNELS/0/OUTPUT")],
        );
        let err = fx.run().unwrap_err().to_string();
        assert!(err.contains("same as the target"));
    }

    #[test]
    fn fixed_amplitude_at_boundary_values_is_valid() {
        let mut fx = TestFixture::new();
        fx.signal(
            0,
            0,
            "SGCHANNELS/1/OUTPUT",
            vec![
                route_with_fixed_amplitude("SGCHANNELS/0/OUTPUT", 0.0),
                route_with_fixed_amplitude("SGCHANNELS/2/OUTPUT", 1.0),
            ],
        );
        assert!(fx.run().is_ok());
    }

    #[test]
    fn fixed_amplitude_above_one_is_an_error() {
        let mut fx = TestFixture::new();
        fx.signal(
            0,
            0,
            "SGCHANNELS/1/OUTPUT",
            vec![route_with_fixed_amplitude("SGCHANNELS/0/OUTPUT", 1.01)],
        );
        let err = fx.run().unwrap_err().to_string();
        assert!(err.contains("[0.0, 1.0]"));
    }

    #[test]
    fn fixed_amplitude_below_zero_is_an_error() {
        let mut fx = TestFixture::new();
        fx.signal(
            0,
            0,
            "SGCHANNELS/1/OUTPUT",
            vec![route_with_fixed_amplitude("SGCHANNELS/0/OUTPUT", -0.01)],
        );
        let err = fx.run().unwrap_err().to_string();
        assert!(err.contains("[0.0, 1.0]"));
    }

    #[test]
    fn parameter_amplitude_out_of_range_is_an_error() {
        let mut fx = TestFixture::new();
        fx.parameter(0, ParameterValues::Float64(vec![0.5, 1.5]));
        fx.signal(
            0,
            0,
            "SGCHANNELS/1/OUTPUT",
            vec![route_with_param_amplitude("SGCHANNELS/0/OUTPUT", 0.into())],
        );
        let err = fx.run().unwrap_err().to_string();
        assert!(err.contains("[0.0, 1.0]"));
    }

    #[test]
    fn parameter_amplitude_non_float_is_an_error() {
        let mut fx = TestFixture::new();
        fx.parameter(0, ParameterValues::Integer64(vec![1]));
        fx.signal(
            0,
            0,
            "SGCHANNELS/1/OUTPUT",
            vec![route_with_param_amplitude("SGCHANNELS/0/OUTPUT", 0.into())],
        );
        let err = fx.run().unwrap_err().to_string();
        assert!(err.contains("floating point"));
    }

    #[test]
    fn parameter_amplitude_valid_range_is_ok() {
        let mut fx = TestFixture::new();
        fx.parameter(0, ParameterValues::Float64(vec![0.0, 0.5, 1.0]));
        fx.signal(
            0,
            0,
            "SGCHANNELS/1/OUTPUT",
            vec![route_with_param_amplitude("SGCHANNELS/0/OUTPUT", 0.into())],
        );
        assert!(fx.run().is_ok());
    }

    #[test]
    fn parameter_nan_amplitude_is_an_error() {
        let mut fx = TestFixture::new();
        fx.parameter(0, ParameterValues::Float64(vec![f64::NAN]));
        fx.signal(
            0,
            0,
            "SGCHANNELS/1/OUTPUT",
            vec![route_with_param_amplitude("SGCHANNELS/0/OUTPUT", 0.into())],
        );
        let err = fx.run().unwrap_err().to_string();
        assert!(err.contains("expects values in the range [0.0, 1.0]"));
    }
}

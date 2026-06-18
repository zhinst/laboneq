// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::collections::HashSet;

use laboneq_compiler_py::compiler_backend::ParameterValues;
use laboneq_dsl::signal_calibration::OutputRoute;
use laboneq_dsl::types::PhysicalChannelUid;
use laboneq_dsl::types::{DeviceUid, ParameterUid, SignalUid};
use laboneq_error::WithContext;
use laboneq_error::{bail, laboneq_error};

use crate::Result;
use crate::experiment_view::{DeviceChannel, ExperimentViewWrapper};
use crate::ports::{expect_one_port, is_shfsg_port};

/// Delay in samples introduced by output routing on SHFSG devices
const OUTPUT_ROUTING_DELAY_SAMPLES: i64 = 52;
/// Maximum number of output routes that can be routed to a single channel on SHFSG devices (RTR option)
const MAX_INBOUND_ROUTES_PER_CHANNEL: u8 = 3;
const OUTPUT_ROUTE_AMPLITUDE_RANGE: std::ops::RangeInclusive<f64> = 0.0..=1.0;

#[derive(Debug)]
pub(crate) struct ProcessedOutputRouting {
    /// Maps source channel name to target channel number for all channels involved in output routing.
    pub channel_map: HashMap<PhysicalChannelUid, u8>,
    /// Maps signal uid to the delay in samples that should be applied to the signal to compensate for the delay introduced by output routing.
    pub delay_signal: HashMap<SignalUid, i64>,
}

/// Process the output routing configuration for the given experiment, validating the configuration and calculating the necessary delays for signals with output routing applied.
pub(crate) fn process_output_routing(
    experiment: &ExperimentViewWrapper,
) -> Result<ProcessedOutputRouting> {
    let routes: Vec<SgChannel> = experiment
        .signals
        .iter()
        // Validate bad config and filter signals
        .map(|s| -> Result<Option<_>> {
            let is_shfsg = s.ports.iter().any(is_shfsg_port);
            if !s.calibration.added_outputs.is_empty() && !is_shfsg {
                return Err(laboneq_error!(
                    "Output routing can only be applied to SGCHANNELS."
                ));
            }
            Ok(is_shfsg.then_some(s))
        })
        .filter_map(|r| r.transpose())
        // Transform to a more convenient structure for processing the output routing configuration
        .map(|r| -> Result<SgChannel> {
            let s = r?;
            let port = expect_one_port(s.ports.as_slice())?;
            Ok(SgChannel {
                uid: s.uid,
                device_uid: s.device_uid,
                channel: port.channel,
                added_outputs: s.calibration.added_outputs.as_slice(),
            })
        })
        .collect::<Result<Vec<_>>>()?;

    process_output_routing_impl(
        &routes,
        |uid| experiment.get_parameter_values(uid),
        |uid| experiment.get_device_channel(uid),
    )
}

#[derive(Debug)]
struct SgChannel<'a> {
    uid: SignalUid,
    device_uid: DeviceUid,
    channel: u8,
    added_outputs: &'a [OutputRoute],
}

impl SgChannel<'_> {
    fn has_output_routes(&self) -> bool {
        !self.added_outputs.is_empty()
    }

    fn channel_key(&self) -> (DeviceUid, u8) {
        (self.device_uid, self.channel)
    }
}

fn process_output_routing_impl<'s>(
    signals: &[SgChannel],
    get_parameter_values: impl Fn(ParameterUid) -> Result<&'s ParameterValues>,
    get_device_channel: impl Fn(PhysicalChannelUid) -> Result<&'s DeviceChannel>,
) -> Result<ProcessedOutputRouting> {
    let mut inbound_route_count_per_channel: HashMap<(DeviceUid, u8), u8> = HashMap::new();
    let mut channel_requires_routing_delay: HashSet<(DeviceUid, u8)> = HashSet::new();
    let mut routed_output_channel_map = HashMap::new();

    for signal in signals.iter().filter(|s| s.has_output_routes()) {
        let result: Result<()> = (|| {
            let channel_key_target = signal.channel_key();
            let mut source_channels = HashSet::new();

            for source in signal.added_outputs {
                let source_channel = get_device_channel(source.source_channel)?;
                let source_port = expect_one_port(&source_channel.ports)?;

                if !source_channels.insert(source_port.channel) {
                    bail!(
                        "Duplicate output routing entries for signal '{}'",
                        signal.uid.0
                    );
                }
                if source_channel.device_uid != signal.device_uid {
                    bail!(
                        "Output routing can be only applied within the same device SGCHANNELS on signal '{}'",
                        signal.uid.0,
                    );
                }
                if source_port.channel == signal.channel {
                    bail!(
                        "Output routing source is the same as the target channel on signal '{}'",
                        signal.uid.0
                    );
                }
                routed_output_channel_map.insert(source_channel.uid, source_port.channel);
                let channel_key_source = (signal.device_uid, source_port.channel);
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

            for output in signal.added_outputs {
                if let Some(amplitude_scaling) = output.amplitude_scaling {
                    if let Some(parameter) =
                        amplitude_scaling.parameter_uid().map(&get_parameter_values)
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
                        if !OUTPUT_ROUTE_AMPLITUDE_RANGE.contains(&max)
                            || !OUTPUT_ROUTE_AMPLITUDE_RANGE.contains(&min)
                        {
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
    let delay_signal = signals
        .iter()
        .filter(|s| channel_requires_routing_delay.contains(&s.channel_key()))
        .map(|s| (s.uid, OUTPUT_ROUTING_DELAY_SAMPLES))
        .collect();

    Ok(ProcessedOutputRouting {
        channel_map: routed_output_channel_map,
        delay_signal,
    })
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

    use crate::ports::{Port, parse_port};
    use laboneq_compiler_py::compiler_backend::ParameterValues;
    use laboneq_dsl::device_setup::InstrumentKind;
    use laboneq_dsl::signal_calibration::OutputRoute;
    use laboneq_dsl::types::ValueOrParameter;

    struct TestSgChannel {
        uid: SignalUid,
        device_uid: DeviceUid,
        port: Port,
        routes: Vec<OutputRoute>,
    }

    struct TestFixture {
        signals: Vec<TestSgChannel>,
        params: HashMap<ParameterUid, ParameterValues>,
        device_channels: Vec<DeviceChannel>,
    }

    impl TestFixture {
        fn new() -> Self {
            Self {
                signals: Vec::new(),
                params: HashMap::new(),
                device_channels: Vec::new(),
            }
        }

        fn add_parameter(&mut self, uid: u32, values: ParameterValues) {
            self.params.insert(uid.into(), values);
        }

        fn add_device_channel(&mut self, uid: u32, device_uid: u32, port_str: &str) {
            let port = parse_port(port_str, InstrumentKind::Shfsg)
                .unwrap_or_else(|_| panic!("invalid port string '{port_str}' in test fixture"));
            self.device_channels.push(DeviceChannel {
                uid: uid.into(),
                device_uid: device_uid.into(),
                ports: vec![port],
            });
        }

        fn add_signal(
            &mut self,
            uid: u32,
            device_uid: u32,
            target_channel_str: &str,
            routes: Vec<OutputRoute>,
        ) {
            let port = parse_port(target_channel_str, InstrumentKind::Shfsg)
                .unwrap_or_else(|_| panic!("invalid port '{target_channel_str}' in test fixture"));
            self.signals.push(TestSgChannel {
                uid: uid.into(),
                device_uid: device_uid.into(),
                port,
                routes,
            });
        }

        fn run(&self) -> Result<ProcessedOutputRouting> {
            let views: Vec<SgChannel> = self
                .signals
                .iter()
                .map(|s| SgChannel {
                    uid: s.uid,
                    device_uid: s.device_uid,
                    channel: s.port.channel,
                    added_outputs: &s.routes,
                })
                .collect();
            process_output_routing_impl(
                &views,
                |uid| {
                    self.params
                        .get(&uid)
                        .ok_or_else(|| laboneq_error!("parameter not found"))
                },
                |uid| {
                    self.device_channels
                        .iter()
                        .find(|dc| dc.uid == uid)
                        .ok_or_else(|| laboneq_error!("device channel '{}' not found", uid.0))
                },
            )
        }
    }

    fn route(source: u32) -> OutputRoute {
        OutputRoute {
            source_channel: source.into(),
            amplitude_scaling: None,
            phase_shift: None,
        }
    }

    fn route_with_fixed_amplitude(source: u32, amplitude: f64) -> OutputRoute {
        OutputRoute {
            source_channel: source.into(),
            amplitude_scaling: Some(ValueOrParameter::Value(amplitude)),
            phase_shift: None,
        }
    }

    fn route_with_param_amplitude(source: u32, param_uid: ParameterUid) -> OutputRoute {
        OutputRoute {
            source_channel: source.into(),
            amplitude_scaling: Some(ValueOrParameter::Parameter(param_uid)),
            phase_shift: None,
        }
    }

    #[test]
    fn no_output_routing_returns_empty_maps() {
        let mut fx = TestFixture::new();
        fx.add_signal(0, 0, "SGCHANNELS/0/OUTPUT", vec![]);

        let result = fx.run().unwrap();
        assert!(result.channel_map.is_empty());
        assert!(result.delay_signal.is_empty());
    }

    #[test]
    fn single_route_populates_channel_map_and_delays() {
        let mut fx = TestFixture::new();
        fx.add_device_channel(0, 0, "SGCHANNELS/0/OUTPUT");
        fx.add_signal(0, 0, "SGCHANNELS/1/OUTPUT", vec![route(0)]);
        fx.add_signal(1, 0, "SGCHANNELS/0/OUTPUT", vec![]);

        let result = fx.run().unwrap();

        assert_eq!(result.channel_map[&0.into()], 0);
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
        fx.add_device_channel(0, 0, "SGCHANNELS/0/OUTPUT");
        fx.add_signal(0, 0, "SGCHANNELS/1/OUTPUT", vec![route(0)]);
        fx.add_signal(1, 0, "SGCHANNELS/0/OUTPUT", vec![]);
        fx.add_signal(2, 0, "SGCHANNELS/2/OUTPUT", vec![]);

        let result = fx.run().unwrap();
        assert!(!result.delay_signal.contains_key(&fx.signals[2].uid));
    }

    #[test]
    fn three_routes_to_same_target_is_valid() {
        let mut fx = TestFixture::new();
        fx.add_device_channel(0, 0, "SGCHANNELS/0/OUTPUT");
        fx.add_device_channel(1, 0, "SGCHANNELS/1/OUTPUT");
        fx.add_device_channel(2, 0, "SGCHANNELS/2/OUTPUT");
        fx.add_signal(
            0,
            0,
            "SGCHANNELS/3/OUTPUT",
            vec![route(0), route(1), route(2)],
        );
        assert!(fx.run().is_ok());
    }

    #[test]
    fn four_routes_to_same_target_is_an_error() {
        let mut fx = TestFixture::new();
        fx.add_device_channel(0, 0, "SGCHANNELS/0/OUTPUT");
        fx.add_device_channel(1, 0, "SGCHANNELS/1/OUTPUT");
        fx.add_device_channel(2, 0, "SGCHANNELS/2/OUTPUT");
        fx.add_device_channel(3, 0, "SGCHANNELS/3/OUTPUT");
        fx.add_signal(
            0,
            0,
            "SGCHANNELS/4/OUTPUT",
            vec![route(0), route(1), route(2), route(3)],
        );
        let err = fx.run().unwrap_err().to_string();
        assert!(err.contains("Maximum of three"));
    }

    #[test]
    fn unknown_source_channel_is_an_error() {
        let mut fx = TestFixture::new();
        fx.add_signal(0, 0, "SGCHANNELS/0/OUTPUT", vec![route(999)]);
        assert!(fx.run().is_err());
    }

    #[test]
    fn duplicate_source_channel_is_an_error() {
        let mut fx = TestFixture::new();
        fx.add_device_channel(0, 0, "SGCHANNELS/0/OUTPUT");
        fx.add_signal(0, 0, "SGCHANNELS/1/OUTPUT", vec![route(0), route(0)]);
        let err = fx.run().unwrap_err().to_string();
        assert!(err.contains("Duplicate"));
    }

    #[test]
    fn source_same_as_target_is_an_error() {
        let mut fx = TestFixture::new();
        fx.add_device_channel(0, 0, "SGCHANNELS/0/OUTPUT");
        fx.add_signal(0, 0, "SGCHANNELS/0/OUTPUT", vec![route(0)]);
        let err = fx.run().unwrap_err().to_string();
        assert!(err.contains("same as the target"));
    }

    #[test]
    fn fixed_amplitude_at_boundary_values_is_valid() {
        let mut fx = TestFixture::new();
        fx.add_device_channel(0, 0, "SGCHANNELS/0/OUTPUT");
        fx.add_device_channel(2, 0, "SGCHANNELS/2/OUTPUT");
        fx.add_signal(
            0,
            0,
            "SGCHANNELS/1/OUTPUT",
            vec![
                route_with_fixed_amplitude(0, 0.0),
                route_with_fixed_amplitude(2, 1.0),
            ],
        );
        assert!(fx.run().is_ok());
    }

    #[test]
    fn fixed_amplitude_above_one_is_an_error() {
        let mut fx = TestFixture::new();
        fx.add_device_channel(0, 0, "SGCHANNELS/0/OUTPUT");
        fx.add_signal(
            0,
            0,
            "SGCHANNELS/1/OUTPUT",
            vec![route_with_fixed_amplitude(0, 1.01)],
        );
        let err = fx.run().unwrap_err().to_string();
        assert!(err.contains("[0.0, 1.0]"));
    }

    #[test]
    fn fixed_amplitude_below_zero_is_an_error() {
        let mut fx = TestFixture::new();
        fx.add_device_channel(0, 0, "SGCHANNELS/0/OUTPUT");
        fx.add_signal(
            0,
            0,
            "SGCHANNELS/1/OUTPUT",
            vec![route_with_fixed_amplitude(0, -0.01)],
        );
        let err = fx.run().unwrap_err().to_string();
        assert!(err.contains("[0.0, 1.0]"));
    }

    #[test]
    fn parameter_amplitude_out_of_range_is_an_error() {
        let mut fx = TestFixture::new();
        fx.add_parameter(0, ParameterValues::Float64(vec![0.5, 1.5]));
        fx.add_device_channel(0, 0, "SGCHANNELS/0/OUTPUT");
        fx.add_signal(
            0,
            0,
            "SGCHANNELS/1/OUTPUT",
            vec![route_with_param_amplitude(0, 0.into())],
        );
        let err = fx.run().unwrap_err().to_string();
        assert!(err.contains("[0.0, 1.0]"));
    }

    #[test]
    fn parameter_amplitude_non_float_is_an_error() {
        let mut fx = TestFixture::new();
        fx.add_parameter(0, ParameterValues::Integer64(vec![1]));
        fx.add_device_channel(0, 0, "SGCHANNELS/0/OUTPUT");
        fx.add_signal(
            0,
            0,
            "SGCHANNELS/1/OUTPUT",
            vec![route_with_param_amplitude(0, 0.into())],
        );
        let err = fx.run().unwrap_err().to_string();
        assert!(err.contains("floating point"));
    }

    #[test]
    fn parameter_amplitude_valid_range_is_ok() {
        let mut fx = TestFixture::new();
        fx.add_parameter(0, ParameterValues::Float64(vec![0.0, 0.5, 1.0]));
        fx.add_device_channel(0, 0, "SGCHANNELS/0/OUTPUT");
        fx.add_signal(
            0,
            0,
            "SGCHANNELS/1/OUTPUT",
            vec![route_with_param_amplitude(0, 0.into())],
        );
        assert!(fx.run().is_ok());
    }

    #[test]
    fn parameter_nan_amplitude_is_an_error() {
        let mut fx = TestFixture::new();
        fx.add_parameter(0, ParameterValues::Float64(vec![f64::NAN]));
        fx.add_device_channel(0, 0, "SGCHANNELS/0/OUTPUT");
        fx.add_signal(
            0,
            0,
            "SGCHANNELS/1/OUTPUT",
            vec![route_with_param_amplitude(0, 0.into())],
        );
        let err = fx.run().unwrap_err().to_string();
        assert!(err.contains("expects values in the range [0.0, 1.0]"));
    }
}

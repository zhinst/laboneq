// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use laboneq_dsl::types::{SignalUid, ValueOrParameter};
use laboneq_ir::system::AwgDevice;

use laboneq_common::types::{AwgKey, DeviceKind, SignalKind};
use laboneq_scheduler::FeedbackCalculator;
use laboneq_units::duration::{Duration, Second, seconds};

use crate::error::Error;
use crate::qccs_feedback_calculator::feedback_model::QCCSFeedbackModel;
use crate::signal_view::SignalView;

type Samples = i64;
/// Latency in samples added by the ExecuteTableEntry command.
const EXECUTETABLEENTRY_LATENCY: Samples = 3;

#[derive(Debug, Clone)]
pub enum FeedbackDevice {
    Shfqa,
    Uhfqa,
    Hdawg,
    Shfsg,
    Shfqc,
}

impl std::fmt::Display for FeedbackDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FeedbackDevice::Shfqa => write!(f, "SHFQA"),
            FeedbackDevice::Uhfqa => write!(f, "UHFQA"),
            FeedbackDevice::Hdawg => write!(f, "HDAWG"),
            FeedbackDevice::Shfsg => write!(f, "SHFSG"),
            FeedbackDevice::Shfqc => write!(f, "SHFQC"),
        }
    }
}

pub trait FeedbackModel {
    /// Get the feedback latency in samples from the feedback model.
    fn get_latency(
        &self,
        acquisition_end_samples: Samples,
        qa_device: &FeedbackDevice,
        sg_device: &FeedbackDevice,
        local_feedback: bool,
    ) -> anyhow::Result<Samples>;
}

/// QCCS feedback calculator for latency calculation in feedback experiments.
///
/// The calculator uses a feedback model to compute the latency based on
/// acquisition and generator signal parameters for each signal involved in
/// the feedback loop.
pub struct QccsFeedbackCalculator<M: FeedbackModel> {
    acquisition_signal_params: HashMap<SignalUid, AcquisitionParameters>,
    generator_signal_params: HashMap<SignalUid, GeneratorParameters>,
    acquisition_generator_mapping: HashMap<SignalUid, SignalUid>,
    /// The feedback model used for latency calculations.
    model: M,
}

/// Parameters required for acquisition latency calculation.
struct AcquisitionParameters {
    acquisition_start_delay: Duration<Second>,
    acquisition_signal_delay: Duration<Second>,
    acquisition_port_delay: ValueOrParameter<Duration<Second>>,
    /// Signal delay of the associated generator signal, if any.
    generator_signal_delay: Duration<Second>,
    /// Port delay of the associated generator signal, if any.
    generator_port_delay: ValueOrParameter<Duration<Second>>,
    integration_dsp_latency: Duration<Second>,
    /// Acquisition device properties
    sampling_rate: f64,
    port_granularity_samples: i64,
    device: FeedbackDevice,
}

impl AcquisitionParameters {
    fn fixed_generator_port_delay(&self) -> Result<Duration<Second>, Error> {
        match self.generator_port_delay {
            ValueOrParameter::Value(value) => Ok(value),
            _ => Err(Error::new(
                "Feedback measure line requires a constant 'port_delay', but it is a sweep parameter.",
            )),
        }
    }

    fn fixed_acquisition_port_delay(&self) -> Result<Duration<Second>, Error> {
        match self.acquisition_port_delay {
            ValueOrParameter::Value(value) => Ok(value),
            _ => Err(Error::new(
                "Feedback acquisition line requires a constant 'port_delay', but it is a sweep parameter.",
            )),
        }
    }
}

/// Parameters required for generator latency calculation.
#[derive(Debug, Clone)]
struct GeneratorParameters {
    lead_time: Duration<Second>,
    signal_delay: Duration<Second>,
    /// Generator device properties
    sampling_rate: f64,
    sample_multiple: i64,
    device: FeedbackDevice,
}

impl QccsFeedbackCalculator<QCCSFeedbackModel> {
    pub fn new<'a>(
        signals: impl Iterator<Item = SignalView<'a>> + 'a,
    ) -> Result<QccsFeedbackCalculator<QCCSFeedbackModel>, Error> {
        Self::new_with_model(signals, QCCSFeedbackModel::new())
    }
}

impl<M> QccsFeedbackCalculator<M>
where
    M: FeedbackModel,
{
    pub(crate) fn new_with_model<'a>(
        signals: impl Iterator<Item = SignalView<'a>> + 'a,
        model: M,
    ) -> Result<QccsFeedbackCalculator<M>, Error> {
        let signal_map = signals
            .into_iter()
            .map(|s| (s.uid(), s))
            .collect::<HashMap<_, _>>();

        // Organize signals into acquisition and generator signals per AWG.
        // Acquisition signals can share AWGs with generator signals.
        let mut acquisition_signals_per_awg: HashMap<AwgKey, Vec<SignalUid>> = HashMap::new();
        // There is only one generator signal per AWG(?)
        let mut generator_signals_per_awg: HashMap<AwgKey, SignalUid> = HashMap::new();

        // First map acquisition and generator signals
        for signal in signal_map.values() {
            match signal.signal_kind() {
                SignalKind::Integration => {
                    acquisition_signals_per_awg
                        .entry(*signal.awg_key())
                        .or_default()
                        .push(signal.uid());
                }
                _ => {
                    generator_signals_per_awg.insert(*signal.awg_key(), signal.uid());
                }
            }
        }
        let mut acquisition_generator_mapping = HashMap::new();
        let mut generator_to_acquisition_mapping: HashMap<SignalUid, Vec<SignalUid>> =
            HashMap::new();
        for (awg_key, acquisition_signal) in acquisition_signals_per_awg {
            for acq_signal_uid in acquisition_signal.iter() {
                if let Some(generator_signal) = generator_signals_per_awg.get(&awg_key) {
                    acquisition_generator_mapping.insert(*acq_signal_uid, *generator_signal);
                    generator_to_acquisition_mapping
                        .entry(*generator_signal)
                        .or_default()
                        .push(*acq_signal_uid);
                }
            }
        }

        // Then extract parameters for acquisition and generator signals, and map acquisition signals to their associated generator signal parameters.
        let mut acquisition_signal_params = HashMap::new();
        let mut generator_signal_params = HashMap::new();

        for signal in signal_map.values() {
            match signal.signal_kind() {
                SignalKind::Integration => {
                    let acq_signal = signal_map.get(&signal.uid()).expect(
                        "Internal error: Acquisition signal not found in feedback calculator",
                    );
                    let gen_signal = acquisition_generator_mapping
                        .get(&signal.uid())
                        .and_then(|gen_uid| signal_map.get(gen_uid));
                    let params = Self::extract_acquisition_parameters(acq_signal, gen_signal)?;
                    acquisition_signal_params.insert(signal.uid(), params);
                }
                _ => {
                    let params = Self::extract_generator_parameters(signal)?;
                    generator_signal_params.insert(signal.uid(), params);
                }
            }
        }

        let result = QccsFeedbackCalculator {
            acquisition_signal_params,
            generator_signal_params,
            acquisition_generator_mapping,
            model,
        };
        Ok(result)
    }

    /// Get the generator parameters for a given signal UID, checking both generator signals and acquisition signals mapped to generator signals.
    fn generator_parameters(&self, signal_uid: SignalUid) -> Option<&GeneratorParameters> {
        if let Some(params) = self.generator_signal_params.get(&signal_uid) {
            return Some(params);
        }
        self.acquisition_generator_mapping
            .get(&signal_uid)
            .and_then(|gen_uid| self.generator_signal_params.get(gen_uid))
    }
}

impl<M> QccsFeedbackCalculator<M>
where
    M: FeedbackModel,
{
    /// Calculate the end of the integration in seconds from trigger.
    ///
    /// The following elements are considered:
    ///
    /// - The start time (in samples from trigger) of the acquisition
    /// - The length of the integration kernel
    /// - The lead time of the acquisition AWG
    /// - The sum of the settings of the delay_signal parameter for the acquisition AWG
    ///   for measure and acquire pulse
    /// - The sum of the settings of the port_delay parameter for the acquisition device
    ///   for measure and acquire pulse
    pub(super) fn compute_start_with_latency(
        &self,
        absolute_start: Duration<Second>,
        acquisition_length: Duration<Second>,
        local_feedback: bool,
        acquisition_signal: SignalUid,
        associated_signals: &[SignalUid],
    ) -> Result<Duration<Second>, Error> {
        let acquisition_params = self
            .acquisition_signal_params
            .get(&acquisition_signal)
            .expect(
                "Internal error: Acquisition signal parameters not found in feedback calculator",
            );
        let acquisition_end_samples = self.calculate_acquisition_end_samples(
            absolute_start,
            acquisition_length,
            acquisition_params,
        )?;
        let mut earliest_execute_table_entry = seconds(0.0);
        for signal_uid in associated_signals {
            if *signal_uid == acquisition_signal {
                continue;
            }
            let generator_params = self.generator_parameters(*signal_uid).ok_or_else(|| {
                Error::new(
                    "Internal error: Generator signal parameters not found in feedback calculator",
                )
            })?;
            let latency_in_s = self.calculate_latency_for_signal(
                acquisition_end_samples,
                acquisition_params,
                generator_params,
                local_feedback,
                absolute_start,
                acquisition_length,
            )?;

            // Calculate the shift of compiler zero time for the SG; we may subtract this
            // from the time of arrival (which is measured since the trigger) to get the
            // start point in compiler time. The following elements need to be considered:
            // - The lead time of the acquisition AWG
            // - The setting of the delay_signal parameter for the acquisition AWG
            // - The time of arrival computed above
            // todo(JL): Check whether also the port_delay can be added - probably not.
            let calculated_latency =
                latency_in_s - (generator_params.lead_time + generator_params.signal_delay);
            if calculated_latency > earliest_execute_table_entry {
                earliest_execute_table_entry = calculated_latency;
            }
        }
        Ok(earliest_execute_table_entry)
    }

    fn calculate_latency_for_signal(
        &self,
        acquisition_end_samples: Samples,
        acquisition_params: &AcquisitionParameters,
        signal: &GeneratorParameters,
        local_feedback: bool,
        absolute_start: Duration<Second>,
        acquisition_length: Duration<Second>,
    ) -> Result<Duration<Second>, Error> {
        if matches!(
            acquisition_params.device,
            FeedbackDevice::Shfqa | FeedbackDevice::Shfqc
        ) {
            // TODO: Currently the Python error is not properly propagated to enable
            // Python tracebacks. We should improve this in the future, currently it is
            // fine as it is not something the users should encounter.
            let time_of_arrival_at_register = self
                .model
                .get_latency(
                    acquisition_end_samples,
                    &acquisition_params.device,
                    &signal.device,
                    local_feedback,
                )
                .map_err(|e| {
                    Error::new(format!(
                        "Failed to get latency from QCCS feedback model: {}",
                        e
                    ))
                })?;
            // We also add three latency cycles here, which then, in the code generator, will
            // be subtracted again for the latency argument of executeTableEntry. The reason
            // is that there is an additional latency of three cycles from the execution
            // of the command in the sequencer until the arrival of the chosen waveform in
            // the wave player queue. For now, we look at the time the pulse is played
            // (arrival time of data in register + 3), which also simplifies phase
            // calculation for software modulated signals, and take care of subtracting it
            // later
            let time_of_pulse_played = time_of_arrival_at_register + EXECUTETABLEENTRY_LATENCY;
            let sequencer_rate = signal.sampling_rate / signal.sample_multiple as f64;
            let sg_seq_dt_for_latency = 1.0 / (2.0 * sequencer_rate);
            let latency = time_of_pulse_played as f64 * sg_seq_dt_for_latency;
            Ok(latency.into())
        } else {
            // Gen1 system
            const LATENCY: Duration<Second> = seconds(900e-9); // https://www.zhinst.com/ch/en/blogs/practical-active-qubit-reset
            let qa_total_port_delay = calculate_total_port_delay(
                &[
                    acquisition_params.fixed_generator_port_delay()?,
                    acquisition_params.fixed_acquisition_port_delay()?,
                    acquisition_params.integration_dsp_latency,
                ],
                acquisition_params.sampling_rate,
                acquisition_params.port_granularity_samples,
            );
            Ok(absolute_start
                + acquisition_length
                + LATENCY
                + acquisition_params.acquisition_start_delay
                + acquisition_params.acquisition_signal_delay
                + acquisition_params.generator_signal_delay
                + qa_total_port_delay)
        }
    }

    fn extract_generator_parameters(
        generator_signal: &SignalView<'_>,
    ) -> Result<GeneratorParameters, Error> {
        let lead_time = generator_signal.start_delay();
        let signal_delay = generator_signal.signal_delay();
        let sampling_rate = generator_signal.sampling_rate();
        let sample_multiple = generator_signal.device_traits().sample_multiple as i64;

        Ok(GeneratorParameters {
            lead_time,
            signal_delay,
            sampling_rate,
            sample_multiple,
            device: awg_device_to_feedback_device(generator_signal.device()),
        })
    }

    fn extract_acquisition_parameters(
        acquisition_signal: &SignalView<'_>,
        generator_signal: Option<&SignalView<'_>>,
    ) -> Result<AcquisitionParameters, Error> {
        let mut generator_signal_delay = seconds(0.0);
        let mut generator_port_delay = ValueOrParameter::Value(seconds(0.0));
        if let Some(generator_signal) = generator_signal {
            generator_signal_delay = generator_signal.signal_delay();
            generator_port_delay = *generator_signal
                .port_delay()
                .unwrap_or(&ValueOrParameter::Value(seconds(0.0)));
        }
        let acquisition_start_delay = acquisition_signal.start_delay();
        let acquisition_signal_delay = acquisition_signal.signal_delay();
        let acquisition_port_delay = *acquisition_signal
            .port_delay()
            .unwrap_or(&ValueOrParameter::Value(seconds(0.0)));

        let integration_dsp_latency = acquisition_signal
            .device_traits()
            .integration_dsp_latency
            .unwrap_or(seconds(0.0));

        let sampling_rate = acquisition_signal.sampling_rate();
        let port_granularity_samples =
            acquisition_signal.device_traits().port_delay_granularity as i64;

        Ok(AcquisitionParameters {
            acquisition_start_delay,
            acquisition_signal_delay,
            acquisition_port_delay,
            generator_signal_delay,
            generator_port_delay,
            integration_dsp_latency,
            sampling_rate,
            port_granularity_samples,
            device: awg_device_to_feedback_device(acquisition_signal.device()),
        })
    }

    fn calculate_acquisition_end_samples(
        &self,
        absolute_start: Duration<Second>,
        acquisition_length: Duration<Second>,
        params: &AcquisitionParameters,
    ) -> Result<Samples, Error> {
        let qa_total_port_delay = calculate_total_port_delay(
            &[
                params.fixed_generator_port_delay()?,
                params.fixed_acquisition_port_delay()?,
                // The controller may offset the integration delay node to compensate the DSP
                // latency, thereby aligning measure and acquire for port_delay=0.
                params.integration_dsp_latency,
            ],
            params.sampling_rate,
            params.port_granularity_samples,
        );
        let result = ((absolute_start
            + acquisition_length
            + params.acquisition_start_delay
            + params.acquisition_signal_delay
            + params.generator_signal_delay
            + qa_total_port_delay)
            * params.sampling_rate)
            .value()
            .round() as Samples;
        Ok(result)
    }
}

/// Calculate the total port delay with granularity adjustment.
///
/// The total port delay is calculated by summing the individual port delays,
/// multiplying by the sample frequency, rounding to the nearest integer,
/// and then adjusting to the nearest multiple of the granularity in samples.
/// Finally, the total delay is converted back to seconds.
fn calculate_total_port_delay(
    port_delays: &[Duration<Second>],
    sample_frequency_hz: f64,
    granularity_samples: i64,
) -> Duration<Second> {
    let total_delay_samples = port_delays
        .iter()
        .map(|d| (d.value() * sample_frequency_hz).round())
        .sum::<f64>() as i64;
    let total_delay = ((total_delay_samples + granularity_samples - 1) / granularity_samples)
        * granularity_samples;
    (total_delay as f64 / sample_frequency_hz).into()
}

impl FeedbackCalculator for QccsFeedbackCalculator<QCCSFeedbackModel> {
    type Error = Error;

    fn compute_feedback_latency(
        &self,
        absolute_start: Duration<Second>,
        acquisition_length: Duration<Second>,
        local_feedback: bool,
        acquisition_signal: SignalUid,
        associated_signals: &[SignalUid],
    ) -> Result<Duration<Second>, Self::Error> {
        let latency_s = self.compute_start_with_latency(
            absolute_start,
            acquisition_length,
            local_feedback,
            acquisition_signal,
            associated_signals,
        )?;
        Ok(latency_s)
    }
}

fn awg_device_to_feedback_device(device: &AwgDevice) -> FeedbackDevice {
    if device.is_shfqc() {
        FeedbackDevice::Shfqc
    } else {
        match device.kind() {
            DeviceKind::Shfqa => FeedbackDevice::Shfqa,
            DeviceKind::Uhfqa => FeedbackDevice::Uhfqa,
            DeviceKind::Hdawg => FeedbackDevice::Hdawg,
            DeviceKind::Shfsg => FeedbackDevice::Shfsg,
            _ => panic!("Unsupported device kind for feedback: {:?}", device.kind()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::abs_diff_eq;
    use laboneq_common::{named_id::NamedId, types::PhysicalDeviceUid, types::SignalKind};
    use laboneq_ir::signal::{Signal, builder::SignalBuilder};
    use laboneq_ir::system::AwgDevice;

    struct MockFeedbackModel {}

    /// A mock feedback model that returns a latency based on acquisition end samples.
    ///
    /// The latency is calculated as acquisition_end_samples / 8.0.
    /// The purpose of this mock is to facilitate testing of the feedback calculator
    /// without relying on the actual QCCS feedback model.
    impl MockFeedbackModel {
        fn new() -> Self {
            Self {}
        }
    }

    impl FeedbackModel for MockFeedbackModel {
        fn get_latency(
            &self,
            _acquisition_end_samples: Samples,
            _qa_device: &FeedbackDevice,
            _sg_device: &FeedbackDevice,
            _local_feedback: bool,
        ) -> anyhow::Result<Samples> {
            Ok((_acquisition_end_samples / 8) as Samples)
        }
    }

    fn create_signal_device(
        uid: u32,
        awg_uid: u64,
        signal_kind: SignalKind,
        device_kind: DeviceKind,
        signal_delay: f64,
        port_delay: f64,
    ) -> (Signal, AwgDevice) {
        let awg_key = AwgKey(awg_uid);
        let device = AwgDevice::builder(0.into(), PhysicalDeviceUid(0), device_kind)
            .shfqc(true)
            .build();

        let signal = SignalBuilder::new(
            NamedId::debug_id(uid).into(),
            2e9,
            awg_key,
            device.uid(),
            signal_kind, // Replace `None` with the appropriate value if needed
        )
        .signal_delay(signal_delay)
        .port_delay(ValueOrParameter::Value(seconds(port_delay)))
        .start_delay(160.0 / 2e9)
        .build();
        (signal, device)
    }

    #[derive(Debug)]
    struct SignalDelayTestCase {
        port_delay_qa_acq: f64,
        port_delay_qa_meas: f64,
        delay_signal_qa_acq: f64,
        delay_signal_qa_meas: f64,
        expected_start: f64,
    }

    /// Default integration delay for SHFQA devices
    const SHFQA_DEFAULT_INTEGRATION_DELAY: f64 = 212e-9;

    /// Test various combinations of port_delay and signal_delay for acquisition and
    /// generation signals on the same AWG.
    ///
    /// The expected start time is calculated based on the baseline start time
    /// plus the port delays of acquisition and generation signals, as well as
    /// the signal delay of the acquisition signal.
    #[test]
    fn test_compute_feedback_latency_same_awg() {
        let baseline_start: f64 = 2.32e-07;

        let properties = [
            // No delays
            SignalDelayTestCase {
                port_delay_qa_acq: 0.0,
                port_delay_qa_meas: 0.0,
                delay_signal_qa_meas: 0.0,
                delay_signal_qa_acq: 0.0,
                expected_start: baseline_start,
            },
            // Single delays on QA acquisition port
            SignalDelayTestCase {
                port_delay_qa_acq: 32e-9,
                port_delay_qa_meas: 0.0,
                delay_signal_qa_meas: 0.0,
                delay_signal_qa_acq: 0.0,
                expected_start: baseline_start + 32e-9,
            },
            // Single delays on QA measurement port
            SignalDelayTestCase {
                port_delay_qa_acq: 0.0,
                port_delay_qa_meas: 32e-9,
                delay_signal_qa_meas: 0.0,
                delay_signal_qa_acq: 0.0,
                expected_start: baseline_start + 32e-9,
            },
            // Single delays on QA measurement signal
            // Cancelled out.
            SignalDelayTestCase {
                port_delay_qa_acq: 0.0,
                port_delay_qa_meas: 0.0,
                delay_signal_qa_meas: 32e-9,
                delay_signal_qa_acq: 0.0,
                expected_start: baseline_start,
            },
            // Single delays on QA acquisition signal
            SignalDelayTestCase {
                port_delay_qa_acq: 0.0,
                port_delay_qa_meas: 0.0,
                delay_signal_qa_meas: 0.0,
                delay_signal_qa_acq: 32e-9,
                expected_start: baseline_start + 32e-9,
            },
            // Combinations of delays
            SignalDelayTestCase {
                port_delay_qa_acq: 16e-9,
                port_delay_qa_meas: 16e-9,
                delay_signal_qa_meas: 0.0,
                delay_signal_qa_acq: 0.0,
                expected_start: baseline_start + 16e-9 + 16e-9,
            },
            // Combination with acquisition signal delay
            SignalDelayTestCase {
                port_delay_qa_acq: 0.0,
                port_delay_qa_meas: 0.0,
                delay_signal_qa_meas: 0.0,
                delay_signal_qa_acq: 16e-9,
                expected_start: baseline_start + 16e-9,
            },
            // Combination with measurement signal delay
            SignalDelayTestCase {
                port_delay_qa_acq: 32e-9,
                port_delay_qa_meas: 0.0,
                delay_signal_qa_meas: 32e-9,
                delay_signal_qa_acq: 0.0,
                expected_start: baseline_start + 32e-9,
            },
        ];

        let acquisition_absolute_start = seconds(100e-9);
        let acquisition_length = seconds(120e-9);

        for case in properties.iter() {
            let (acq_signal, acq_device) = create_signal_device(
                0,
                0,
                SignalKind::Integration,
                DeviceKind::Shfqa,
                case.delay_signal_qa_acq - SHFQA_DEFAULT_INTEGRATION_DELAY,
                case.port_delay_qa_acq,
            );
            let (gen_signal, gen_device) = create_signal_device(
                1,
                0,
                SignalKind::Iq,
                DeviceKind::Shfsg,
                case.delay_signal_qa_meas,
                case.port_delay_qa_meas,
            );

            let mock_model = MockFeedbackModel::new();
            let calculator = QccsFeedbackCalculator::new_with_model(
                [
                    SignalView::new(&acq_device, &acq_signal),
                    SignalView::new(&gen_device, &gen_signal),
                ]
                .into_iter(),
                mock_model,
            )
            .unwrap();

            let result = calculator.compute_start_with_latency(
                acquisition_absolute_start,
                acquisition_length,
                false,
                acq_signal.uid,
                &[gen_signal.uid],
            );

            let latency = result.unwrap();
            let msg = format!("Failed for case: {:?}, expected: {}", case, latency.value());
            assert!(
                abs_diff_eq!(latency.value(), case.expected_start),
                "{}",
                msg
            );
        }
    }

    /// Test various combinations of port_delay and signal_delay for acquisition and
    /// generation signals on different AWGs.
    #[test]
    fn test_compute_feedback_latency_associated_signal() {
        let baseline_start: f64 = 2.32e-07;

        let properties = [
            // No delays
            SignalDelayTestCase {
                port_delay_qa_acq: 0.0,
                port_delay_qa_meas: 0.0,
                delay_signal_qa_meas: 0.0,
                delay_signal_qa_acq: 0.0,
                expected_start: baseline_start,
            },
            // Generator delay will shift to earlier time.
            SignalDelayTestCase {
                port_delay_qa_acq: 0.0,
                port_delay_qa_meas: 0.0,
                delay_signal_qa_meas: 32e-9,
                delay_signal_qa_acq: 0.0,
                expected_start: baseline_start - 32e-9,
            },
            // Delays on acquisition and measurement signals cancelled out.
            SignalDelayTestCase {
                port_delay_qa_acq: 32e-9,
                port_delay_qa_meas: 0.0,
                delay_signal_qa_meas: 32e-9,
                delay_signal_qa_acq: 0.0,
                expected_start: baseline_start,
            },
        ];

        let acquisition_absolute_start = seconds(100e-9);
        let acquisition_length = seconds(120e-9);

        for case in properties.iter() {
            let (acq_signal, acq_device) = create_signal_device(
                0,
                0,
                SignalKind::Integration,
                DeviceKind::Shfqa,
                case.delay_signal_qa_acq - SHFQA_DEFAULT_INTEGRATION_DELAY,
                case.port_delay_qa_acq,
            );
            let (gen_signal, gen_device) = create_signal_device(
                1,
                1,
                SignalKind::Iq,
                DeviceKind::Shfsg,
                case.delay_signal_qa_meas,
                case.port_delay_qa_meas,
            );

            let mock_model = MockFeedbackModel::new();
            let calculator = QccsFeedbackCalculator::new_with_model(
                [
                    SignalView::new(&acq_device, &acq_signal),
                    SignalView::new(&gen_device, &gen_signal),
                ]
                .into_iter(),
                mock_model,
            )
            .unwrap();

            let result = calculator
                .compute_start_with_latency(
                    acquisition_absolute_start,
                    acquisition_length,
                    false,
                    acq_signal.uid,
                    &[gen_signal.uid],
                )
                .unwrap();

            let latency = result;
            let msg = format!("Failed for case: {:?}, expected: {}", case, latency.value());
            assert!(
                abs_diff_eq!(latency.value(), case.expected_start),
                "{}",
                msg
            );
        }
    }
}

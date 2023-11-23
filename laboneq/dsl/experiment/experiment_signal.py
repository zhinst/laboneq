# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from laboneq.core.types.enums import PortMode
from laboneq.core.utilities.dsl_dataclass_decorator import classformatter
from laboneq.dsl.calibration import MixerCalibration, SignalCalibration
from laboneq.dsl.calibration.amplifier_pump import AmplifierPump
from laboneq.dsl.calibration.oscillator import Oscillator
from laboneq.dsl.calibration.precompensation import Precompensation
from laboneq.dsl.device.io_units.logical_signal import (
    LogicalSignalRef,
    resolve_logical_signal_ref,
)

experiment_signal_id = 0


def experiment_signal_id_generator():
    global experiment_signal_id
    retval = f"s{experiment_signal_id}"
    experiment_signal_id += 1
    return retval


@classformatter
@dataclass(init=False, repr=True, order=True)
class ExperimentSignal:
    """A signal within an experiment.

    Experiment signals are mapped to logical signals before an
    experiment is executed.

    The experiment signal calibration maybe specified here either
    by passing the `calibration` parameter or by specifying the
    parts of the calibration as they would be passed to
    [SignalCalibration][laboneq.dsl.calibration.signal_calibration.SignalCalibration].
    See the documentation of
    [SignalCalibration][laboneq.dsl.calibration.signal_calibration.SignalCalibration]
    for details of the individual calibration options.

    Parameters:
        uid:
            The unique identifier for the signal. If not specified,
            one will be automatically generated.
        map_to:
            The logical signal to map to this experiment signal to.
            If not specified, it should be set before the experiment
            is compiled. If both this and `mapped_logical_signal_path`,
            the `mapped_logical_signal_path` value is used.
        calibration:
            The signal calibration. If provided, the
            values of the other calibration parameters are ignored.
            If the signal calibration is not specified via either
            this parameter or the other parameters,
            it should be set before the experiment is compiled.
        oscillator:
            The oscillator assigned to the signal calibration.
            Ignored if the `calibration` parameter is set.
        amplitude:
            The signal calibration amplitude.
            Only supported by the SHFQA.
            The amplitude setting applies to all signals on the same channel.
            Ignored if the `calibration` parameter is set.
        port_delay:
            The signal calibration port delay.
            Ignored if the `calibration` parameter is set.
        mixer_calibration:
            The signal mixer calibration.
            Ignored if the `calibration` parameter is set.
        precompensation:
            The signal calibration precompenstation settings.
            Ignored if the `calibration` parameter is set.
            Only supported by HDAWG signals.
        local_oscillator:
            The local oscillator assigned to the signal calibration.
            Only supported by SHFSG, SHFQA and SHFQC signals.
            Ignored if the `calibration` parameter is set.
        range:
            The output or input range setting for the signal calibration.
            Ignored if the `calibration` parameter is set.
        port_mode:
            The SHFSG, SHFQA and SHFQC port mode signal calibration.
            Ignored if the `calibration` parameter is set.
        threshold:
            The sginal calibration state discrimation threshold.
            Only supported for acquisition signals on the UHFQA, SHFQA
            and SHFQC.
            Ignored if the `calibration` parameter is set.
        mapped_logical_signal_path:
            The path of the logical signal to map this experiment
            signal to. If not specified, it should be set before
            the experiment is compiled. If both this and `map_to`
            are specified, this value is used.

    Attributes:
        uid (str):
            The unique identifier for the signal.
        calibration (SignalCalibration | None):
            The signal calibration. Must be set before the experiment
            is executed if the calibration for this signal is
            used by the experiment.
        mapped_logical_signal_path (str | None):
            The path of the logical signal mapped to this
            experiment signal. Must be set before the experiment
            is executed.
    """

    uid: str
    calibration: Optional[SignalCalibration]
    mapped_logical_signal_path: str | None

    def __init__(
        self,
        uid: str | None = None,
        map_to: LogicalSignalRef | None = None,
        calibration: SignalCalibration | None = None,
        oscillator: Oscillator | None = None,
        amplitude: float | None = None,
        port_delay: float | None = None,
        delay_signal: float | None = None,
        mixer_calibration: MixerCalibration | None = None,
        precompensation: Precompensation | None = None,
        local_oscillator: Oscillator | None = None,
        range: float | None = None,
        port_mode: PortMode | None = None,
        threshold: float | None = None,
        mapped_logical_signal_path: str | None = None,
    ):
        if uid is None:
            self.uid = experiment_signal_id_generator()
        else:
            self.uid = uid

        if (calibration is None) and (
            oscillator
            or amplitude is not None
            or port_delay is not None
            or delay_signal is not None
            or mixer_calibration is not None
            or precompensation is not None
            or local_oscillator is not None
            or range is not None
            or port_mode is not None
            or threshold is not None
        ):
            self.calibration = SignalCalibration(
                oscillator=oscillator,
                amplitude=amplitude,
                port_delay=port_delay,
                delay_signal=delay_signal,
                mixer_calibration=mixer_calibration,
                precompensation=precompensation,
                local_oscillator=local_oscillator,
                range=range,
                port_mode=port_mode,
                threshold=threshold,
            )
        else:
            self.calibration = calibration
        self.mapped_logical_signal_path = None
        self.map(map_to)
        if mapped_logical_signal_path is not None:
            self.map(mapped_logical_signal_path)

    def is_mapped(self) -> bool:
        """Return true if the signal is mapped to a logical signal path.

        Returns:
            is_mapped:
                True if this experiment signal is mapped to a logical signal.
        """
        return self.mapped_logical_signal_path is not None

    def map(self, to: LogicalSignalRef):
        """Map this signal to a logical signal.

        Parameters:
            to:
                The logical signal to map this experiment signal to.
        """
        self.mapped_logical_signal_path = resolve_logical_signal_ref(to)

    def disconnect(self):
        """Disconnect the experiment signal from the logical signal."""
        self.mapped_logical_signal_path = None

    @property
    def mixer_calibration(self) -> MixerCalibration | None:
        """The mixer calibration assigned to this experiment signal or
        `None` if none is assigned.
        """
        return self.calibration.mixer_calibration if self.is_calibrated() else None

    @mixer_calibration.setter
    def mixer_calibration(self, value):
        if self.is_calibrated():
            self.calibration.mixer_calibration = value
        else:
            self.calibration = SignalCalibration(mixer_calibration=value)

    @property
    def precompensation(self) -> Precompensation | None:
        """The calibration precompensation assigned to this experiment signal
        or `None` if none is assigned.
        """
        return self.calibration.precompensation if self.is_calibrated() else None

    @precompensation.setter
    def precompensation(self, value):
        if self.is_calibrated():
            self.calibration.precompensation = value
        else:
            self.calibration = SignalCalibration(precompensation=value)

    @property
    def oscillator(self) -> Oscillator | None:
        """The oscillator assigned to this experiment signal or `None` if
        none is assigned.
        """
        return self.calibration.oscillator if self.is_calibrated() else None

    @oscillator.setter
    def oscillator(self, value):
        if self.is_calibrated():
            self.calibration.oscillator = value
        else:
            self.calibration = SignalCalibration(oscillator=value)

    @property
    def amplitude(self) -> float | None:
        """The amplitude to multiply all waveforms played on this signal by,
        or `None` to not modify the amplitude.

        Only supported by the SHFQA.

        !!! note
            The amplitude setting applies to all signals on the same channel.
        """
        return self.calibration.amplitude if self.is_calibrated() else None

    @amplitude.setter
    def amplitude(self, value):
        if not self.is_calibrated():
            self.calibration = SignalCalibration(amplitude=value)
        else:
            self.calibration.amplitude = value

    @property
    def port_delay(self) -> float | None:
        """The port delay (in seconds) set on this signal,
        or `None` if none is set.
        """
        return self.calibration.port_delay if self.is_calibrated() else None

    @port_delay.setter
    def port_delay(self, value):
        if not self.is_calibrated():
            self.calibration = SignalCalibration(port_delay=value)
        else:
            self.calibration.port_delay = value

    @property
    def delay_signal(self):
        """The signal delay (in seconds) set on this signal,
        or `None` if none is set.
        """
        return self.calibration.delay_signal if self.is_calibrated() else None

    @delay_signal.setter
    def delay_signal(self, value):
        if not self.is_calibrated():
            self.calibration = SignalCalibration(delay_signal=value)
        else:
            self.calibration.delay_signal = value

    @property
    def voltage_offset(self) -> float | None:
        """The voltage offset set for this signal or `None` if none
        is set.

        Only supported by HDAWG lines.
        """
        return self.calibration.voltage_offset if self.is_calibrated() else None

    @voltage_offset.setter
    def voltage_offset(self, value):
        if self.is_calibrated():
            self.calibration.voltage_offset = value
        else:
            self.calibration = SignalCalibration(voltage_offset=value)

    @property
    def local_oscillator(self):
        """The local oscillator settings assigned to this signal
        or `None` if none is assigned.
        """
        return self.calibration.local_oscillator if self.is_calibrated() else None

    @local_oscillator.setter
    def local_oscillator(self, value):
        if self.is_calibrated():
            self.calibration.local_oscillator = value
        else:
            self.calibration = SignalCalibration(local_oscillator=value)

    @property
    def range(self) -> float | None:
        """The output or input range setting for the signal if set
        or `None` if not set.
        """
        return self.calibration.range if self.is_calibrated() else None

    @range.setter
    def range(self, value):
        if self.is_calibrated():
            self.calibration.range = value
        else:
            self.calibration = SignalCalibration(range=value)

    @property
    def port_mode(self) -> PortMode | None:
        """The port mode for the signal if set or `None` if not set."""
        return self.calibration.port_mode if self.is_calibrated() else None

    @port_mode.setter
    def port_mode(self, value):
        if self.is_calibrated():
            self.calibration.port_mode = value
        else:
            self.calibration = SignalCalibration(port_mode=value)

    @property
    def threshold(self) -> float | list[float] | None:
        """The state discrimination threshold if set or `None` if not set.

        Only supported for acquisition signals on the UHFQA, SHFQA and
        SHFQC.
        """
        return self.calibration.threshold if self.is_calibrated() else None

    @threshold.setter
    def threshold(self, value: float | list[float]):
        if not self.is_calibrated():
            self.calibration = SignalCalibration(threshold=value)
        else:
            self.calibration.threshold = value

    @property
    def amplifier_pump(self) -> AmplifierPump | None:
        """The amplifier pump settings assigned to this signal
        or `None` if none is assigned.
        """
        return self.calibration.amplifier_pump if self.is_calibrated() else None

    @amplifier_pump.setter
    def amplifier_pump(self, value: AmplifierPump):
        if not self.is_calibrated():
            self.calibration = SignalCalibration(amplifier_pump=value)
        else:
            self.calibration.amplifier_pump = value

    @property
    def added_outputs(self):
        return self.calibration.added_outputs if self.is_calibrated() else None

    @added_outputs.setter
    def added_outputs(self, value):
        if not self.is_calibrated():
            self.calibration = SignalCalibration(added_outputs=value)
        else:
            self.calibration.added_outputs = value

    def is_calibrated(self) -> bool:
        """True if calibration has been set for this experiment signal.
        False otherwise.

        Returns:
            is_calibrated:
                True if the signal has calibration set. False otherwise.
        """
        return self.calibration is not None

    def reset_calibration(self, calibration: SignalCalibration | None = None):
        """Reset the calibration and apply the specified new calibration
        if provided.

        Parameters:
            calibration:
                The new calibration to apply or `None` if this signal
                is to be left uncalibrated after the reset.
        """
        self.calibration = calibration

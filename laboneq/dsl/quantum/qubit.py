# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional

from laboneq.core.utilities.dsl_dataclass_decorator import classformatter
from laboneq.dsl.calibration import Calibration, Oscillator, SignalCalibration
from laboneq.dsl.device import LogicalSignalGroup
from laboneq.dsl.device.io_units import LogicalSignal
from laboneq.dsl.enums import ModulationType
from laboneq.dsl.quantum.quantum_element import QuantumElement, SignalType


@classformatter
@dataclass
class QubitParameters:
    #: Resonance frequency of the qubit.
    resonance_frequency: Optional[float] = None
    #: Local oscillator frequency for the qubit drive line.
    drive_lo_frequency: Optional[float] = None
    #: Resonance frequency of the readout resonators used to read the state of the qubit.
    readout_resonator_frequency: Optional[float] = None
    #: Local oscillator frequency for the readout lines.
    readout_lo_frequency: Optional[float] = None
    #: integration delay between readout pulse and data acquisition, defaults to 20 ns.
    readout_integration_delay: Optional[float] = 20e-9
    #: drive power setting, defaults to 10 dBm.
    drive_range: Optional[float] = 10
    #: readout output power setting, defaults to 5 dBm.
    readout_range_out: Optional[float] = 5
    #: readout input power setting, defaults to 10 dBm.
    readout_range_in: Optional[float] = 10
    #: offset voltage for flux control line - defaults to 0.
    flux_offset_voltage: Optional[float] = 0
    #: Free form dictionary of user defined parameters.
    user_defined: dict | None = field(default_factory=dict)

    @property
    def drive_frequency(self) -> float | None:
        """Qubit drive frequency.

        Calculated from `resonance_frequency` and `drive_lo_frequency`,

        Returns:
            Calculated value if both attributes are defined, otherwise `None`.
        """
        try:
            return self.resonance_frequency - self.drive_lo_frequency
        except TypeError:
            return None

    @property
    def readout_frequency(self) -> float | None:
        """Readout baseband frequency.

        Calculated from `readout_resonator_frequency` and `readout_lo_frequency`,

        Returns:
            Calculated value if both attributes are defined, otherwise `None`.
        """
        try:
            return self.readout_resonator_frequency - self.readout_lo_frequency
        except TypeError:
            return None


@classformatter
@dataclass(init=False, repr=True, eq=False)
class Qubit(QuantumElement):
    """A class for a generic two-level Qubit."""

    parameters: QubitParameters

    def __init__(
        self,
        uid: str | None = None,
        signals: dict[str, LogicalSignal] | None = None,
        parameters: QubitParameters | dict[str, Any] | None = None,
    ):
        """
        Initializes a new Qubit.

        Args:
            uid: A unique identifier for the Qubit.
            signals: A mapping of logical signals associated with the qubit.
                Qubit accepts the following keys in the mapping: 'drive', 'measure', 'acquire', 'flux'

                This is so that the Qubit parameters are assigned into the correct signal lines in
                calibration.
            parameters: Parameters associated with the qubit.
                Required for generating calibration and experiment signals via `calibration()` and `experiment_signals()`.
        """
        if parameters is None:
            self.parameters = QubitParameters()
        elif isinstance(parameters, dict):
            self.parameters = QubitParameters(**parameters)
        else:
            self.parameters = parameters
        super().__init__(uid=uid, signals=signals)

    @classmethod
    def from_logical_signal_group(
        cls,
        uid: str,
        lsg: LogicalSignalGroup,
        parameters: QubitParameters | dict[str, Any] | None = None,
    ) -> "Qubit":
        """Qubit from logical signal group.

        Args:
            uid: A unique identifier for the Qubit.
            lsg: Logical signal group.
                Qubit understands the following signal line names:

                    - drive: 'drive', 'drive_line'
                    - measure: 'measure', 'measure_line'
                    - acquire: 'acquire', 'acquire_line'
                    - flux: 'flux', 'flux_line'

                This is so that the Qubit parameters are assigned into the correct signal lines in
                calibration.
            parameters: Parameters associated with the qubit.
        """
        signal_type_map = {
            SignalType.DRIVE: ["drive", "drive_line"],
            SignalType.MEASURE: ["measure", "measure_line"],
            SignalType.ACQUIRE: ["acquire", "acquire_line"],
            SignalType.FLUX: ["flux", "flux_line"],
        }
        if parameters is None:
            parameters = QubitParameters()
        elif isinstance(parameters, dict):
            parameters = QubitParameters(**parameters)
        return cls._from_logical_signal_group(
            uid=uid,
            lsg=lsg,
            parameters=parameters,
            signal_type_map=signal_type_map,
        )

    def calibration(self) -> Calibration:
        """Generate calibration from the qubits parameters and signal lines.

        `Qubit` requires `parameters` for it to be able to produce a calibration object.

        Returns:
            Prefilled calibration object from Qubit parameters.
        """
        calib = {}
        drive_lo = None
        readout_lo = None
        if self.parameters.drive_lo_frequency:
            drive_lo = Oscillator(
                uid=f"{self.uid}_drive_local_osc",
                frequency=self.parameters.drive_lo_frequency,
            )
        if self.parameters.readout_lo_frequency:
            readout_lo = Oscillator(
                uid=f"{self.uid}_readout_local_osc",
                frequency=self.parameters.readout_lo_frequency,
            )

        if "drive" in self.signals:
            sig_cal = SignalCalibration()
            if self.parameters.drive_frequency:
                sig_cal.oscillator = Oscillator(
                    uid=f"{self.uid}_drive_osc",
                    frequency=self.parameters.drive_frequency,
                    modulation_type=ModulationType.HARDWARE,
                )
            sig_cal.local_oscillator = drive_lo
            sig_cal.range = self.parameters.drive_range
            calib[self.signals["drive"]] = sig_cal
        if "measure" in self.signals:
            sig_cal = SignalCalibration()
            if self.parameters.readout_frequency:
                sig_cal.oscillator = Oscillator(
                    uid=f"{self.uid}_measure_osc",
                    frequency=self.parameters.readout_frequency,
                    modulation_type=ModulationType.SOFTWARE,
                )
            sig_cal.local_oscillator = readout_lo
            sig_cal.range = self.parameters.readout_range_out
            calib[self.signals["measure"]] = sig_cal
        if "acquire" in self.signals:
            sig_cal = SignalCalibration()
            if self.parameters.readout_frequency:
                sig_cal.oscillator = Oscillator(
                    uid=f"{self.uid}_acquire_osc",
                    frequency=self.parameters.readout_frequency,
                    modulation_type=ModulationType.SOFTWARE,
                )
            sig_cal.local_oscillator = readout_lo
            sig_cal.range = self.parameters.readout_range_out
            calib[self.signals["acquire"]] = sig_cal
        if "flux" in self.signals:
            calib[self.signals["flux"]] = SignalCalibration(
                voltage_offset=self.parameters.flux_offset_voltage,
            )
        return Calibration(calib)

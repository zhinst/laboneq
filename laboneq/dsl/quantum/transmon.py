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
class TransmonParameters:
    #: Resonance frequency of the qubits g-e transition.
    resonance_frequency_ge: Optional[float] = None
    #: Resonance frequency of the qubits e-f transition.
    resonance_frequency_ef: Optional[float] = None
    #: Local oscillator frequency for the drive signals.
    drive_lo_frequency: Optional[float] = None
    #: Readout resonantor frequency of the qubit.
    readout_resonator_frequency: Optional[float] = None
    #: local oscillator frequency for the readout lines.
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
    def drive_frequency_ge(self) -> float | None:
        """Qubit drive frequency."""
        try:
            return self.resonance_frequency_ge - self.drive_lo_frequency
        except TypeError:
            return None

    @property
    def drive_frequency_ef(self) -> float | None:
        """Qubit drive frequency."""
        try:
            return self.resonance_frequency_ef - self.drive_lo_frequency
        except TypeError:
            return None

    @property
    def readout_frequency(self) -> float | None:
        """Readout baseband frequency."""
        try:
            return self.readout_resonator_frequency - self.readout_lo_frequency
        except TypeError:
            return None


@classformatter
@dataclass(init=False, repr=True, eq=False)
class Transmon(QuantumElement):
    """A class for a superconducting, flux-tuneable Transmon Qubit."""

    parameters: TransmonParameters

    def __init__(
        self,
        uid: str | None = None,
        signals: dict[str, LogicalSignal] | None = None,
        parameters: TransmonParameters | dict[str, Any] | None = None,
    ):
        """
        Initializes a new Transmon Qubit.

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
            self.parameters = TransmonParameters()
        elif isinstance(parameters, dict):
            self.parameters = TransmonParameters(**parameters)
        else:
            self.parameters = parameters
        super().__init__(uid=uid, signals=signals)

    @classmethod
    def from_logical_signal_group(
        cls,
        uid: str,
        lsg: LogicalSignalGroup,
        parameters: TransmonParameters | dict[str, Any] | None = None,
    ) -> "Transmon":
        """Transmon Qubit from logical signal group.

        Args:
            uid: A unique identifier for the Qubit.
            lsg: Logical signal group.
                Transmon Qubit understands the following signal line names:

                    - drive: 'drive', 'drive_line'
                    - drive_ef: 'drive_ef', 'drive_line_ef'
                    - measure: 'measure', 'measure_line'
                    - acquire: 'acquire', 'acquire_line'
                    - flux: 'flux', 'flux_line'

                This is so that the Qubit parameters are assigned into the correct signal lines in
                calibration.
            parameters: Parameters associated with the qubit.
        """
        signal_type_map = {
            SignalType.DRIVE: ["drive", "drive_line"],
            SignalType.DRIVE_EF: ["drive_ef", "drive_line_ef"],
            SignalType.MEASURE: ["measure", "measure_line"],
            SignalType.ACQUIRE: ["acquire", "acquire_line"],
            SignalType.FLUX: ["flux", "flux_line"],
        }
        if parameters is None:
            parameters = TransmonParameters()
        elif isinstance(parameters, dict):
            parameters = TransmonParameters(**parameters)
        return cls._from_logical_signal_group(
            uid=uid,
            lsg=lsg,
            parameters=parameters,
            signal_type_map=signal_type_map,
        )

    def calibration(self, set_local_oscillators=True) -> Calibration:
        """Generate calibration from the parameters and attached signal lines.

        `Qubit` requires `parameters` for it to be able to produce calibration objects.

        Args:
            set_local_oscillators (bool):
                If True, adds local oscillator settings to the calibration.

        Returns:
            calibration:
                Prefilled calibration object from Qubit parameters.
        """
        drive_lo = None
        readout_lo = None
        if set_local_oscillators:
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

        calib = {}
        if "drive" in self.signals:
            sig_cal = SignalCalibration()
            if self.parameters.drive_frequency_ge is not None:
                sig_cal.oscillator = Oscillator(
                    uid=f"{self.uid}_drive_ge_osc",
                    frequency=self.parameters.drive_frequency_ge,
                    modulation_type=ModulationType.HARDWARE,
                )
            sig_cal.local_oscillator = drive_lo
            sig_cal.range = self.parameters.drive_range
            calib[self.signals["drive"]] = sig_cal
        if "drive_ef" in self.signals:
            sig_cal = SignalCalibration()
            if self.parameters.drive_frequency_ef:
                sig_cal.oscillator = Oscillator(
                    uid=f"{self.uid}_drive_ef_osc",
                    frequency=self.parameters.drive_frequency_ef,
                    modulation_type=ModulationType.HARDWARE,
                )
            sig_cal.local_oscillator = drive_lo
            sig_cal.range = self.parameters.drive_range
            calib[self.signals["drive_ef"]] = sig_cal
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
            sig_cal.range = self.parameters.readout_range_in
            sig_cal.port_delay = self.parameters.readout_integration_delay
            calib[self.signals["acquire"]] = sig_cal
        if "flux" in self.signals:
            calib[self.signals["flux"]] = SignalCalibration(
                voltage_offset=self.parameters.flux_offset_voltage,
            )
        return Calibration(calib)

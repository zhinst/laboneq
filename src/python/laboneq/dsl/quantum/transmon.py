# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

# This module and the qubit module are both deprecated.

from __future__ import annotations

from typing import Optional

import attrs

from laboneq.core.utilities.dsl_dataclass_decorator import classformatter
from laboneq.dsl.calibration import Calibration, Oscillator, SignalCalibration
from laboneq.dsl.enums import ModulationType
from laboneq.dsl.quantum.quantum_element import (
    QuantumElement,
    QuantumParameters,
)


@classformatter
@attrs.define(kw_only=True)
class TransmonParameters(QuantumParameters):
    """A class for the parameters of a superconducting, flux-tuneable transmon qubit.

    !!! version-changed "Deprecated in version 2.43.0."

        This class is deprecated and was intended primarily for demonstration purposes.
        Instead of using it write a class that directly inherits from
        [QuantumParameters][laboneq.dsl.quantum.quantum_element.QuantumParameters].
    """

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
    user_defined: dict | None = attrs.field(factory=dict)

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
@attrs.define()
class Transmon(QuantumElement):
    """A class for a superconducting, flux-tuneable Transmon Qubit.

    !!! version-changed "Deprecated in version 2.43.0."

        This class is deprecated and was intended primarily for demonstration purposes.
        Instead of using it write a class that directly inherits from
        [QuantumElement][laboneq.dsl.quantum.quantum_element.QuantumElement].
    """

    PARAMETERS_TYPE = TransmonParameters

    REQUIRED_SIGNALS = (
        "acquire",
        "drive",
        "measure",
    )

    OPTIONAL_SIGNALS = (
        "drive_ef",
        "drive_cr",
        "flux",
    )

    SIGNAL_ALIASES = {
        "acquire_line": "acquire",
        "drive_line": "drive",
        "measure_line": "measure",
        "drive_ef_line": "drive_ef",
        "drive_line_ef": "drive_ef",
        "drive_cr_line": "drive_cr",
        "flux_line": "flux",
    }

    def calibration(self) -> Calibration:
        """Return the experiment calibration for this transmon.

        Calibration for each experiment signal is generated from the transmon
        parameters.

        Returns:
            The experiment calibration.
        """
        drive_lo = None
        readout_lo = None
        readout_oscillator = None

        if self.parameters.drive_lo_frequency is not None:
            drive_lo = Oscillator(
                uid=f"{self.uid}_drive_local_osc",
                frequency=self.parameters.drive_lo_frequency,
            )
        if self.parameters.readout_lo_frequency is not None:
            readout_lo = Oscillator(
                uid=f"{self.uid}_readout_local_osc",
                frequency=self.parameters.readout_lo_frequency,
            )
        if self.parameters.readout_frequency is not None:
            readout_oscillator = Oscillator(
                uid=f"{self.uid}_readout_acquire_osc",
                frequency=self.parameters.readout_frequency,
                modulation_type=ModulationType.AUTO,
            )

        calib = {}
        if "drive" in self.signals:
            sig_cal = SignalCalibration()
            if self.parameters.drive_frequency_ge is not None:
                sig_cal.oscillator = Oscillator(
                    uid=f"{self.uid}_drive_ge_osc",
                    frequency=self.parameters.drive_frequency_ge,
                    modulation_type=ModulationType.AUTO,
                )
            sig_cal.local_oscillator = drive_lo
            sig_cal.range = self.parameters.drive_range
            calib[self.signals["drive"]] = sig_cal
        if "drive_ef" in self.signals:
            sig_cal = SignalCalibration()
            if self.parameters.drive_frequency_ef is not None:
                sig_cal.oscillator = Oscillator(
                    uid=f"{self.uid}_drive_ef_osc",
                    frequency=self.parameters.drive_frequency_ef,
                    modulation_type=ModulationType.AUTO,
                )
            sig_cal.local_oscillator = drive_lo
            sig_cal.range = self.parameters.drive_range
            calib[self.signals["drive_ef"]] = sig_cal
        if "measure" in self.signals:
            sig_cal = SignalCalibration()
            if readout_oscillator:
                sig_cal.oscillator = readout_oscillator
            sig_cal.local_oscillator = readout_lo
            sig_cal.range = self.parameters.readout_range_out
            calib[self.signals["measure"]] = sig_cal
        if "acquire" in self.signals:
            sig_cal = SignalCalibration()
            if readout_oscillator:
                sig_cal.oscillator = readout_oscillator
            sig_cal.local_oscillator = readout_lo
            sig_cal.range = self.parameters.readout_range_in
            sig_cal.port_delay = self.parameters.readout_integration_delay
            calib[self.signals["acquire"]] = sig_cal
        if "flux" in self.signals:
            calib[self.signals["flux"]] = SignalCalibration(
                voltage_offset=self.parameters.flux_offset_voltage,
            )
        return Calibration(calib)

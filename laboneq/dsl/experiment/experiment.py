# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Deque, Dict, List, Optional, Union

from laboneq.core.exceptions import LabOneQException
from laboneq.core.types.enums import DSLVersion
from laboneq.dsl.calibration.calibration import Calibration
from laboneq.dsl.device.io_units.logical_signal import LogicalSignal
from laboneq.dsl.enums import (
    AcquisitionType,
    AveragingMode,
    ExecutionType,
    RepetitionMode,
)
from laboneq.dsl.experiment.pulse import Pulse

from .experiment_signal import ExperimentSignal
from .section import AcquireLoopNt, AcquireLoopRt, Case, Match, Section, Sweep

if TYPE_CHECKING:
    from .. import Parameter

experiment_id = 0


def experiment_id_generator():
    global experiment_id
    retval = f"exp_{experiment_id}"
    experiment_id += 1
    return retval


@dataclass(init=True, repr=True, order=True)
class Experiment:
    uid: str = field(default_factory=experiment_id_generator)
    signals: Union[Dict[str, ExperimentSignal], List[ExperimentSignal]] = field(
        default_factory=dict
    )
    version: DSLVersion = field(default=DSLVersion.V3_0_0)
    epsilon: float = field(default=0.0)
    sections: List[Section] = field(default_factory=list)
    _section_stack: Deque[Section] = field(
        default_factory=deque, repr=False, compare=False
    )

    def __post_init__(self):
        if self.signals is not None and isinstance(self.signals, List):
            signals_dict = {}
            for s in self.signals:
                if isinstance(s, str):
                    signals_dict[s] = ExperimentSignal(uid=s)
                else:
                    signals_dict[s.uid] = s
            self.signals = signals_dict

    def add_signal(
        self, uid: str = None, connect_to: Union[LogicalSignal, str] = None
    ) -> ExperimentSignal:
        """Add an experiment signal to the experiment.

        :param uid: The unique id of the new experiment signal (optional).
        :type uid: UID = str
        :param connect_to:
            The `LogicalSignal` this experiment signal shall be connected to.
            Defaults to None, meaning that there is no connection defined yet.
        :type connect_to: :class:`~.LogicalSignal`, optional

        :return: The created and added signal.
        :rtype: :class:`~.ExperimentSignal`

        .. seealso:: :func:`map_signal`
        """
        if uid is not None and uid in self.signals.keys():
            raise LabOneQException(f"Signal with id {uid} already exists.")
        signal = ExperimentSignal(uid=uid, map_to=connect_to)
        self.signals[uid] = signal
        return signal

    def add(self, section: Section):
        """Add a sweep, a section or an acquire loop to the experiment.
        :param section: The object to add.
        """
        self._add_section_to_current_section(section)

    @property
    def experiment_signals_uids(self):
        """A list of experiment signal uids defined in this experiment."""
        return self.signals.keys

    def list_experiment_signals(self) -> List[ExperimentSignal]:
        """A list of experiment signals defined in this experiment."""
        return list(self.signals.values())

    def is_experiment_signal(self, uid: str) -> bool:
        """Check if an experiment signal is defined for this experiment.

        :param uid: The unique id of the experiment signal to check for.
        :type uid: UID = str
        :return (bool): `True` if the experiment signal is defined in this
            experiment, `False` otherwise.
        """
        return uid in self.signals.keys()

    def map_signal(
        self, experiment_signal_uid: str, logical_signal: Union[LogicalSignal, str]
    ):
        """Connect an experiment signal to a logical signal.

        In order to relate an experiment signal to a logical signal defined in a
        device setup (:class:`~.DeviceSetup`), you need to make
        a connection between these two types of signals.

        :param experiment_signal_uid: The unique id of the experiment signal to
            be connected.
        :type experiment_signal_uid: `UID = str`
        :param logical_signal: The logical signal to connect to.
        :type logical_signal: :class:`~.LogicalSignal`

        .. seealso:: :func:`add_signal`
        """
        if not self.is_experiment_signal(experiment_signal_uid):
            raise LabOneQException(
                f"Unknown experiment signal: {experiment_signal_uid}. Call experiment.experiment_signals to get a "
                f"list of experiment signals defined in this experiment. Call experiment.add_signal() to add a signal "
                f"prior to connect id to a LogicalSignal. "
            )

        self.signals[experiment_signal_uid].map(logical_signal)

    def reset_signal_map(self, signal_map: Dict[str, Union[LogicalSignal, str]] = None):
        """Reset i.e. disconnect all defined signal connections."""

        for signal in self.signals.values():
            signal.disconnect()
        if signal_map:
            self.set_signal_map(signal_map)

    @property
    def signal_mapping_status(self) -> Dict[str, Any]:
        """Get an overview of the signal mapping.

        :return: A dictionary with entries for:

            - `is_all_mapped`:
               `True` if all experiment signals are mapped to a logical signal, `False` otherwise.
            - `mapped_signals`:
               A list of experiment signal uids that have a mapping to a logical signal.
            - `not_mapped_signals`:
               A list of experiment signal uids that have no mapping to a logical signal.
        """
        is_all_mapped = True
        mapped_signals = list()
        not_mapped_signals = list()

        for signal in self.signals.values():
            if signal.is_mapped():
                mapped_signals.append(signal.uid)
            else:
                is_all_mapped = False
                not_mapped_signals.append(signal.uid)
        return {
            "is_all_mapped": is_all_mapped,
            "mapped_signals": mapped_signals,
            "not_mapped_signals": not_mapped_signals,
        }

    def get_signal_map(self) -> Dict[str, str]:
        return {
            signal.uid: signal.mapped_logical_signal_path
            for signal in self.signals.values()
            if signal.is_mapped()
        }

    def set_signal_map(self, signal_map: Dict[str, Union[LogicalSignal, str]]):
        for signal_uid, logical_signal_uid in signal_map.items():
            if signal_uid not in self.signals.keys():
                self._signal_not_found_error(signal_uid, "Cannot apply signal map.")

            self.signals[signal_uid].map(to=logical_signal_uid)

    # Calibration ....................................

    def _signal_not_found_error(self, signal_uid: str, msg: str = None):
        if msg is None:
            msg = ""
        raise LabOneQException(
            f"Signal with id '{signal_uid}' not found in experiment{': ' if msg else '.'}{msg}"
        )

    def set_calibration(self, calibration: Calibration):
        for signal_uid, calib_item in calibration.calibration_items.items():
            if calib_item is not None:
                if signal_uid not in self.signals.keys():
                    self._signal_not_found_error(
                        signal_uid, "Cannot apply experiment signal calibration."
                    )
                self.signals[signal_uid].calibration = calib_item

    def get_calibration(self):
        from ..calibration import Calibration

        experiment_signals_calibration = dict()
        for sig in self.signals.values():
            experiment_signals_calibration[sig.uid] = (
                sig.calibration if sig.is_calibrated() else None
            )

        calibration = Calibration(calibration_items=experiment_signals_calibration)
        return calibration

    def reset_calibration(self, calibration=None):
        try:
            signals = self.signals.values()
        except AttributeError:
            signals = self.signals
        for sig in signals:
            sig.reset_calibration()
        if calibration:
            self.set_calibration(calibration)

    def list_calibratables(self):
        from ..calibration import Calibratable

        calibratables = dict()
        for signal in self.signals.values():
            if isinstance(signal, Calibratable):
                calibratables[signal.uid] = signal.create_info()
        return calibratables

    # Operations .....................................

    def set_node(self, path: str, value: Any):
        """Set the value of an instrument node.

        Args:
            path: Path to the node whose value should be set.
            value: Value that should be set.

        .. versionchanged:: 2.0
            Method name renamed from `set` to `set_node`.
            Removed `key` argument.
        """
        current_section = self._peek_section()
        current_section.set(path=path, value=value)

    def play(
        self,
        signal,
        pulse,
        amplitude=None,
        phase=None,
        increment_oscillator_phase=None,
        set_oscillator_phase=None,
        length=None,
        pulse_parameters: Optional[Dict[str, Any]] = None,
        precompensation_clear: Optional[bool] = None,
        marker=None,
    ):
        """Play a pulse.

        :param signal: The unique id of the signal to play the pulse on.
        :type signal: `str`
        :param pulse: The pulse description of the pulse to be played.
        :type pulse: :class:`~.experiment.pulse.Pulse`
        :param amplitude: Amplitude the pulse shall be played with. Defaults to
            `None`, meaning that the pulse is played as is.
        :type amplitude: `float` or :class:`~.dsl.parameter.Parameter`, optional
        :param length: Length for which the pulse shall be played. Defaults to
            `None`, meaning that the pulse is played for its whole length.
        :type length: `float` or :class:`~.dsl.parameter.Parameter`, optional
        :param phase: The desired baseband phase (baseband rotation) with which the
            pulse shall be played. Given in radians, defaults to `None`,
            meaning that the pulse is played with its phase as defined.
        :type phase: `float`, optional
        :param set_oscillator_phase: The desired oscillator phase at the start of the played pulse, in radians.
            The phase setting affects the pulse played in this command, and all following pulses.
            Defaults to `None`, meaning no change is made and the phase remains continuous.
        :type set_oscillator_phase: `float`, optional
        :param increment_oscillator_phase: The desired phase increment of the oscillator phase
            at the start of the played pulse, in radians.
            The new, incremented phase affects the pulse played in this command, and all following pulses.
            Defaults to `None`, meaning no change is made and the phase remains continuous.
        :type increment_oscillator_phase: `float`, optional
        :param pulse_parameters: Dictionary with user pulse function parameters (re)binding.
        :type pulse_parameters: `dict`, optional
        :param marker: Dictionary with markers to play. Example: `marker={"marker1": {"enable": True}}`
        :type marker: `dict`, optional

        """
        current_section = self._peek_section()
        current_section.play(
            signal=signal,
            pulse=pulse,
            amplitude=amplitude,
            phase=phase,
            increment_oscillator_phase=increment_oscillator_phase,
            set_oscillator_phase=set_oscillator_phase,
            length=length,
            pulse_parameters=pulse_parameters,
            precompensation_clear=precompensation_clear,
            marker=marker,
        )

    def delay(
        self,
        signal: str,
        time: Union[float, Parameter],
        precompensation_clear: Optional[bool] = None,
    ):
        """Delay execution of next operation on the given experiment signal.

        :param signal: The unique id of the signal to delay execution of next
            operation.
        :param time: The delay time in seconds. The parameter can either be
            given as a float or as a sweep parameter
            (:class:`~.dsl.parameter.Parameter`).
        :param precompensation_clear: Clear the precompensation filter during this delay.
        """
        current_section = self._peek_section()
        current_section.delay(
            signal=signal, time=time, precompensation_clear=precompensation_clear
        )

    def reserve(self, signal):
        """Reserves an experiment signal for the active section.

        Reserving an experiment signal in a section means that if there is no
        operation defined on that signal, it is not available for other sections
        as long as the active section is scoped.

        :param signal: The unique id of the signal to be reserved in the active
            section.
        :type signal: `str`
        """
        current_section = self._peek_section()
        current_section.reserve(signal=signal)

    def acquire(
        self,
        signal: str,
        handle: str,
        kernel: Pulse = None,
        length: float = None,
        pulse_parameters: Optional[Dict[str, Any]] = None,
    ):
        """Acquire a signal and make it available in :class:`Result`.

        Args:
            signal: The input signal to acquire data on.
            handle: A unique identifier string that allows to retrieve the
                acquired data in the :class:`Result` object.
            kernel: Pulse for filtering the acquired signal.
            length: Integration length for spectroscopy mode.
            pulse_parameters: Dictionary with user pulse function parameters (re)binding.
        """
        current_section = self._peek_rt_section()
        current_section.acquire(
            signal=signal,
            handle=handle,
            kernel=kernel,
            length=length,
            pulse_parameters=pulse_parameters,
        )

    def measure(
        self,
        acquire_signal: str,
        handle: str,
        integration_kernel: Optional[Pulse] = None,
        integration_kernel_parameters: Optional[Dict[str, Any]] = None,
        integration_length: Optional[float] = None,
        measure_signal: Optional[str] = None,
        measure_pulse: Optional[Pulse] = None,
        measure_pulse_length: Optional[float] = None,
        measure_pulse_parameters: Optional[Dict[str, Any]] = None,
        measure_pulse_amplitude: Optional[float] = None,
        acquire_delay: Optional[float] = None,
        reset_delay: Optional[float] = None,
    ):
        """
        Execute a measurement in the experiment - contains the optional playback of a measurement pulse, the signal acquisition and an optional delay after the signal acquisition.

        :param acquire_signal: A string that specifies the signal to acquire.
        :type acquire_signal: str
        :param handle: A string that specifies the handle of the acquisition.
        :type handle: str
        :param integration_kernel: An optional Pulse object that specifies the kernel for integration.
        :type integration_kernel: Optional[Pulse]
        :param integration_kernel_parameters: An optional dictionary that contains parameters for the integration kernel.
        :type integration_kernel_parameters: Optional[Dict[str, Any]]
        :param integration_length: An optional float that specifies the integration length.
        :type integration_length: Optional[float]
        :param measure_signal: An optional string that specifies the signal to measure.
        :type measure_signal: Optional[str]
        :param measure_pulse: An optional Pulse object that specifies the pulse for measurement.
        :type measure_pulse: Optional[Pulse]
        :param measure_pulse_length: An optional float that specifies the length of the measurement pulse.
        :type measure_pulse_length: Optional[float]
        :param measure_pulse_parameters: An optional dictionary that contains parameters for the measurement pulse.
        :type measure_pulse_parameters: Optional[Dict[str, Any]]
        :param measure_pulse_amplitude: An optional float that specifies the amplitude of the measurement pulse.
        :type measure_pulse_amplitude: Optional[float]
        :param acquire_delay: An optional float that specifies the delay between the acquisition and the measurement.
        :type acquire_delay: Optional[float]
        :param reset_delay: An optional float that specifies the delay after the acquisition to allow for state relaxation or signal processing.
        :type reset_delay: Optional[float]
        """

        if measure_signal is None and not isinstance(acquire_signal, str):
            raise ValueError("acquire_signal must be specified.")

        if isinstance(measure_signal, str) and not isinstance(acquire_signal, str):
            raise ValueError("acquire_signal must be specified.")

        current_section = self._peek_section()

        if measure_signal is None and isinstance(acquire_signal, str):
            current_section.acquire(
                signal=acquire_signal,
                handle=handle,
                length=integration_length,
            )

        elif isinstance(measure_signal, str) and isinstance(acquire_signal, str):
            current_section.play(
                signal=measure_signal,
                pulse=measure_pulse,
                amplitude=measure_pulse_amplitude,
                length=measure_pulse_length,
                pulse_parameters=measure_pulse_parameters,
            )

            if acquire_delay is not None:
                current_section.delay(
                    signal=acquire_signal,
                    time=acquire_delay,
                )
            current_section.acquire(
                signal=acquire_signal,
                handle=handle,
                kernel=integration_kernel,
                length=integration_length,
                pulse_parameters=integration_kernel_parameters,
            )

        if reset_delay is not None:
            current_section.delay(
                signal=acquire_signal,
                time=reset_delay,
            )

    def call(self, func_name, **kwargs):
        """Add a callback function in the execution of the experiment.

        The callback is called by the LabOne Q software as part of executing the
        experiment and in sequence with the other experiment operations.

        Args:
            func_name: The callback function.
            kwargs: These arguments are passed to the callback function as is.
        """
        current_section = self._peek_section()
        current_section.call(func_name=func_name, **kwargs)

    # Sections .......................................

    def sweep(
        self,
        parameter,
        execution_type=None,
        uid=None,
        alignment=None,
        reset_oscillator_phase=False,
    ):
        """Define a sweep section.

        Sections need to open a scope in the following way::

            with exp.sweep(...):
                # here come the operations that shall be executed in the sweep section

        :note: A near time section cannot be defined in the scope of a real
            time section.

        Args:
            uid: The unique ID for this section.
            parameter: The sweep parameter(s) that is used in this section.
                The argument can be given as a single sweep parameter or a list
                of sweep parameters of equal length. If multiple sweep parameters are given, the
                parameters are executed in parallel in this sweep loop.
            execution_type: Defines if the sweep is executed in near time or
                real time. Defaults to :class:`.~ExecutionType.NEAR_TIME`.
            alignment: Alignment of the operations in the section. Defaults to
                :class:`.~SectionAlignment.LEFT`.
            reset_oscillator_phase: When True, reset all oscillators at the start of
                each step.

        """
        parameters = parameter if isinstance(parameter, list) else [parameter]
        return Experiment._SweepSectionContext(
            self,
            uid=uid,
            parameters=parameters,
            execution_type=execution_type,
            alignment=alignment,
            reset_oscillator_phase=reset_oscillator_phase,
        )

    class _SweepSectionContext:
        def __init__(
            self,
            experiment,
            uid,
            parameters,
            execution_type,
            alignment,
            reset_oscillator_phase,
        ):
            self.exp = experiment
            args = {"parameters": parameters}
            if uid is not None:
                args["uid"] = uid
            if execution_type is not None:
                args["execution_type"] = execution_type
                if execution_type is ExecutionType.NEAR_TIME:
                    try:
                        parent_section = experiment._peek_rt_section()
                    except LabOneQException:
                        pass
                    else:
                        if parent_section is not None:
                            raise LabOneQException(
                                "Near-time sweep not allowed within real-time context"
                            )

            if alignment is not None:
                args["alignment"] = alignment

            if reset_oscillator_phase is not None:
                args["reset_oscillator_phase"] = reset_oscillator_phase

            self.sweep = Sweep(**args)

        def __enter__(self):
            self.exp._push_section(self.sweep)
            return self.sweep

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.exp._pop_and_add_section()

    def acquire_loop_nt(self, count, averaging_mode=AveragingMode.CYCLIC, uid=None):
        """Define an acquire section with averaging in near time.

        Sections need to open a scope in the following way::

            with exp.acquire_loop_nt(...):
                # here come the operations that shall be executed in the acquire_loop_nt section

        :note: A near time section cannot be defined in the scope of a real
            time section.

        Args:
            uid: The unique ID for this section.
            count: The number of acquire iterations.
            averaging_mode: The mode of how to average the acquired data.
                Defaults to :attr:`~.AveragingMode.CYCLIC`.
        """
        return Experiment._AcquireLoopNtSectionContext(
            self, uid=uid, count=count, averaging_mode=averaging_mode
        )

    class _AcquireLoopNtSectionContext:
        def __init__(self, experiment, count, averaging_mode, uid=None):
            self.exp = experiment
            self.acquire_loop = AcquireLoopNt(
                uid=uid, count=count, averaging_mode=averaging_mode
            )

        def __enter__(self):
            self.exp._push_section(self.acquire_loop)
            return self.acquire_loop

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.exp._pop_and_add_section()

    def acquire_loop_rt(
        self,
        count,
        averaging_mode=AveragingMode.CYCLIC,
        repetition_mode=RepetitionMode.FASTEST,
        repetition_time=None,
        acquisition_type=AcquisitionType.INTEGRATION,
        uid=None,
        reset_oscillator_phase=False,
    ):
        """Define an acquire section with averaging in real time.

        Sections need to open a scope in the following way::

            with exp.acquire_loop_rt(...):
                # here come the operations that shall be executed in the acquire_loop_rt section

        :note: A near time section cannot be defined in the scope of a real
            time section.

        Args:
            uid: The unique ID for this section.
            count: The number of acquire iterations.
            averaging_mode: The mode of how to average the acquired data.
                Defaults to :attr:`.AveragingMode.CYCLIC`.
                Further options: :attr:`.AveragingMode.SEQUENTIAL` and :attr:`.AveragingMode.SINGLE_SHOT`.
                Single shot measurements are always averaged in cyclic mode.
            repetition_mode: Defines the shot repetition mode. Defaults to
                :attr:`.RepetitionMode.FASTEST`. Further options are
                :attr:`.RepetitionMode.CONSTANT` and :attr:`.RepetitionMode.AUTO`.
            repetition_time: This is the shot repetition time in sec. This
                argument is only required and valid if :param:repetition_mode is
                :class:`RepetitionMode.CONSTANT`. The parameter can either be
                given as a float or as a sweep parameter (:class:`~.dsl.parameter.Parameter`).
            acquisition_type: This is the acquisition type.
                Defaults to :attr:`.AcquisitionType.INTEGRATION`. Further options are
                :attr:`.AcquisitionType.SPECTROSCOPY`, :attr:`.AcquisitionType.DISCRIMINATION` and :attr:`.AcquisitionType.RAW`.
            reset_oscillator_phase: When True, the phase of every oscillator is reset at
                the start of the each step of the acquire loop.
        """
        return Experiment._AcquireLoopRtSectionContext(
            self,
            uid=uid,
            count=count,
            averaging_mode=averaging_mode,
            repetition_mode=repetition_mode,
            repetition_time=repetition_time,
            acquisition_type=acquisition_type,
            reset_oscillator_phase=reset_oscillator_phase,
        )

    class _AcquireLoopRtSectionContext:
        def __init__(
            self,
            experiment,
            count,
            averaging_mode,
            repetition_mode,
            repetition_time,
            acquisition_type,
            reset_oscillator_phase,
            uid=None,
        ):
            self.exp = experiment
            kwargs = dict(
                count=count,
                averaging_mode=averaging_mode,
                repetition_mode=repetition_mode,
                repetition_time=repetition_time,
                acquisition_type=acquisition_type,
                reset_oscillator_phase=reset_oscillator_phase,
            )
            if uid is not None:
                kwargs["uid"] = uid
            self.acquire_shots = AcquireLoopRt(**kwargs)

        def __enter__(self):
            self.exp._push_section(self.acquire_shots)
            return self.acquire_shots

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.exp._pop_and_add_section()

    def for_(self, timing, parameters=None, count=0, uid=None):
        return Experiment._ForSectionContext(
            self, timing=timing, parameters=parameters, count=count, uid=uid
        )

    class _ForSectionContext:
        def __init__(self, experiment, timing, uid, parameters=None, count=0):
            if parameters is None:
                parameters = []
            self.exp = experiment

            if parameters and not count:
                self.average = None
                if uid is None:
                    self.sweep = Sweep(
                        parameters=parameters, reset_oscillator_phase=False
                    )
                else:
                    self.sweep = Sweep(
                        uid=uid, parameters=parameters, reset_oscillator_phase=False
                    )
                self.sweep.execution_type = timing
            elif count and not parameters:
                self.sweep = None
                args = {"count": count}
                if uid is not None:
                    args["uid"] = uid
                if timing == ExecutionType.NEAR_TIME:
                    self.average = AcquireLoopNt(**args)
                else:
                    self.average = AcquireLoopRt(**args)
            else:
                raise LabOneQException(
                    "Invalid parameters: Either use kwarg 'count' or 'parameters', but not both and not none."
                )

        def __enter__(self):
            self.exp._push_section(self.sweep if self.sweep else self.average)

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.exp._pop_and_add_section()

    def section(
        self,
        length=None,
        alignment=None,
        uid=None,
        on_system_grid=None,
        play_after: Optional[Union[str, List[str]]] = None,
        trigger: Optional[Dict[str, Dict[str, int]]] = None,
    ):
        """Define an section for scoping operations.

        Sections need to open a scope in the following way::

            with exp.section(...):
                # here come the operations that shall be executed in the section

        :note: A near time section cannot be defined in the scope of a real
            time section.

        Args:
            uid: The unique ID for this section.
            length: The minimal duration of the section in seconds. The
                scheduled section might be slightly longer, as its length is
                rounded to the next multiple of the section timing grid.
                Defaults to `None` which means that the section length is
                derived automatically from the contained operations.
                The parameter can either be given as a float or as a sweep
                parameter (:class:`~.Parameter`).
            alignment: Alignment of the operations in the section. Defaults to
                :class:`~.SectionAlignment.LEFT`.
            play_after: Play this section after the end of the section(s) with the
                given ID(s) (single string or list of strings). Defaults to None.
            trigger: Play a pulse a trigger pulse for the duration of this section.
                See below for details.
            on_system_grid: If True, the section boundaries are always rounded to the
                system grid, even if the signals would allow for tighter alignment.

        The individual trigger (a.k.a marker) ports on the device are addressed via the
        experiment signal that is mapped to the corresponding analog port.
        For playing trigger pulses, pass a dictionary via the `trigger` argument. The
        keys of the dictionary must be an ID of an :class:`~.ExperimentSignal`. Each
        value is another ``dict`` of the form: ::

            {"state": value}

        ``value`` is a bit field that enables the individual trigger signals (on the
        devices that feature more than a single one).

        ..  code-block:: python

            {"state": 1}  # raise trigger signal 1
            {"state": 0b10}  # raise trigger signal 2 (on supported devices)
            {"state": 0b11}  # raise both trigger signals

        As a more complete example, to fire a trigger pulse on the first port associated
        with signal ``"drive_line"``, call ::

            with exp.section(..., trigger={"drive_line": {"state": 0b01}}):
                ...

        When trigger signals on the same signal are issued in nested sections, the values
        are ORed.

        .. versionchanged:: 2.0.0

            Removed deprecated `offset` argument.
        """
        return Experiment._SectionSectionContext(
            self,
            uid=uid,
            length=length,
            alignment=alignment,
            play_after=play_after,
            trigger=trigger,
            on_system_grid=on_system_grid,
        )

    class _SectionSectionContext:
        def __init__(
            self,
            experiment,
            uid,
            length=None,
            alignment=None,
            play_after=None,
            trigger=None,
            on_system_grid=None,
        ):
            self.exp = experiment
            args = {}
            if uid is not None:
                args["uid"] = uid
            if length is not None:
                args["length"] = length
            if alignment is not None:
                args["alignment"] = alignment
            if play_after is not None:
                args["play_after"] = play_after
            if trigger is not None:
                args["trigger"] = trigger
            if on_system_grid is not None:
                args["on_system_grid"] = on_system_grid

            self.section = Section(**args)

        def __enter__(self):
            self.exp._push_section(self.section)
            return self.section

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.exp._pop_and_add_section()

    def _push_section(self, section):
        if section.execution_type is None:
            if self._section_stack:
                parent_section = self._peek_section()
                execution_type = parent_section.execution_type
            else:
                execution_type = ExecutionType.NEAR_TIME
            section.execution_type = execution_type
        self._section_stack.append(section)

    def _pop_and_add_section(self):
        if not self._section_stack:
            raise LabOneQException(
                "Internal error: Section stack should not be empty. Unbalanced push/pop."
            )
        section = self._section_stack.pop()
        self._add_section_to_current_section(section)

    def _add_section_to_current_section(self, section):
        if not self._section_stack:
            self.sections.append(section)
        else:
            current_section = self._section_stack[-1]
            current_section.add(section)

    def _peek_section(self):
        if not self._section_stack:
            raise LabOneQException(
                "No section in experiment. Use 'with your_exp.section(...):' to create a section scope first."
            )
        return self._section_stack[-1]

    def _peek_rt_section(self):
        if not self._section_stack:
            raise LabOneQException(
                "No section in experiment. Use 'with your_exp.section(...):' to create a section scope first."
            )
        for s in reversed(self._section_stack):
            if s.execution_type == ExecutionType.REAL_TIME:
                return s
        raise LabOneQException(
            "No surrounding realtime section in experiment. Use 'with your_exp.acquire_loop_rt(...):' to create a section scope first."
        )

    def accept_section_visitor(self, visitor, sections=None):
        if sections is None:
            sections = self.sections
        for section in sections:
            visitor(section)
            self.accept_section_visitor(visitor, section.sections)

    def match_local(
        self,
        handle: str,
        uid: str = None,
        play_after: Optional[Union[str, List[str]]] = None,
    ):
        """Define a section which switches between different child sections based
        on a QA measurement on an SHFQC.

        Match needs to open a scope in the following way::

            with exp.match_local(...):
                # here come the different branches to be selected

        :note: Only subsections of type ``Case`` are allowed.

        Args:
            uid: The unique ID for this section.
            handle: A unique identifier string that allows to retrieve the
                acquired data.
            play_after Play this section after the end of the section(s) with the
                given ID(s) (single string or list of strings). Defaults to None.

        """
        return Experiment._MatchSectionContext(
            self, uid=uid, handle=handle, play_after=play_after, local=True
        )

    def match_global(
        self,
        handle: str,
        uid: str = None,
        play_after: Optional[Union[str, List[str]]] = None,
    ):
        """Define a section which switches between different child sections based
        on a QA measurement via the PQSC.

        Match needs to open a scope in the following way::

            with exp.match_global(...):
                # here come the different branches to be selected

        :note: Only subsections of type ``Case`` are allowed.

        Args:
            uid: The unique ID for this section.
            handle: A unique identifier string that allows to retrieve the
                acquired data.
            play_after Play this section after the end of the section(s) with the
                given ID(s) (single string or list of strings). Defaults to None.

        """
        raise LabOneQException("Global feedback via the PQSC is not yet supported.")

    class _MatchSectionContext:
        def __init__(
            self,
            experiment,
            uid,
            handle,
            local,
            play_after=None,
        ):
            self.exp = experiment
            args = {"handle": handle}
            if uid is not None:
                args["uid"] = uid
            if play_after is not None:
                args["play_after"] = play_after
            args["local"] = local

            self.section = Match(**args)

        def __enter__(self):
            self.exp._push_section(self.section)
            return self.section

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.exp._pop_and_add_section()

    def case(self, state: int, uid: str = None):
        """Define a section which plays after matching with the given value to the
        result of a QA measurement.

        Case needs to open a scope in the following way::

            with exp.case(...):
                # here come the operations that shall be executed in the section

        :note: No subsections are allowed, only :meth:`play` and :meth:`delay`.

        Args:
            uid: The unique ID for this section.
            state: The state that this section is executed for.
        """
        if not isinstance(self._peek_section(), Match):
            raise LabOneQException("Case section must be inside a Match section")
        return Experiment._CaseSectionContext(
            self,
            uid=uid,
            state=state,
        )

    class _CaseSectionContext:
        def __init__(
            self,
            experiment,
            uid,
            state,
        ):
            self.exp = experiment
            args = {"state": state}
            if uid is not None:
                args["uid"] = uid

            self.section = Case(**args)

        def __enter__(self):
            self.exp._push_section(self.section)
            return self.section

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.exp._pop_and_add_section()

    @staticmethod
    def load(filename):
        from ..serialization import Serializer

        # TODO ErC: Error handling
        return Serializer.from_json_file(filename, Experiment)

    def save(self, filename):
        from ..serialization import Serializer

        # TODO ErC: Error handling
        Serializer.to_json_file(self, filename)

    def load_signal_map(self, filename):
        from ..serialization import Serializer

        # TODO ErC: Error handling
        signal_map = Serializer.from_json_file(filename, dict)
        self.set_signal_map(signal_map)

    def save_signal_map(self, filename):
        from ..serialization import Serializer

        # TODO ErC: Error handling
        Serializer.to_json_file(self.get_signal_map(), filename)

    @staticmethod
    def _all_subsections(section):
        retval = [section]
        for s in section.sections:
            retval.extend(Experiment._all_subsections(s))

        return retval

    def all_sections(self):
        retval = []
        for s in self.sections:
            retval.extend(Experiment._all_subsections(s))
        return retval

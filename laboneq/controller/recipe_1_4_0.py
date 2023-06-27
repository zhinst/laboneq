# Copyright 2019 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from marshmallow import EXCLUDE, Schema, fields, post_load

from .recipe_enums import (
    AcquisitionType,
    NtStepKey,
    RefClkType,
    SignalType,
    TriggeringMode,
)
from .util import LabOneQControllerException


class QCCSSchema(Schema):
    @post_load
    def from_json(self, data, **kwargs):
        return self.Data(**data)


class Server(QCCSSchema):
    class Meta:
        fields = ("server_uid", "host", "port", "api_level")
        ordered = True

    @dataclass
    class Data:
        server_uid: str
        host: str
        port: int
        api_level: int

    server_uid = fields.Str()
    host = fields.Str()
    port = fields.Integer()
    api_level = fields.Integer()


class DriverOption(fields.Field):
    class Meta:
        fields = ("parameter_name", "value")
        ordered = False

    @dataclass
    class Data:
        key: str
        value: str

    key = fields.Str(required=True)
    value = fields.Str(required=True)


class Device(QCCSSchema):
    class Meta:
        fields = ("device_uid", "driver", "options")
        ordered = True
        unknown = EXCLUDE

    @dataclass
    class Data:
        device_uid: str
        driver: str
        options: list[DriverOption] | None = None

        def _get_option(self, key):
            for option in self.options:
                if key == option["parameter_name"]:
                    return option["value"]
            return None

        @property
        def serial(self):
            return self._get_option("serial")

        @property
        def server_uid(self):
            return self._get_option("server_uid")

        def __str__(self):
            serial = self.serial
            if serial is not None:
                return f"{self.driver}:{serial}"
            else:
                return self.driver

    device_uid = fields.Str()
    driver = fields.Str()
    options = fields.List(DriverOption(), required=False)


class Gains(QCCSSchema):
    class Meta:
        fields = ("diagonal", "off_diagonal")
        ordered = True

    @dataclass
    class Data:
        diagonal: float
        off_diagonal: float

    diagonal = fields.Float()
    off_diagonal = fields.Float()


class IO(QCCSSchema):
    class Meta:
        fields = (
            "channel",
            "enable",
            "modulation",
            "oscillator",
            "oscillator_frequency",
            "offset",
            "gains",
            "range",
            "range_unit",
            "precompensation",
            "lo_frequency",
            "port_mode",
            "port_delay",
            "scheduler_port_delay",
            "delay_signal",
            "marker_mode",
            "amplitude",
        )
        ordered = True

    @dataclass
    class Data:
        channel: int
        enable: bool | None = None
        modulation: bool | None = None
        oscillator: int | None = None
        oscillator_frequency: int | None = None
        offset: float | None = None
        gains: Gains | None = None
        range: float | None = None
        range_unit: str | None = None
        precompensation: dict[str, dict] | None = None
        lo_frequency: Any | None = None
        port_mode: str | None = None
        port_delay: Any | None = None
        scheduler_port_delay: float = 0.0
        delay_signal: float | None = None
        marker_mode: str | None = None
        amplitude: Any | None = None

    channel = fields.Integer()
    enable = fields.Boolean(required=False)
    modulation = fields.Boolean(required=False)
    oscillator = fields.Integer(required=False)
    oscillator_frequency = fields.Integer(required=False)
    offset = fields.Float(required=False)
    gains = fields.Nested(Gains, required=False)
    range = fields.Float(required=False)
    range_unit = fields.Str(required=False)
    precompensation = fields.Dict(required=False)
    lo_frequency = fields.Raw(required=False)
    port_mode = fields.Str(required=False)
    port_delay = fields.Raw(required=False)
    scheduler_port_delay = fields.Float(required=False)
    delay_signal = fields.Float(required=False)
    marker_mode = fields.Str(required=False)
    amplitude = fields.Raw(required=False)


class SignalTypeField(fields.Field):
    def _serialize(self, value, attr, obj, **kwargs):
        return value.name

    def _deserialize(self, value, attr, data, **kwargs):
        return SignalType[value.upper()]


class AWG(QCCSSchema):
    class Meta:
        fields = (
            "awg",
            "signal_type",
            "qa_signal_id",
            "command_table_match_offset",
            "feedback_register",
        )
        ordered = False

    @dataclass
    class Data:
        awg: int
        signal_type: SignalType = SignalType.SINGLE
        qa_signal_id: str | None = None
        command_table_match_offset: int | None = None
        feedback_register: int | None = None

    awg = fields.Integer()
    signal_type = SignalTypeField()
    qa_signal_id = fields.Str(required=False, allow_none=True)
    command_table_match_offset = fields.Integer(required=False, allow_none=True)
    feedback_register = fields.Integer(required=False, allow_none=True)


class Port(QCCSSchema):
    class Meta:
        fields = ("port", "device_uid")
        ordered = True

    @dataclass
    class Data:
        port: int
        device_uid: str

    port = fields.Integer()
    device_uid = fields.Str()


class Measurement(QCCSSchema):
    class Meta:
        fields = ("length", "channel")

    @dataclass
    class Data:
        length: int
        channel: int = 0

    length = fields.Integer()
    channel = fields.Integer()


class RefClkTypeField(fields.Field):
    def _serialize(self, value, attr, obj, **kwargs):
        return value.name

    def _deserialize(self, value, attr, data, **kwargs):
        if value == 10e6:
            return RefClkType._10MHZ.value
        elif value == 100e6:
            return RefClkType._100MHZ.value
        else:
            raise LabOneQControllerException(
                f"UNsupported reference clock value {value}"
            )


class TriggeringModeField(fields.Field):
    def _serialize(self, value, attr, obj, **kwargs):
        return value.name

    def _deserialize(self, value, attr, data, **kwargs):
        return TriggeringMode[value.upper()]


class AcquisitionTypeField(fields.Field):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["allow_none"] = True
        super().__init__(*args, **kwargs)

    def _serialize(self, value, attr, obj, **kwargs):
        if value is None:
            return None
        return value.name

    def _deserialize(self, value, attr, data, **kwargs):
        if value is None:
            return None
        return AcquisitionType[value.upper()]


class Config(QCCSSchema):
    class Meta:
        fields = (
            "repetitions",
            "reference_clock",
            "holdoff",
            "triggering_mode",
            "sampling_rate",
        )
        ordered = True

    @dataclass
    class Data:
        repetitions: int = 1
        reference_clock: RefClkType = None
        holdoff: float = 0
        triggering_mode: TriggeringMode = TriggeringMode.DIO_FOLLOWER
        sampling_rate: float | None = None

    repetitions = fields.Int()
    reference_clock = RefClkTypeField()
    holdoff = fields.Float()
    triggering_mode = TriggeringModeField()
    sampling_rate = fields.Float()


class Initialization(QCCSSchema):
    class Meta:
        fields = (
            "device_uid",
            "config",
            "awgs",
            "outputs",
            "inputs",
            "measurements",
            "ppchannels",
        )
        ordered = True

    @dataclass
    class Data:
        device_uid: str
        config: Config.Data
        awgs: list[AWG.Data] = None
        outputs: list[IO.Data] = None
        inputs: list[IO.Data] = None
        measurements: list[Measurement.Data] = field(default_factory=list)
        ppchannels: list[dict[str, Any]] | None = None

    device_uid = fields.Str()
    config = fields.Nested(Config)
    awgs = fields.List(fields.Nested(AWG), required=False)
    outputs = fields.List(fields.Nested(IO), required=False)
    inputs = fields.List(fields.Nested(IO), required=False)
    measurements = fields.List(fields.Nested(Measurement), required=False)
    ppchannels = fields.List(fields.Raw, required=False, allow_none=True)


class OscillatorParam(QCCSSchema):
    class Meta:
        fields = ("id", "device_id", "channel", "frequency", "param")
        ordered = True

    @dataclass
    class Data:
        id: str
        device_id: str
        channel: int
        frequency: float = None
        param: str = None

    id = fields.Str()
    device_id = fields.Str()
    channel = fields.Int()
    frequency = fields.Float(required=False, allow_none=True)
    param = fields.Str(required=False, allow_none=True)


class IntegratorAllocation(QCCSSchema):
    class Meta:
        fields = ("signal_id", "device_id", "awg", "channels", "weights", "threshold")
        ordered = True

    @dataclass
    class Data:
        signal_id: str
        device_id: str
        awg: int
        channels: list[int]
        weights: str = None
        threshold: float = 0.0

    signal_id = fields.Str()
    device_id = fields.Str()
    awg = fields.Int()
    channels = fields.List(fields.Int())
    weights = fields.Str(required=False, allow_none=True)
    threshold = fields.Float(required=False)


class AcquireLength(QCCSSchema):
    class Meta:
        fields = ("section_id", "signal_id", "acquire_length")
        ordered = True

    @dataclass
    class Data:
        section_id: str
        signal_id: str
        acquire_length: int

    section_id = fields.Str()
    signal_id = fields.Str()
    acquire_length = fields.Int()


class NtStepKeyField(fields.Field):
    def _deserialize(self, value, attr, data, **kwargs):
        return NtStepKey(indices=tuple(value["indices"]))


class RealtimeExecutionInit(QCCSSchema):
    class Meta:
        fields = (
            "device_id",
            "awg_id",
            "seqc_ref",
            "wave_indices_ref",
            "nt_step",
        )
        ordered = True

    @dataclass
    class Data:
        device_id: str
        awg_id: int
        seqc_ref: str
        wave_indices_ref: str
        nt_step: NtStepKey

    device_id = fields.Str()
    awg_id = fields.Int()
    seqc_ref = fields.Str()
    wave_indices_ref = fields.Str()
    nt_step = NtStepKeyField()


class Experiment(QCCSSchema):
    class Meta:
        fields = (
            "initializations",
            "realtime_execution_init",
            "oscillator_params",
            "integrator_allocations",
            "acquire_lengths",
            "simultaneous_acquires",
            "total_execution_time",
            "max_step_execution_time",
            "acquisition_type",
        )
        ordered = True

    @dataclass
    class Data:
        initializations: list[Initialization.Data]
        realtime_execution_init: list[RealtimeExecutionInit.Data]
        oscillator_params: list[OscillatorParam.Data] = field(default_factory=list)
        integrator_allocations: list[IntegratorAllocation.Data] = field(
            default_factory=list
        )
        acquire_lengths: list[AcquireLength.Data] = field(default_factory=list)
        simultaneous_acquires: list[dict[str, str]] = field(default_factory=list)
        total_execution_time: float = None
        max_step_execution_time: float = None
        acquisition_type: AcquisitionType = AcquisitionTypeField()

    initializations = fields.List(fields.Nested(Initialization))
    realtime_execution_init = fields.List(fields.Nested(RealtimeExecutionInit))
    oscillator_params = fields.List(fields.Nested(OscillatorParam), required=False)
    integrator_allocations = fields.List(
        fields.Nested(IntegratorAllocation), required=False
    )
    acquire_lengths = fields.List(fields.Nested(AcquireLength), required=False)
    simultaneous_acquires = fields.List(
        fields.Dict(fields.Str(), fields.Str()), required=False, allow_none=True
    )
    total_execution_time = fields.Float(required=False, allow_none=True)
    max_step_execution_time = fields.Float(required=False, allow_none=True)
    acquisition_type = AcquisitionTypeField()


class Recipe(QCCSSchema):
    class Meta:
        unknown = EXCLUDE
        fields = ("line_endings", "experiment", "servers", "devices")
        ordered = False

    @dataclass
    class Data:
        line_endings: str
        experiment: Experiment.Data
        servers: list[Server.Data] | None = None
        devices: list[Device.Data] | None = None

    line_endings = fields.Str()
    experiment = fields.Nested(Experiment)
    servers = fields.List(fields.Nested(Server), required=False)
    devices = fields.List(fields.Nested(Device), required=False)

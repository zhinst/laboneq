# Copyright 2019 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import Any, AnyStr, Dict, List, Optional

from marshmallow import EXCLUDE, Schema, fields, post_load

from .recipe_enums import (
    DIOConfigType,
    ExecutionType,
    OperationType,
    RefClkType,
    ReferenceClockSource,
    SignalType,
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
        server_uid: AnyStr
        host: AnyStr
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
        key: AnyStr
        value: AnyStr

    key = fields.Str(required=True)
    value = fields.Str(required=True)


class Device(QCCSSchema):
    class Meta:
        fields = ("device_uid", "driver", "options")
        ordered = True
        unknown = EXCLUDE

    @dataclass
    class Data:
        device_uid: AnyStr
        driver: AnyStr
        options: Optional[List[DriverOption]] = None

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
            "delay_signal",
            "marker_mode",
        )
        ordered = True

    @dataclass
    class Data:
        channel: int
        enable: Optional[bool] = None
        modulation: Optional[bool] = None
        oscillator: Optional[int] = None
        oscillator_frequency: Optional[int] = None
        offset: Optional[float] = None
        gains: Optional[Gains] = None
        range: Optional[float] = None
        range_unit: Optional[str] = None
        precompensation: Optional[Dict[str, Dict]] = None
        lo_frequency: Optional[float] = None
        port_mode: Optional[str] = None
        port_delay: Optional[float] = None
        delay_signal: Optional[float] = None
        marker_mode: Optional[str] = None

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
    lo_frequency = fields.Float(required=False)
    port_mode = fields.Str(required=False)
    port_delay = fields.Float(required=False)
    delay_signal = fields.Float(required=False)
    marker_mode = fields.Str(required=False)


class SignalTypeField(fields.Field):
    def _serialize(self, value, attr, obj, **kwargs):
        return value.name

    def _deserialize(self, value, attr, data, **kwargs):
        return SignalType[value.upper()]


class AWG(QCCSSchema):
    class Meta:
        fields = (
            "awg",
            "seqc",
            "signal_type",
            "qa_signal_id",
            "command_table_match_offset",
        )
        ordered = False

    @dataclass
    class Data:
        awg: int
        seqc: str
        signal_type: SignalType = SignalType.SINGLE
        qa_signal_id: Optional[str] = None
        command_table_match_offset: Optional[int] = None

    awg = fields.Integer()
    seqc = fields.Str()
    signal_type = SignalTypeField()
    qa_signal_id = fields.Str(required=False, allow_none=True)
    command_table_match_offset = fields.Integer(required=False, allow_none=True)


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


class ExecutionTypeField(fields.Field):
    def _serialize(self, value, attr, obj, **kwargs):
        return value.name

    def _deserialize(self, value, attr, data, **kwargs):
        return ExecutionType[value.upper()]


class listElement(QCCSSchema):
    class Meta:
        # TODO(MG) data_type is deprecated and should be removed
        fields = ("source", "data_type")
        ordered = True

    @dataclass
    class Data:
        source: AnyStr
        # TODO(MG) data_type is deprecated and should be removed
        data_type: AnyStr

    source = fields.Str()
    # TODO(MG) data_type is deprecated and should be removed
    data_type = fields.Str()


class Location(QCCSSchema):
    @dataclass
    class Data:
        type: AnyStr
        index: int

    type = fields.Str()
    index = fields.Integer()


class Parameter(QCCSSchema):
    class Meta:
        fields = (
            "device_uid",
            "location",
            "parameter_uid",
            "index",
            "list",
            "start",
            "step",
        )
        ordered = True

    @dataclass
    class Data:
        device_uid: AnyStr = None
        location: Optional[Location] = None
        parameter_uid: Optional[AnyStr] = None
        index: Optional[int] = None
        list: Optional[listElement] = None
        start: Optional[float] = None
        step: Optional[float] = None

    device_uid = fields.Str()
    location = fields.Nested(Location)
    parameter_uid = fields.Str(required=False)
    index = fields.Integer(required=False)
    list = fields.Nested(listElement, required=False)
    start = fields.Float(required=False)
    step: fields.Float(required=False)


class OperationTypeField(fields.Field):
    def _serialize(self, value, attr, obj, **kwargs):
        return value.name

    def _deserialize(self, value, attr, data, **kwargs):
        return OperationType[value.upper()]


class SectionOperation(QCCSSchema):
    class Meta:
        fields = ("op_type", "operation", "args")
        ordered = True

    @dataclass
    class Data:
        op_type: OperationType = None
        operation: str = None
        args: Dict[str, Any] = None

    op_type = OperationTypeField(required=True)
    operation = fields.Str(required=True)
    args = fields.Dict(None, required=False)


class Execution(QCCSSchema):
    class Meta:
        fields = ("type", "count", "parameters", "children")
        ordered = True

    @dataclass
    class Data:
        type: ExecutionType
        count: int = None
        parameters: List[Parameter.Data] = None
        children: List[Any] = None

    type = ExecutionTypeField(required=False)
    count = fields.Integer(required=False)
    parameters = fields.List(fields.Nested(Parameter), required=False)
    children = fields.List(fields.Nested(lambda: Execution()), required=False)


class Measurement(QCCSSchema):
    class Meta:
        fields = ("length", "delay", "channel")

    @dataclass
    class Data:
        length: int
        delay: int
        channel: int = 0

    length = fields.Integer()
    delay = fields.Integer()
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


class DIOConfigTypeField(fields.Field):
    def _serialize(self, value, attr, obj, **kwargs):
        return value.name

    def _deserialize(self, value, attr, data, **kwargs):
        return DIOConfigType[value.upper()]


class ReferenceClockSourceField(fields.Field):
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
        return ReferenceClockSource[value.upper()]


class Config(QCCSSchema):
    class Meta:
        fields = (
            "repetitions",
            "reference_clock",
            "holdoff",
            "dio_mode",
            "sampling_rate",
            "reference_clock_source",
        )
        ordered = True

    @dataclass
    class Data:
        repetitions: int = 1
        reference_clock: RefClkType = None
        holdoff: float = 0
        dio_mode: DIOConfigType = DIOConfigType.HDAWG
        sampling_rate: Optional[float] = None
        reference_clock_source: Optional[ReferenceClockSource] = None

    repetitions = fields.Int()
    reference_clock = RefClkTypeField()
    holdoff = fields.Float()
    dio_mode = DIOConfigTypeField()
    sampling_rate = fields.Float()
    reference_clock_source = ReferenceClockSourceField()


class Initialization(QCCSSchema):
    class Meta:
        fields = (
            "device_uid",
            "config",
            "awgs",
            "ports",
            "outputs",
            "inputs",
            "measurements",
        )
        ordered = True

    @dataclass
    class Data:
        device_uid: AnyStr
        config: Config.Data
        awgs: List[AWG.Data] = None
        ports: List[Port.Data] = None
        outputs: List[IO.Data] = None
        inputs: List[IO.Data] = None
        measurements: List[Measurement.Data] = field(default_factory=list)

    device_uid = fields.Str()
    config = fields.Nested(Config)
    awgs = fields.List(fields.Nested(AWG), required=False)
    ports = fields.List(fields.Nested(Port), required=False)
    outputs = fields.List(fields.Nested(IO), required=False)
    inputs = fields.List(fields.Nested(IO), required=False)
    measurements = fields.List(fields.Nested(Measurement), required=False)


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
        channels: List[int]
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


class Experiment(QCCSSchema):
    class Meta:
        fields = (
            "initializations",
            "oscillator_params",
            "integrator_allocations",
            "acquire_lengths",
            "simultaneous_acquires",
            "total_execution_time",
        )
        ordered = True

    @dataclass
    class Data:
        initializations: List[Initialization.Data]
        oscillator_params: List[OscillatorParam.Data] = field(default_factory=list)
        integrator_allocations: List[IntegratorAllocation.Data] = field(
            default_factory=list
        )
        acquire_lengths: List[AcquireLength.Data] = field(default_factory=list)
        simultaneous_acquires: List[Dict[str, str]] = field(default_factory=list)
        total_execution_time: float = None

    initializations = fields.List(fields.Nested(Initialization))
    oscillator_params = fields.List(fields.Nested(OscillatorParam), required=False)
    integrator_allocations = fields.List(
        fields.Nested(IntegratorAllocation), required=False
    )
    acquire_lengths = fields.List(fields.Nested(AcquireLength), required=False)
    simultaneous_acquires = fields.List(
        fields.Dict(fields.Str(), fields.Str()), required=False, allow_none=True
    )
    total_execution_time = fields.Float(required=False, allow_none=True)


class Recipe(QCCSSchema):
    class Meta:
        unknown = EXCLUDE
        fields = ("line_endings", "experiment", "servers", "devices")
        ordered = False

    @dataclass
    class Data:
        line_endings: AnyStr
        experiment: Experiment.Data
        servers: Optional[List[Server.Data]] = None
        devices: Optional[List[Device.Data]] = None

    line_endings = fields.Str()
    experiment = fields.Nested(Experiment)
    servers = fields.List(fields.Nested(Server), required=False)
    devices = fields.List(fields.Nested(Device), required=False)

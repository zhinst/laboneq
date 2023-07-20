# Copyright 2019 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from marshmallow import EXCLUDE, Schema, fields, post_load

from laboneq.data import recipe


@dataclass
class JsonRecipe:
    line_endings: str
    experiment: recipe.Recipe
    servers: list[Any] | None = None
    devices: list[Any] | None = None


explicit_mapping = {
    "JsonRecipeLoader": JsonRecipe,
    "Experiment": recipe.Recipe,
    "Device": dict,
}


def convert_from_legacy_json_recipe(legacy_recipe: dict) -> recipe.Recipe:
    return cast(JsonRecipe, JsonRecipeLoader().load(legacy_recipe)).experiment


class QCCSSchema(Schema):
    @post_load
    def from_json(self, data, **kwargs):
        cls = explicit_mapping.get(self.__class__.__name__)
        if cls is None:
            cls = getattr(recipe, self.__class__.__name__)
        return cls(**data)


class Server(QCCSSchema):
    class Meta:
        fields = ("server_uid", "host", "port", "api_level")
        ordered = True

    server_uid = fields.Str()
    host = fields.Str()
    port = fields.Integer()
    api_level = fields.Integer()


class DriverOption(fields.Field):
    class Meta:
        fields = ("parameter_name", "value")
        ordered = False

    key = fields.Str(required=True)
    value = fields.Str(required=True)


class Device(QCCSSchema):
    class Meta:
        fields = ("device_uid", "driver", "options")
        ordered = True
        unknown = EXCLUDE

    device_uid = fields.Str()
    driver = fields.Str()
    options = fields.List(DriverOption(), required=False)


class Gains(QCCSSchema):
    class Meta:
        fields = ("diagonal", "off_diagonal")
        ordered = True

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

    channel = fields.Integer()
    enable = fields.Boolean(required=False)
    modulation = fields.Boolean(required=False)
    oscillator = fields.Integer(required=False)
    oscillator_frequency = fields.Float(required=False)
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
        return recipe.SignalType[value.upper()]


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

    awg = fields.Integer()
    signal_type = SignalTypeField()
    qa_signal_id = fields.Str(required=False, allow_none=True)
    command_table_match_offset = fields.Integer(required=False, allow_none=True)
    feedback_register = fields.Integer(required=False, allow_none=True)


class Port(QCCSSchema):
    class Meta:
        fields = ("port", "device_uid")
        ordered = True

    port = fields.Integer()
    device_uid = fields.Str()


class Measurement(QCCSSchema):
    class Meta:
        fields = ("length", "channel")

    length = fields.Integer()
    channel = fields.Integer()


class RefClkTypeField(fields.Field):
    def _serialize(self, value, attr, obj, **kwargs):
        return value.name

    def _deserialize(self, value, attr, data, **kwargs):
        if value == 10e6:
            return recipe.RefClkType._10MHZ.value
        else:
            return recipe.RefClkType._100MHZ.value


class TriggeringModeField(fields.Field):
    def _serialize(self, value, attr, obj, **kwargs):
        return value.name

    def _deserialize(self, value, attr, data, **kwargs):
        return recipe.TriggeringMode[value.upper()]


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

    id = fields.Str()
    device_id = fields.Str()
    channel = fields.Int()
    frequency = fields.Float(required=False, allow_none=True)
    param = fields.Str(required=False, allow_none=True)


class IntegratorAllocation(QCCSSchema):
    class Meta:
        fields = ("signal_id", "device_id", "awg", "channels", "weights", "threshold")
        ordered = True

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

    section_id = fields.Str()
    signal_id = fields.Str()
    acquire_length = fields.Int()


class NtStepKeyField(fields.Field):
    def _deserialize(self, value, attr, data, **kwargs):
        return recipe.NtStepKey(indices=tuple(value["indices"]))


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
        )
        ordered = True
        unknown = EXCLUDE

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


class JsonRecipeLoader(QCCSSchema):
    class Meta:
        unknown = EXCLUDE
        fields = ("line_endings", "experiment", "servers", "devices")
        ordered = False

    line_endings = fields.Str()
    experiment = fields.Nested(Experiment)
    servers = fields.List(fields.Nested(Server), required=False)
    devices = fields.List(fields.Nested(Device), required=False)

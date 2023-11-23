# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto


class AttributeName(Enum):
    OSCILLATOR_FREQ = auto()
    OUTPUT_SCHEDULER_PORT_DELAY = auto()
    OUTPUT_PORT_DELAY = auto()
    INPUT_SCHEDULER_PORT_DELAY = auto()
    INPUT_PORT_DELAY = auto()
    PPC_PUMP_FREQ = auto()
    PPC_PUMP_POWER = auto()
    PPC_PROBE_FREQUENCY = auto()
    PPC_PROBE_POWER = auto()
    QA_OUT_AMPLITUDE = auto()
    QA_CENTER_FREQ = auto()
    SG_SYNTH_CENTER_FREQ = auto()
    SG_DIG_MIXER_CENTER_FREQ = auto()
    OUTPUT_ROUTE_1 = auto()
    OUTPUT_ROUTE_2 = auto()
    OUTPUT_ROUTE_3 = auto()
    OUTPUT_ROUTE_1_AMPLITUDE = auto()
    OUTPUT_ROUTE_2_AMPLITUDE = auto()
    OUTPUT_ROUTE_3_AMPLITUDE = auto()
    OUTPUT_ROUTE_1_PHASE = auto()
    OUTPUT_ROUTE_2_PHASE = auto()
    OUTPUT_ROUTE_3_PHASE = auto()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}.{self.name}"


@dataclass
class DeviceAttribute:
    name: AttributeName
    index: int | None
    value_or_param: float | str


@dataclass
class AttributeValue:
    value: float | None
    updated: bool = field(init=False, default=False)

    def __post_init__(self):
        if self.value is not None:
            self.updated = True

    def update(self, value: float):
        self.value = value
        self.updated = True

    def is_updated(self) -> bool:
        return self.updated

    def reset_updated(self):
        self.updated = False

    def get(self) -> float | None:
        return self.value


@dataclass(frozen=True)
class AttributeKey:
    device_uid: str
    name: AttributeName
    index: int | None


@dataclass
class AttributeValueTracker:
    param_to_attributes_map: dict[str, list[AttributeKey]] = field(
        default_factory=lambda: defaultdict(list)
    )
    attributes: dict[AttributeKey, AttributeValue] = field(default_factory=dict)

    def add_attribute(
        self,
        device_uid: str,
        attribute: DeviceAttribute,
    ):
        is_param = isinstance(attribute.value_or_param, str)
        param = attribute.value_or_param if is_param else None
        value = None if is_param else attribute.value_or_param

        attribute_key = AttributeKey(
            device_uid=device_uid, name=attribute.name, index=attribute.index
        )
        self.attributes[attribute_key] = AttributeValue(value)
        if is_param:
            self.param_to_attributes_map[param].append(attribute_key)

    def update(self, param: str, value: float):
        for attribute_key in self.param_to_attributes_map.get(param, []):
            self.attributes[attribute_key].update(value)

    def device_view(self, device_uid: str) -> DeviceAttributesView:
        return DeviceAttributesView(device_uid=device_uid, attributes=self.attributes)

    def reset_updated(self):
        for attribute_value in self.attributes.values():
            attribute_value.reset_updated()


@dataclass
class DeviceAttributesView:
    device_uid: str
    attributes: dict[AttributeKey, AttributeValue]

    def resolve(
        self, keys: list[tuple[AttributeName, int]]
    ) -> tuple[list[float], bool]:
        resolved = []
        any_updated = False
        for key in keys:
            attr_key = AttributeKey(
                device_uid=self.device_uid, name=key[0], index=key[1]
            )
            attr_value = self.attributes.get(attr_key)
            if attr_value is None:
                resolved.append(None)
            else:
                resolved.append(attr_value.get())
                any_updated = any_updated or attr_value.is_updated()
        return resolved, any_updated

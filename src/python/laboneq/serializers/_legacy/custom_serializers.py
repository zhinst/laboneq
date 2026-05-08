# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

import re

from laboneq.data.scheduled_experiment import (
    ResultShapeInfo,
    ResultSource,
)


class ResultShapeInfoDeserializer:
    _type_ = "ResultShapeInfo"

    @staticmethod
    def deserialize(mapping: dict) -> ResultShapeInfo:
        result_handle_maps = {
            _deserialize_result_source(k): v
            for k, v in mapping["result_handle_maps"].items()
        }
        return ResultShapeInfo(mapping["shapes"], result_handle_maps)


def serialize_to_string(obj) -> str:
    if isinstance(obj, ResultSource):
        return _serialize_result_source(obj)

    return obj


def _serialize_result_source(obj: ResultSource):
    return f"ResultSource({obj.device_id}, {obj.awg_id}, {obj.integrator_idx})"


def _deserialize_result_source(obj) -> ResultSource:
    match_result = re.fullmatch(r"ResultSource\((.*), (.*), (.*)\)", obj)
    assert match_result is not None
    device_id, awg_id, integrator_idx = match_result.groups()
    if awg_id.isnumeric():
        awg_id = int(awg_id)
    return ResultSource(
        device_id, awg_id, int(integrator_idx) if integrator_idx != "None" else None
    )

# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

import re

from laboneq.compiler.common.result_shape import ResultShapeInfo, ResultSource
from laboneq.data.awg_info import AwgKey


class ResultShapeInfoDeserializer:
    _type_ = "ResultShapeInfo"

    @staticmethod
    def deserialize(mapping: dict) -> ResultShapeInfo:
        result_handle_maps = {
            _deserialize_result_source(k): v
            for k, v in mapping["result_handle_maps"].items()
        }
        result_lengths = {
            _deserialize_awg_key(k): v for k, v in mapping["result_lengths"].items()
        }
        return ResultShapeInfo(mapping["shapes"], result_handle_maps, result_lengths)


def serialize_to_string(obj) -> str:
    if isinstance(obj, AwgKey):
        return _serialize_awg_key(obj)

    if isinstance(obj, ResultSource):
        return _serialize_result_source(obj)

    return obj


def _serialize_awg_key(obj: AwgKey):
    return f"AwgKey({obj.device_id}, {obj.awg_id})"


def _deserialize_awg_key(obj) -> AwgKey:
    match_result = re.fullmatch(r"AwgKey\((.*), (.*)\)", obj)
    assert match_result is not None
    device_id, awg_idx = match_result.groups()
    if awg_idx.isnumeric():
        awg_idx = int(awg_idx)
    return AwgKey(device_id, awg_idx)


def _serialize_result_source(obj: ResultSource):
    return f"ResultSource({obj.device_id}, {obj.awg_id}, {obj.integrator_idx})"


def _deserialize_result_source(obj) -> ResultSource:
    match_result = re.fullmatch(r"ResultSource\((.*), (.*), (.*)\)", obj)
    assert match_result is not None
    device_id, awg_id, integrator_idx = match_result.groups()
    if awg_id.isnumeric():
        awg_id = int(awg_id)
    return ResultSource(device_id, awg_id, int(integrator_idx))

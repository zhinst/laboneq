# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from laboneq.core.serialization.simple_serialization import serialize_to_dict
from laboneq.data.execution_payload import NearTimeProgram


class ExecutionPayloadHelper:
    @staticmethod
    def dump_near_time_program(near_time_program: NearTimeProgram):
        return serialize_to_dict(near_time_program)

    @staticmethod
    def descend(current_node, visitor, context, parent):
        for c in current_node.children:
            ExecutionPayloadHelper.descend(c, visitor, context, current_node)
        visitor(current_node, context, parent)

    @staticmethod
    def accept_near_time_program_visitor(
        near_time_program: NearTimeProgram, visitor, context=None
    ):
        ExecutionPayloadHelper.descend(near_time_program, visitor, context, None)

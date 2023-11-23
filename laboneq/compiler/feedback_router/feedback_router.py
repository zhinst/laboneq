# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import defaultdict
from functools import singledispatch
from typing import Literal, Union

from laboneq._utils import cached_method
from laboneq.compiler.common.awg_info import AWGInfo, AwgKey
from laboneq.compiler.common.awg_signal_type import AWGSignalType
from laboneq.compiler.common.device_type import DeviceType
from laboneq.compiler.common.feedback_connection import FeedbackConnection
from laboneq.compiler.common.feedback_register_config import FeedbackRegisterConfig
from laboneq.compiler.common.signal_obj import SignalObj
from laboneq.compiler.workflow.compiler_output import (
    CombinedRealtimeCompilerOutput,
    CombinedRealtimeCompilerOutputCode,
    CombinedRealtimeCompilerOutputPrettyPrinter,
)
from laboneq.core.exceptions import LabOneQException

# list of (width, signal) tuples
FeedbackRegisterLayout = list[tuple[int, Union[str, None]]]


class FeedbackRouter:
    def __init__(
        self,
        signal_objs: dict[str, SignalObj],
        integration_unit_allocation: dict,
        feedback_connections: dict[str, FeedbackConnection],
        feedback_register_configs: dict[AwgKey, FeedbackRegisterConfig],
    ):
        self._signal_objs = signal_objs
        self._awgs = {
            AwgKey(signal.awg.device_id, signal.awg.awg_number): signal.awg
            for signal in self._signal_objs.values()
        }
        self._integration_unit_allocation = integration_unit_allocation
        self._feedback_connections = feedback_connections
        self._feedback_register_configs = feedback_register_configs

    def calculate_feedback_routing(self):
        """Complete the feedback routing information.

        After code generation, `self._combined_compiler_output.feedback_register_configurations`
        is still incomplete. Notably, it is missing the actual bit fiddling required to
        retrieve a specific measurement from the feedback registers. This is elaborated
        here.
        """

        feedback_register_configs = self._feedback_register_configs

        register_layout = self._calculate_feedback_register_layout(
            feedback_register_configs, self._integration_unit_allocation
        )

        for awg in self._awgs.values():
            if (tx_qa := self._transmitter_qa_for_awg(awg.key)) is None:
                continue
            qa_awg, qa_signal = tx_qa
            register = feedback_register_configs[qa_awg].target_feedback_register
            assert register is not None

            use_local_feedback = self._local_feedback_allowed(awg, self._awgs[qa_awg])

            register_bitshift, width, mask = self._register_bitshift(
                register,
                qa_signal,
                register_layout,
                force_local_alignment=use_local_feedback,
            )
            if use_local_feedback:
                register = "local"

            if register == "local":
                codeword_bitshift = register_bitshift
                register_index_select = None
            else:
                # feedback through PQSC: assign index based on AWG number
                register_index_select = register_bitshift // 2
                codeword_bitshift = 2 * awg.awg_number + register_bitshift % 2
                if register_bitshift % 2 and width > 1:
                    raise AssertionError(
                        "Measurement result must not span across indices"
                    )

            feedback_register_config = feedback_register_configs[awg.key]
            feedback_register_config.source_feedback_register = register
            feedback_register_config.codeword_bitshift = codeword_bitshift
            feedback_register_config.register_index_select = register_index_select
            feedback_register_config.codeword_bitmask = mask

    @cached_method()
    def _transmitter_qa_for_awg(self, awg_key: AwgKey) -> tuple[AwgKey, str] | None:
        """Find the QA core that is transmitting feedback data to this AWG."""
        awg = self._awgs[AwgKey(awg_key.device_id, awg_key.awg_number)]
        signal_type = awg.signal_type
        if signal_type == AWGSignalType.DOUBLE:
            awg_signals = {f"{awg.signal_channels[0][0]}_{awg.signal_channels[1][0]}"}
        else:
            awg_signals = {c for c, _ in awg.signal_channels}
        qa_signal_ids = {
            h.acquire
            for h in self._feedback_connections.values()
            if h.drive.intersection(awg_signals)
        }
        if len(qa_signal_ids) == 0:
            return None
        if len(qa_signal_ids) > 1:
            raise LabOneQException(
                f"The drive signal(s) ({set(awg_signals)}) can only react to "
                f"one acquire signal for feedback, got {qa_signal_ids}."
            )
        [qa_signal_id] = qa_signal_ids
        assert qa_signal_id is not None
        qa_awg = self._signal_objs[qa_signal_id].awg.key

        return qa_awg, qa_signal_id

    @staticmethod
    def _calculate_feedback_register_layout(
        feedback_register_configs: dict[AwgKey, FeedbackRegisterConfig],
        integration_unit_allocation,
    ):
        feedback_register_layout: defaultdict[
            int | Literal["local"], FeedbackRegisterLayout
        ] = defaultdict(FeedbackRegisterLayout)

        for signal_id, integration_unit in sorted(
            integration_unit_allocation.items(),
            key=lambda i: i[1]["channels"],
        ):
            device_id = integration_unit["device_id"]
            awg_nr = integration_unit["awg_nr"]

            qa_awg_key = AwgKey(device_id, awg_nr)
            if qa_awg_key not in feedback_register_configs:
                continue
            register = feedback_register_configs[qa_awg_key].target_feedback_register
            if register is None:
                continue  # not transmitted

            bit_width = 1
            if integration_unit["is_msd"] or register == "local":
                bit_width = 2

            feedback_register_layout[register].append((bit_width, signal_id))

            if bit_width < len(integration_unit["channels"]):
                # On UHFQA, with `AcquisitionType.INTEGRATION`, we have 2 integrators
                # per signal. For discrimination, that 2nd integrator is irrelevant, so
                # we mark that bit as a 'dummy' field.
                feedback_register_layout[register].append((1, None))

        return dict(feedback_register_layout)  # cast defaultdict to dict

    @staticmethod
    def _register_bitshift(
        register: int, qa_signal: str, register_layout, force_local_alignment=False
    ):
        """Calculate offset and mask into register for given qa_signal

        If `force_local_alignment` is true, assume that every measurement spans 2 bits,
        i.e. how the data are laid out in the local feedback register.
        """
        register_bitshift = 0  # offset into the register
        for width, signal in register_layout[register]:
            if signal == qa_signal:
                mask = (1 << width) - 1
                break
            else:
                register_bitshift += width if not force_local_alignment else 2
        else:
            raise AssertionError(f"Signal {qa_signal} not found in register {register}")
        return register_bitshift, width, mask

    @staticmethod
    def _local_feedback_allowed(sg_awg: AWGInfo, qa_awg: AWGInfo):
        # todo: this check for QC is quite brittle

        return (
            sg_awg.device_type == DeviceType.SHFSG
            and qa_awg.device_type == DeviceType.SHFQA
            and sg_awg.device_id == f"{qa_awg.device_id}_sg"
        )


@singledispatch
def _do_compute_feedback_routing(
    combined_compiler_output, signal_objs, integration_unit_allocation
):
    raise NotImplementedError()


@_do_compute_feedback_routing.register
def _(
    combined_compiler_output: CombinedRealtimeCompilerOutputCode,
    signal_objs,
    integration_unit_allocation,
):
    FeedbackRouter(
        signal_objs,
        integration_unit_allocation,
        combined_compiler_output.feedback_connections,
        combined_compiler_output.feedback_register_configurations,
    ).calculate_feedback_routing()


@_do_compute_feedback_routing.register
def _(
    combined_compiler_output: CombinedRealtimeCompilerOutputPrettyPrinter,
    signal_objs,
    integration_unit_allocation,
):
    ...


def compute_feedback_routing(
    signal_objs,
    integration_unit_allocation,
    combined_compiler_output: CombinedRealtimeCompilerOutput,
):
    for output in combined_compiler_output.combined_output.values():
        _do_compute_feedback_routing(output, signal_objs, integration_unit_allocation)

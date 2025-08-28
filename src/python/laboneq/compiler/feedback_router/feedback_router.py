# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, NamedTuple, Union

from laboneq.compiler.common.awg_info import AwgKey

if TYPE_CHECKING:
    from laboneq.compiler.workflow.compiler import IntegrationUnitAllocation


class LocalFeedbackRegister(NamedTuple):
    device: str


class GlobalFeedbackRegister(NamedTuple):
    source: AwgKey


# list of (width, signal) tuples
SingleFeedbackRegisterLayout = list[tuple[int, Union[str, None]]]

FeedbackRegisterLayout = dict[
    Union[LocalFeedbackRegister, GlobalFeedbackRegister], SingleFeedbackRegisterLayout
]


def calculate_feedback_register_layout(
    integration_unit_allocation: dict[str, IntegrationUnitAllocation],
) -> FeedbackRegisterLayout:
    feedback_register_layouts: FeedbackRegisterLayout = defaultdict(
        SingleFeedbackRegisterLayout
    )

    for signal_id, integration_unit in sorted(
        integration_unit_allocation.items(),
        key=lambda i: i[1].channels,
    ):
        device_id = integration_unit.device_id
        awg_nr = integration_unit.awg_nr
        qa_awg_key = AwgKey(device_id, awg_nr)

        for local in (True, False):
            if local and not integration_unit.has_local_bus:
                continue

            register_key = (
                LocalFeedbackRegister(device_id)
                if local
                else GlobalFeedbackRegister(qa_awg_key)
            )

            bit_width = 1
            is_msd = (integration_unit.kernel_count or 0) > 1
            if is_msd or local:
                bit_width = 2

            feedback_register_layouts[register_key].append((bit_width, signal_id))

            if bit_width < len(integration_unit.channels):
                # On UHFQA, with `AcquisitionType.INTEGRATION`, we have 2 integrators
                # per signal. For discrimination, that 2nd integrator is irrelevant, so
                # we mark that bit as a 'dummy' field.
                feedback_register_layouts[register_key].append((1, None))

    # remove default dict-ness
    feedback_register_layouts = dict(feedback_register_layouts)

    return feedback_register_layouts

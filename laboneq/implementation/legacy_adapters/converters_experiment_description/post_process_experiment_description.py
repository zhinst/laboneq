# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

import logging

from laboneq.data.experiment_description import Experiment, PlayPulse
from laboneq.implementation.legacy_adapters.dynamic_converter import convert_dynamic

_logger = logging.getLogger(__name__)

PULSES = {}


def post_process(source, target, conversion_function_lookup):
    global PULSES

    if type(source).__name__ in ["Section", "AcquireLoopNt", "AcquireLoopRt"]:
        _logger.info(
            f"Converting {type(source).__name__} {source},\n converting children"
        )
        if source.children is not None:
            target.children = convert_dynamic(
                source.children,
                source_type_string="List",
                conversion_function_lookup=conversion_function_lookup,
            )
        if source.trigger is not None:
            target.trigger = convert_dynamic(
                source.trigger,
                source_type_string="Dict",
                conversion_function_lookup=conversion_function_lookup,
            )
        return target

    if type(target) == Experiment:
        _logger.info(f"Postprocess_experiment for {source.uid}")
        target.pulses = list(PULSES.values())
        PULSES = {}
        return target

    if type(target) == PlayPulse:
        _logger.info(f"Postprocess_experiment for {source}")
        if source.pulse.uid not in PULSES:
            PULSES[source.pulse.uid] = conversion_function_lookup(type(source.pulse))(
                source.pulse
            )
        target.pulse = PULSES[source.pulse.uid]
        target.signal_uid = source.signal

    return target

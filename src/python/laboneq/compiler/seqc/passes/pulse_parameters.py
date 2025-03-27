# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from laboneq.compiler import ir
from laboneq.compiler.common.pulse_parameters import (
    encode_pulse_parameters,
    PulseParams,
)


def _detach_pulse_params(
    node: ir.IntervalIR,
    param_map: dict,
):
    if type(node) is ir.PulseIR:
        if not node.pulse_pulse_params and not node.play_pulse_params:
            return
        params = PulseParams(
            pulse_params=node.pulse_pulse_params, play_params=node.play_pulse_params
        )
        params_combined = encode_pulse_parameters(params.combined())
        if params_combined not in param_map:
            params_id = len(param_map)
            param_map[params_combined] = (params_id, params)
        else:
            params_id, _ = param_map[params_combined]
        # NOTE: Must always be same
        node.play_pulse_params = params_id
        node.pulse_pulse_params = params_id
    elif type(node) is ir.AcquireGroupIR:
        ids = []
        for pulse_p, play_p in zip(node.pulse_pulse_params, node.play_pulse_params):
            if not pulse_p and not play_p:
                ids.append(None)
                continue
            params = PulseParams(pulse_params=pulse_p, play_params=play_p)
            params_combined = encode_pulse_parameters(params.combined())
            if params_combined not in param_map:
                params_id = len(param_map)
                param_map[params_combined] = (params_id, params)
            else:
                params_id, _ = param_map[params_combined]
            ids.append(params_id)
        # NOTE: Must always be same
        node.play_pulse_params = ids
        node.pulse_pulse_params = ids
    else:
        for child in node.children:
            _detach_pulse_params(child, param_map)


def detach_pulse_params(root: ir.IntervalIR) -> list[PulseParams]:
    """Pass to extra arbitrary pulse parameters out of the IR.

    The parameters are replaced with an ID that points to the index of the
    returned value.
    """
    params = {}
    _detach_pulse_params(root, param_map=params)
    return [p[1] for p in params.values()]

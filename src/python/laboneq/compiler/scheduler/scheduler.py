# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

from laboneq.compiler.scheduler.parameter_store import ParameterStore

if TYPE_CHECKING:
    from laboneq._rust import compiler as compiler_rs


class Scheduler:
    def __init__(
        self,
        compiler_module: compiler_rs,
    ):
        self._compiler_module = compiler_module

    def run(
        self,
        experiment,
        nt_parameters: ParameterStore[str, float],
    ):
        if nt_parameters is None:
            nt_parameters = ParameterStore[str, float]()

        self._scheduled_experiment_rs = self._compiler_module.schedule_experiment(
            experiment=experiment,
            parameters={
                k: v
                for k, v in nt_parameters.items()
                if k not in ("__chunk_index", "__chunk_count")
            },
            chunking_info=None
            if "__chunk_index" not in nt_parameters
            else (nt_parameters["__chunk_index"], nt_parameters["__chunk_count"]),
        )
        # Flush used `nt_parameters` so that they get registered
        for used_parameter in self._scheduled_experiment_rs.used_parameters:
            nt_parameters.mark_used(used_parameter)
        return self._scheduled_experiment_rs.experiment_ir

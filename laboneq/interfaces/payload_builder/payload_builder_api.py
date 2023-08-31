# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from abc import ABC

from laboneq.data.execution_payload import ExecutionPayload, TargetSetup
from laboneq.data.experiment_description import Experiment
from laboneq.data.setup_description import Setup


class PayloadBuilderAPI(ABC):
    def build_payload(
        self,
        device_setup: Setup,
        experiment_descriptor: Experiment,
        signal_mappings: dict[str, str],
        compiler_settings: dict = None,
    ) -> ExecutionPayload:
        """
        Compose an experiment from a setup descriptor and an experiment descriptor.
        """
        raise NotImplementedError

    def convert_to_target_setup(self, device_setup: Setup) -> TargetSetup:
        """
        Convert the given device setup to a target setup.
        """
        raise NotImplementedError

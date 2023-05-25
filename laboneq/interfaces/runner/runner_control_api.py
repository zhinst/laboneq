# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from abc import ABC

from laboneq.data.execution_payload import TargetSetup


class RunnerControlAPI(ABC):
    def connect(self, setup: TargetSetup, do_emulation: bool = True):
        """
        Connect to the setup
        """
        raise NotImplementedError

    def start(self):
        """
        Start the experiment runner. It will start processing jobs from the job queue.
        """
        raise NotImplementedError

    def stop(self):
        """
        Stop the experiment runner. It will stop processing jobs from the job queue.
        """
        raise NotImplementedError

    def disconnect(self):
        """
        Disconnect from the setup.
        """
        raise NotImplementedError

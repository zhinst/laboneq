# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod

from laboneq.data.execution_payload import TargetSetup


class RunnerControlAPI(ABC):
    @abstractmethod
    def connect(self, setup: TargetSetup, do_emulation: bool = True):
        """
        Connect to the setup
        """
        pass

    @abstractmethod
    def start(self):
        """
        Start the experiment runner. It will start processing jobs from the job queue.
        """
        pass

    @abstractmethod
    def stop(self):
        """
        Stop the experiment runner. It will stop processing jobs from the job queue.
        """
        pass

    @abstractmethod
    def disconnect(self):
        """
        Disconnect from the setup.
        """
        pass

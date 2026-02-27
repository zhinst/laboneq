# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Backend plugin system for LabOne Q.

Backends can register themselves via entry points in the "laboneq.backends" group.
Each backend must provide a class that implements the BackendPlugin interface.
"""

from abc import ABC, abstractmethod


class BackendPlugin(ABC):
    """Base class for LabOne Q backend plugins.

    Backend plugins are registered via entry points and loaded automatically
    when laboneq is imported. Each backend must implement the register() method
    to set up its device types, compiler hooks, and other integrations.
    """

    @abstractmethod
    def register(self) -> None:
        """Register the backend with LabOne Q.

        This method is called once during laboneq import. It should:
        - Register device types with ``DeviceFactory.register_device()``
        - Register compiler hooks via ``register_compiler_hooks()``
        - Register logger namespaces via ``register_logger_namespace()``
        - Register instrumentors via ``LabOneQInstrumentor.register_instrumentor()``
        - Register serializers or other framework extensions
        """
        ...

# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""A package for creating automation frameworks.

The package provides tools and building blocks to define automation frameworks.
"""

from laboneq.automation.automation import Automation
from laboneq.automation.layer import AutomationLayer, AutomationLayerResult
from laboneq.automation.logic import AutomationLogic
from laboneq.automation.node import AutomationNode, NodeKey
from laboneq.automation.status import AutomationStatus

__all__ = [
    "Automation",
    "AutomationLayer",
    "AutomationLayerResult",
    "AutomationLogic",
    "AutomationNode",
    "AutomationStatus",
    "NodeKey",
]

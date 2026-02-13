# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""A package for creating automation frameworks.

The package provides tools and building blocks to define automation frameworks.
"""

from laboneq._automation.automation import Automation
from laboneq._automation.element import AutomationElement, AutomationElementStatus
from laboneq._automation.layer import AutomationLayer
from laboneq._automation.node import AutomationNode

__all__ = [
    "Automation",
    "AutomationElement",
    "AutomationElementStatus",
    "AutomationLayer",
    "AutomationNode",
]

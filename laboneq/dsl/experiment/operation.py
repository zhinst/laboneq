# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from abc import abstractmethod


class Operation:
    """Class representing a operation."""

    @abstractmethod
    def get_all_signals(self) -> set:
        """Retrieve all signals that are linked to this operation.

        Returns:
            Set of signals that are linked to this operation.
        """
        pass

# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from laboneq.controller.utilities.exception import LabOneQControllerException

if TYPE_CHECKING:
    from laboneq.controller.controller import ControllerSubmission


class SubmissionHandle:
    """A handle representing a submitted experiment for tracking and identification.

    This handle serves as a unique identifier for experiment submissions within the
    LabOne Q controller API. It provides a lightweight way to reference and track
    experiments without exposing internal implementation details.

    The handle is designed to be opaque to users - all interactions with submitted
    experiments should be performed through the ControllerAPI methods using this handle
    as a reference.

    Attributes:
        id (int): A unique 128-bit identifier derived from UUID v4. Used internally
            for tracking and serialization purposes.
        hex (str): A 32-character hexadecimal string representation of the handle ID,
            useful for debugging and logging purposes.

    Note:
        This class implements `__hash__` to allow handles to be used as dictionary
        keys or in sets for efficient collection operations.

    Example:
        >>> handle = controller.submit_experiment(scheduled_experiment)
        >>> print(handle.hex)  # 32-character hex string
        >>> # Use with controller API:
        >>> status = controller.get_experiment_status(handle)
    """

    def __init__(self, id: int | None = None):
        """Initialize a new submission handle.

        Args:
            id: Optional pre-existing handle ID. If None, a new
                UUID v4-based identifier is automatically generated.
        """
        self.id = uuid.uuid4().int if id is None else id

    def __eq__(self, other: object) -> bool:
        return isinstance(other, SubmissionHandle) and self.id == other.id

    def __hash__(self) -> int:
        return self.id

    @property
    def hex(self) -> str:
        """Hexadecimal string representation of the handle ID."""
        return "%032x" % self.id


class SubmissionRegistry:
    """Maps SubmissionHandle to ControllerSubmission instances."""

    def __init__(self) -> None:
        self._submissions: dict[SubmissionHandle, ControllerSubmission] = {}

    def ensure_unique_handle(self, handle: SubmissionHandle | None) -> SubmissionHandle:
        """Return a valid, unused handle - create one if None, raise if already in use."""
        if handle is None:
            return SubmissionHandle()
        if handle in self._submissions:
            raise LabOneQControllerException("Handle already in use.")
        return handle

    def add(self, handle: SubmissionHandle, submission: ControllerSubmission) -> None:
        self._submissions[handle] = submission

    def get(self, handle: SubmissionHandle) -> ControllerSubmission:
        submission = self._submissions.get(handle)
        if submission is None:
            raise LabOneQControllerException(f"Experiment {handle.hex} not found")
        return submission

    def remove(self, handle: SubmissionHandle) -> None:
        self._submissions.pop(handle)

    def __iter__(self):
        # Snapshot to allow remove() calls during iteration (e.g. in close/aclose)
        return iter(list(self._submissions))

    def __contains__(self, handle: SubmissionHandle) -> bool:
        return handle in self._submissions

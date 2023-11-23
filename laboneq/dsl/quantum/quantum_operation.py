# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
import uuid
from os import PathLike
from typing import Dict, Optional, Tuple, Union

from laboneq.dsl.experiment.section import Section
from laboneq.dsl.quantum.qubit import QuantumElement
from laboneq.dsl.serialization import Serializer

QuantumElementTuple = Tuple[QuantumElement, ...]


class QuantumOperation:
    """A class for quantum operations."""

    def __init__(
        self,
        uid: str | None = None,
        operation_map: Optional[
            Dict[Union[QuantumElement, QuantumElementTuple], Section]
        ] = None,
    ) -> None:
        """
        Initialize a new QuantumOperation object.

        Args:
            uid: A unique identifier for the quantum operation.
            operation_map: A dictionary of sections associated with quantum elements.
        """
        if uid is None:
            self.uid = uuid.uuid4().hex
        else:
            self.uid = uid
        self.operation_map = (
            {}
            if operation_map is None
            else {
                k if isinstance(k, str) else self._get_uids(k): v
                for k, v in operation_map.items()
            }
        )

    def _get_uids(self, elements: Union[QuantumElement, QuantumElementTuple]) -> str:
        if isinstance(elements, QuantumElement):
            return elements.uid
        else:
            return "/".join(elem.uid for elem in elements)

    def add_operation(
        self,
        elements: Union[QuantumElement, QuantumElementTuple],
        section: Section,
    ):
        """Add a section to the quantum operation.

        Args:
            elements: The quantum element(s) to which the section is associated.
            section: The section to add to the quantum operation.
        """
        self.operation_map[self._get_uids(elements)] = section

    def __call__(self, elements: QuantumElementTuple):
        try:
            section = self.operation_map[self._get_uids(elements)]
        except KeyError:
            raise ValueError(
                "No section defined for the given quantum elements"
            ) from None
        return section

    @classmethod
    def load(cls, filename: Union[str, bytes, "PathLike"]) -> "QuantumOperation":
        """
        Load a QuantumOperation object to a JSON file.

        Args:
            filename: The name of the JSON file to load the QuantumOperation object.
        """

        return Serializer.from_json_file(filename, cls)

    def save(self, filename: Union[str, bytes, "PathLike"]):
        """
        Save a QuantumOperation object to a JSON file.

        Args:
            filename: The name of the JSON file to save the QuantumOperation object.
        """
        Serializer.to_json_file(self, filename)

    @staticmethod
    def from_dict(tuneup: Dict[Union[QuantumElement, QuantumElementTuple], Section]):
        """Create a QuantumOperation object from a dictionary.

        Args:
            tuneup: A dictionary of quantum elements and sections.
        """
        op = QuantumOperation()
        for elements, section in tuneup.items():
            op.add_operation(elements, section)
        return op

    def __eq__(self, __value: object) -> bool:
        if isinstance(__value, QuantumOperation):
            return (
                self.operation_map == __value.operation_map and self.uid == __value.uid
            )
        return False

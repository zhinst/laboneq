# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0


import uuid
from os import PathLike
from typing import Dict, Optional, Tuple, Union

from laboneq.dsl.device.qubits import QuantumElement
from laboneq.dsl.experiment.section import Section
from laboneq.dsl.serialization import Serializer

QuantumElementTuple = Tuple[QuantumElement, ...]


class QuantumOperation:
    """A class for quantum operations."""

    def __init__(
        self, uid: Optional[str] = None, lookup: Optional[Dict[str, Section]] = None
    ) -> None:
        """
        Initializes a new QuantumOperation object.

        Args:
            uid: A unique identifier for the quantum operation.
            lookup: A dictionary of sections associated with the uids of quantum elements.
        """
        if uid is None:
            self.uid = uuid.uuid4().hex
        else:
            self.uid = uid
        self.lookup = {} if lookup is None else lookup

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
        self.lookup[self._get_uids(elements)] = section

    def __call__(self, elements: QuantumElementTuple):
        try:
            section = self.lookup[self._get_uids(elements)]
        except KeyError:
            raise ValueError("No section defined for the given quantum elements")
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
        """Creates a QuantumOperation object from a dictionary.

        Args:
            tuneup: A dictionary of quantum elements and sections.
        """
        op = QuantumOperation()
        for elements, section in tuneup.items():
            op.add_operation(elements, section)
        return op

    def __eq__(self, __value: object) -> bool:
        if isinstance(__value, QuantumOperation):
            return self.lookup == __value.lookup and self.uid == __value.uid
        return False

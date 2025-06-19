# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from laboneq.dsl.quantum.qpu import QPU
from laboneq.dsl.quantum.qpu_topology import TopologyEdge
from laboneq.serializers.base import VersionedClassSerializer
from laboneq.serializers.core import from_dict, import_cls, to_dict
from laboneq.serializers.serializer_registry import serializer
from laboneq.serializers.types import (
    DeserializationOptions,
    JsonSerializableType,
    SerializationOptions,
)


@serializer(types=[QPU], public=True)
class QPUSerializer(VersionedClassSerializer[QPU]):
    SERIALIZER_ID = "laboneq.serializers.implementations.QPUSerializer"
    VERSION = 2

    @classmethod
    def _edge_to_dict_v2(cls, edge: TopologyEdge) -> dict[str, object]:
        """Convert an edge to a dict for serialization."""
        return {
            "tag": edge.tag,
            "source_node": edge.source_node.uid,
            "target_node": edge.target_node.uid,
            "parameters": to_dict(edge.parameters),
            "quantum_element": edge.quantum_element.uid
            if edge.quantum_element is not None
            else None,
        }

    @classmethod
    def _edge_from_dict_v2(cls, edge: dict) -> dict:
        """Convert a serialized edge dictionary to keyword arguments for .add_edge."""
        return {
            "tag": edge["tag"],
            "source_node": edge["source_node"],
            "target_node": edge["target_node"],
            "parameters": from_dict(edge["parameters"]),
            "quantum_element": edge["quantum_element"],
        }

    @classmethod
    def to_dict(
        cls, obj: QPU, options: SerializationOptions | None = None
    ) -> JsonSerializableType:
        # Note: The "topology_edges" entry in "__data__" is optional in
        # v2 of the serializer.
        qop_cls = obj.quantum_operations.__class__
        topology_edges = [cls._edge_to_dict_v2(edge) for edge in obj.topology.edges()]

        return {
            "__serializer__": cls.serializer_id(),
            "__version__": cls.version(),
            "__data__": {
                "quantum_elements": [to_dict(q) for q in obj.quantum_elements],
                # We should use __qualname__ here but that complicates things
                # for import_cls
                "quantum_operations_class": f"{qop_cls.__module__}.{qop_cls.__name__}",
                "topology_edges": topology_edges,
            },
        }

    @classmethod
    def from_dict_v2(
        cls,
        serialized_data: JsonSerializableType,
        options: DeserializationOptions | None = None,
    ) -> QPU:
        # Note: The "topology_edges" entry in "__data__" is optional in
        # v2 of the serializer.

        data = serialized_data["__data__"]
        quantum_elements = [from_dict(q) for q in data["quantum_elements"]]
        qop_cls = import_cls(data["quantum_operations_class"])
        qop = qop_cls()
        qpu = QPU(
            quantum_elements=quantum_elements,
            quantum_operations=qop,
        )

        topology_edges = data.get("topology_edges")
        if topology_edges is not None:
            for edge in topology_edges:
                qpu.topology.add_edge(**cls._edge_from_dict_v2(edge))

        return qpu

    @classmethod
    def from_dict_v1(
        cls,
        serialized_data: JsonSerializableType,
        options: DeserializationOptions | None = None,
    ) -> QPU:
        data = serialized_data["__data__"]
        qubits = [from_dict(q) for q in data["qubits"]]
        qop_cls = import_cls(data["quantum_operations_class"])
        qop = qop_cls()
        return QPU(
            quantum_elements=qubits,
            quantum_operations=qop,
        )

# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

import copy
import itertools
import logging
from enum import Enum
from types import SimpleNamespace
from typing import Any, Dict, Optional

import rustworkx

from laboneq.compiler.common.event_type import EventType

_logger = logging.getLogger(__name__)


class EventRelation(Enum):
    AFTER = "AFTER"
    AFTER_OR_AT = "AFTER_OR_AT"
    AFTER_AT_LEAST = "AFTER_AT_LEAST"
    AFTER_EXACTLY = "AFTER_EXACTLY"
    AFTER_LOOP = "AFTER_LOOP"
    BEFORE = "BEFORE"
    BEFORE_OR_AT = "BEFORE_OR_AT"
    BEFORE_AT_LEAST = "BEFORE_AT_LEAST"
    RELATIVE_BEFORE = "RELATIVE_BEFORE"
    USES_EARLY_REFERENCE = "USES_EARLY_REFERENCE"
    USES_LATE_REFERENCE = "USES_LATE_REFERENCE"


class EventGraph:
    def __init__(self, event_graph=None, node_data=None):
        if event_graph is None:
            self._event_graph = rustworkx.PyDiGraph(check_cycle=False, multigraph=False)
        else:
            self._event_graph = event_graph
        self._section_events = {}
        if node_data is None:
            self._node_data: Dict[int, Any] = {}
        else:
            self._node_data = copy.deepcopy(node_data)
            for node in self._node_data.values():
                section_name = node.get("section_name")
                event_type = node.get("event_type")
                if section_name is not None and event_type is not None:
                    self._cache_section_event(section_name, event_type, node)

    def after_or_at(self, a, b):
        self._event_graph.add_edge(a, b, {"relation": EventRelation.AFTER_OR_AT})

    def after_exactly(self, a, b, delta):
        if delta is None:
            raise Exception(
                f"Trying to add after_exactly node between {a} and {b}, but delta is None"
            )
        self._event_graph.add_edge(
            a, b, {"relation": EventRelation.AFTER_EXACTLY, "delta": delta}
        )

    def uses_early_reference(self, a, b):
        self._event_graph.add_edge(
            a, b, {"relation": EventRelation.USES_EARLY_REFERENCE}
        )

    def uses_late_reference(self, a, b):
        self._event_graph.add_edge(
            a, b, {"relation": EventRelation.USES_LATE_REFERENCE}
        )

    def before_or_at(self, a, b):
        self._event_graph.add_edge(a, b, {"relation": EventRelation.BEFORE_OR_AT})

    def relative_before(self, a, b):
        self._event_graph.add_edge(a, b, {"relation": EventRelation.RELATIVE_BEFORE})

    def after_at_least(self, a, b, delta):
        if delta is None:
            raise Exception(
                f"Trying to add after_at_least node between {a} and {b}, but delta is None"
            )

        self._event_graph.add_edge(
            a, b, {"relation": EventRelation.AFTER_AT_LEAST, "delta": delta}
        )

    def before_at_least(self, a, b, delta):
        self._event_graph.add_edge(
            a, b, {"relation": EventRelation.BEFORE_AT_LEAST, "delta": delta}
        )

    def add_edge(self, a, b, relation):
        self._event_graph.add_edge(a, b, {"relation": relation})

    def edge_data(self, a, b):
        return self._event_graph.get_edge_data(a, b)

    def edge_list(self):
        return self._event_graph.weighted_edge_list()

    def add_node(self, **kwargs):
        node_id = self._event_graph.add_node({})
        attributes: Dict = {}
        for key, value in kwargs.items():
            attributes[key] = value
        attributes["id"] = node_id

        self._node_data[node_id] = attributes

        _logger.debug("Added node %s to nx graph", node_id)

        new_node = attributes

        if "section_name" in kwargs:
            section_name = kwargs.get("section_name")
            if "event_type" in kwargs:
                event_type = kwargs.get("event_type")
                self._cache_section_event(section_name, event_type, new_node)

        return node_id

    def _cache_section_event(self, section_name, event_type, node):
        if section_name not in self._section_events:
            self._section_events[section_name] = {}
        section_events = self._section_events[section_name]
        if event_type not in section_events:
            section_events[event_type] = []
        section_events[event_type].append(node)

    def find_section_events_by_type(
        self,
        section_name: str,
        event_type: Optional[str] = None,
        properties: Dict[str, Any] = None,
    ):
        if event_type is None:
            events = list(
                itertools.chain.from_iterable(
                    self._section_events[section_name].values()
                )
            )
        else:
            if event_type in self._section_events[section_name]:
                events = self._section_events[section_name][event_type]
            else:
                return []
        retval = []
        if properties is None:
            return [event for event in events]
        for event in events:
            match = True
            if properties is not None:
                for k, v in properties.items():
                    if not k in event or event[k] != v:
                        match = False
                        break
            if match:
                retval.append(event)
        return retval

    def find_section_start_end(self, section_name):
        retval = SimpleNamespace()
        section_events = self._section_events.get(section_name, {})
        for event in reversed(section_events.get(EventType.SECTION_START, [])):
            if not event.get("shadow", False):
                setattr(retval, "start", event["id"])
                break
        for event in reversed(section_events.get(EventType.SECTION_END, [])):
            if not event.get("shadow", False):
                setattr(retval, "end", event["id"])
                break
        for event in reversed(section_events.get(EventType.SKELETON, [])):
            if event.get("skeleton_of") == EventType.SECTION_END:
                if not event.get("shadow", False):
                    setattr(retval, "skeleton_end", event["id"])
                    break
        for event in reversed(section_events.get(EventType.SKELETON, [])):
            if event.get("skeleton_of") == EventType.SECTION_START:
                if not event.get("shadow", False):
                    setattr(retval, "skeleton_start", event["id"])
                    break

        return retval

    def node(self, id) -> Any:
        return self._node_data[id]

    def descendants(self, id):
        return rustworkx.descendants(self._event_graph, id)

    def log_graph_lines(self):
        log_lines = []
        sequence_nr = 0
        for event_id in self.node_ids():
            event_data = self._node_data[event_id]

            log_lines.append(f"Event {sequence_nr}:  {event_data}")
            for edge in self._event_graph.out_edges(event_id):
                node1 = self._node_data[edge[0]]
                node2 = self._node_data[edge[1]]
                delta = edge[2].get("delta")
                if delta is None:
                    delta = ""

                log_lines.append(
                    f"  Edge   {node_info(node1)} ( {edge[2]['relation'].name} {delta} -> ) {node_info(node2)}"
                )
            sequence_nr += 1
        return log_lines

    def log_graph(self):
        for line in self.log_graph_lines():
            _logger.debug(line)

    def out_edges(self, event_id):
        return self._event_graph.out_edges(event_id)

        # neighbors = self._event_graph.adj_direction(event_id, True)
        # return [ (event_id,k,v) for k,v in neighbors.items()]

    def in_edges(self, event_id):
        return self._event_graph.in_edges(event_id)
        # neighbors = self._event_graph.adj_direction(event_id, False)
        # return [ (event_id,k,v) for k,v in neighbors.items()]

    def node_ids(self):
        return self._event_graph.node_indices()

    def sorted_events(self):

        _logger.debug(
            "Sorting event graph (%d nodes, %d edges )",
            self._event_graph.num_nodes(),
            self._event_graph.num_edges(),
        )
        try:
            sorted_events = rustworkx.topological_sort(self._event_graph)[::-1]
        except rustworkx.DAGHasCycle as ex:
            _logger.error("Event graph has cycles:")
            first_cycle = [
                e[0] for e in rustworkx.digraph_find_cycle(self._event_graph)
            ]
            _logger.debug("First cycle: %s", first_cycle)
            for node in first_cycle:
                graph_node = self.node(node)
                _logger.debug("   node_info=%s", node_info(graph_node))
                for edge in self.out_edges(node):
                    member = "   "
                    if edge[1] in first_cycle:
                        member = "***"
                    _logger.debug("   %s out edge: %s", member, edge)

            raise ex

        _logger.debug("Event graph sorted")
        return sorted_events

    def set_node_attributes(self, node_id, attributes):
        for k, v in attributes.items():
            self._node_data[node_id][k] = v

    def set_edge_attribute(self, node1_id, node2_id, key, value):
        self._event_graph.get_edge_data(node1_id, node2_id)[key] = value

    def visualization_dict(self, node_filter=None):
        node_dir = {}
        out_graph = {"nodes": [], "links": []}
        out_node_ids = set()
        for node_id in self.node_ids():
            node = self.node(node_id)
            if node_filter is not None:
                if not node_filter(node):
                    continue
            out_node_ids.add(node_id)
            event_type = node["event_type"]
            if str(node["event_type"]) == "SKELETON":
                event_type = "SK_" + node.get("skeleton_of")

            node_key = event_type
            if node.get("section_name") is not None:
                node_key += "_" + node.get("section_name")
            node_dir[node_id] = node_key
            out_graph["nodes"].append(
                {
                    "id": str(node["id"]),
                    "key": node_key,
                    "group": 1,
                    "time_ns": node.get("time", node["id"] / 1e9 * 100) * 1e9,
                    "orig_id": node["id"],
                    "play_wave_id": node.get("play_wave_id"),
                    "section_name": node.get("section_name"),
                    "event_type": node.get("event_type"),
                    "subsection_name": node.get("subsection_name"),
                }
            )

        first_cycle_edges = []
        try:
            rustworkx.topological_sort(self._event_graph)[::-1]
        except rustworkx.DAGHasCycle:

            # Note: If root is not given, tha algorithm below will pick one random node, and then sometimes find a cycle, sometimes not
            # Whether the root we pick here is a good one is currently unclear. The networkx implementation of find_cycle
            # is much more robust, the documentation says: "
            #     source: The node from which the traversal begins.
            #     If None, then a source is chosen arbitrarily and repeatedly until all edges from each node in the graph are searched"
            # we implement it now in an inefficient but effective way; this is for development-time visualization only, so it should not matter
            for root in self._event_graph.node_indices():
                first_cycle_edges = list(
                    rustworkx.digraph_find_cycle(self._event_graph, root)
                )
                if len(first_cycle_edges) > 0:
                    break

        _logger.debug(
            "first_cycle_edges=%s, edges = %s",
            first_cycle_edges,
            list(self._event_graph.edge_list()),
        )
        for edge in self.edge_list():
            out_edge = {"source": str(edge[0]), "target": str(edge[1])}
            if not (edge[0] in out_node_ids and edge[1] in out_node_ids):
                continue
            if (edge[0], edge[1]) in first_cycle_edges:
                out_edge["cycle"] = 1
            else:
                out_edge["cycle"] = None
            out_edge["relation"] = edge[2]["relation"].value
            out_graph["links"].append(out_edge)

        return out_graph


def node_info(node):
    play_wave = ""
    if "play_wave_id" in node:
        play_wave = node["play_wave_id"]
    section = ""
    if "section_name" in node:
        section = node["section_name"]
    skeleton_of = ""
    if "skeleton_of" in node:
        skeleton_of = " skeleton of " + node["skeleton_of"]
    node_id = "UNKNOWN_ID"
    if "id" in node:
        node_id = node["id"]
    event_type = "UNKNOWN_EVENT_TYPE"
    if "event_type" in node:
        event_type = node["event_type"]
    return (
        str(node_id) + " " + event_type + " " + section + " " + play_wave + skeleton_of
    )

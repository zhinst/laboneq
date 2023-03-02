# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

import copy
import json
import logging
from typing import Dict, List, Tuple

import rustworkx as rx

from laboneq.compiler.experiment_access.experiment_dao import ExperimentDAO, SectionInfo

_logger = logging.getLogger(__name__)


class SectionGraph:
    def __init__(
        self,
        section_graph: rx.PyDiGraph,
        section_graph_parents: rx.PyDiGraph,
        node_ids: Dict[str, int],
        section_infos: Dict[str, SectionInfo],
        root_sections: List[str],
    ):

        self._section_graph = section_graph
        self._section_graph_parents = section_graph_parents
        self._node_ids = node_ids
        self._section_infos = section_infos
        self._root_sections = root_sections

    def section_info(self, section_id) -> SectionInfo:
        return self._section_infos[section_id]

    def sections(self):
        return [node["section_id"] for node in self._section_graph.nodes()]

    def topologically_sorted_sections(self) -> List[str]:
        return [
            self._section_graph[node]["section_id"]
            for node in rx.topological_sort(self._section_graph)
        ]

    def has_cycles(self):
        try:
            rx.topological_sort(self._section_graph)
        except rx.DAGHasCycle:
            return True
        return False

    def parent(self, section):
        node_id = self._node_ids[section]
        edges_list = self._section_graph.in_edges(node_id)
        parent = None
        for edge in edges_list:
            if edge[2]["type"] == "parent":
                parent = edge[0]
        if parent is None:
            return None
        return self._section_graph[parent]["section_id"]

    def followers(self, section):
        node = self._node_ids[section]
        return [
            self._section_graph[edge[1]]["section_id"]
            for edge in self._section_graph.out_edges(node)
            if edge[2]["type"] == "previous"
        ]

    def section_children(self, section_name):
        node = self._node_ids[section_name]
        return [
            self._section_graph_parents[node]["section_id"]
            for node in rx.descendants(self._section_graph_parents, node)
        ]

    def depth_map(self) -> Dict[str, int]:
        root_sections = self.root_sections()
        if len(root_sections) == 0:
            return {}
        result = {}
        for root_section in root_sections:
            root_node = self._node_ids[root_section]
            paths = rx.digraph_dijkstra_shortest_paths(
                self._section_graph_parents, root_node
            )
            result.update(
                {
                    self._section_graph[destination_node]["section_id"]: len(path)
                    for destination_node, path in paths.items()
                }
            )
            result[root_section] = 0
        return result

    def subsection_map(self):
        return {section: self.section_children(section) for section in self.sections()}

    def preorder_map(self):
        new_graph: rx.PyDiGraph = self._section_graph.copy()
        new_graph.remove_edges_from(new_graph.edge_list())
        for node in self._section_graph.node_indices():
            prio_edge = None
            for u, v, data in self._section_graph.in_edges(node):
                if data["type"] == "previous" or prio_edge is None:
                    prio_edge = u, v
            if prio_edge is not None:
                new_graph.add_edge(*prio_edge, None)
        new_edges = set()
        for node in new_graph.node_indices():
            for parent, _, edge_data in self._section_graph.in_edges(node):
                if edge_data["type"] != "parent":
                    continue
                _logger.debug("Found parent relation between %s and %s", parent, node)
                for u, v, sequential_edge in new_graph.out_edges(parent):
                    if v == node:
                        continue
                    _logger.debug("sequential_edge=%s", sequential_edge)
                    if not self._section_graph_parents.has_edge(u, v):
                        _logger.debug(
                            "descendants of %s : %s",
                            sequential_edge,
                            rx.descendants(self._section_graph_parents, u),
                        )
                        for descendant in rx.descendants(
                            self._section_graph_parents, u
                        ):
                            if v == descendant:
                                continue
                            _logger.debug(
                                "adding edge %s",
                                (descendant, v),
                            )
                            new_edges.add((node, v, None))
        new_graph.add_edges_from(tuple(new_edges))
        toposorted = list(rx.topological_sort(new_graph))
        current_level = 0
        preorder_map = {}
        depth_map = self.depth_map()
        for first, second in list(zip(toposorted, toposorted[1:] + [None])):
            _logger.debug(
                "first=%s second=%s current_level=%d", first, second, current_level
            )
            first_section_id = self._section_graph[first]["section_id"]
            second_section_id = (
                self._section_graph[second]["section_id"]
                if second is not None
                else None
            )
            preorder_map[first_section_id] = current_level
            if second is not None:
                in_edges = self._section_graph.in_edges(second)
                _logger.debug("%s in_edges=%s", second, in_edges)

            is_independent = True
            if (
                second is not None
                and self._section_graph.has_edge(first, second)
                and self._section_graph.has_edge(first, second)
                and self._section_graph.get_edge_data(first, second)["type"]
                == "previous"
            ):
                is_independent = False
            if (
                second is not None
                and first_section_id in depth_map
                and second_section_id in depth_map
                and (
                    depth_map[first_section_id] != depth_map[second_section_id]
                    or is_independent
                )
            ):
                current_level += 1
        return preorder_map

    def root_sections(self):
        return self._root_sections

    @staticmethod
    def _compute_root_sections(
        section_instance_tree: rx.PyDiGraph, section_infos, node_ids
    ):
        nt_parents_of_rt = set()

        section_graph_parents = section_instance_tree.copy()
        section_graph_parents.remove_edges_from(
            [
                (u, v)
                for (u, v, data) in section_instance_tree.weighted_edge_list()
                if data["type"] != "parent"
            ]
        )

        rt_root_sections: List[str] = []
        for node in section_graph_parents.node_indices():
            section_id = section_graph_parents[node]["section_id"]
            if section_infos[section_id].execution_type == "controller":
                continue

            edges_list = section_graph_parents.in_edges(node)
            parent = None
            for u, v, data in edges_list:
                if data["type"] == "parent":
                    parent = u
                    break
            if parent is None:
                nt_parents_of_rt.add(None)
                rt_root_sections.append(section_id)
            else:
                parent_section_id: str = section_graph_parents[parent]["section_id"]
                if section_infos[parent_section_id].execution_type == "controller":
                    nt_parents_of_rt.add(parent_section_id)
                    rt_root_sections.append(section_id)

        if len(rt_root_sections) == 0:
            return []

        if len(nt_parents_of_rt) != 1:
            raise RuntimeError(
                "Real-time root sections must all be located at the same level."
            )

        for nt_section in nt_parents_of_rt:
            if nt_section is None:
                children = rt_root_sections
            else:
                children = [
                    section_graph_parents[child]["section_id"]
                    for child in rx.descendants(
                        section_graph_parents, node_ids[nt_section]
                    )
                ]

            for child in children:
                if section_infos[child].execution_type == "controller":
                    raise RuntimeError(
                        "Real-time root sections cannot be siblings of near-time sections."
                    )

        return rt_root_sections

    @staticmethod
    def from_dao(experiment_dao: ExperimentDAO):
        section_infos: Dict[str, SectionInfo] = {}
        section_graph_instances = rx.PyDiGraph(multigraph=False)
        node_ids: Dict[str, int] = {}  # look-up node IDs by section ID

        # Note: sections directly at the root of the experiment are are not linked to
        # be played back sequentially even if they share signals.
        # Such an arrangement is invalid as per our DSL rules, so this is not an
        # immediate issue.

        for section_id in experiment_dao.sections():
            section_info = experiment_dao.section_info(section_id)
            section_infos[section_id] = section_info
            node_id = section_graph_instances.add_node(section_id)
            node_ids[section_id] = node_id

        link_nodes: Dict[Tuple[str, str], int] = {}

        for section_id in experiment_dao.sections():
            previous_instance_signals = {}
            section_node_id = node_ids[section_id]
            direct_section_children = experiment_dao.direct_section_children(section_id)
            for i, section_ref_id in enumerate(direct_section_children):
                ref_section_node_id = node_ids[section_ref_id]
                ref_section_info = experiment_dao.section_info(section_ref_id)
                link_node_id = section_graph_instances.add_node(
                    f"{section_ref_id}_{section_id}_{i}"
                )
                link_nodes[(section_id, section_ref_id)] = link_node_id
                section_graph_instances.add_edge(
                    ref_section_node_id, link_node_id, dict(type="referenced_through")
                )
                section_graph_instances.add_edge(
                    link_node_id, section_node_id, dict(type="referenced_by")
                )
                current_signals = experiment_dao.section_signals_with_children(
                    section_ref_id
                )

                # Sections with shared signals are played after each other, except for
                # the branches of a Match section, when they are played in parallel:
                if not section_infos[section_id].handle:
                    for previous_node_id, signals in previous_instance_signals.items():
                        common_signals = signals.intersection(current_signals)
                        if len(common_signals) > 0:
                            section_graph_instances.add_edge(
                                previous_node_id, link_node_id, dict(type="previous")
                            )
                previous_instance_signals[link_node_id] = current_signals

                # Sections follow a previous section specified via play_after, except
                # for the branches of a match section, where we don't allow that:
                play_after = ref_section_info.play_after or []
                if isinstance(play_after, str):
                    play_after = [play_after]
                if section_infos[section_id].handle:
                    play_after = []
                for pa in play_after:
                    for j, sibling_section_ref_id in enumerate(direct_section_children):
                        if j == i:  # That's us
                            raise ValueError(
                                f"Could not find section {pa} mentioned in play_after "
                                f"of {section_ref_id}."
                            )
                        if sibling_section_ref_id == pa:
                            play_after_link_node_id = link_nodes[
                                (section_id, sibling_section_ref_id)
                            ]
                            section_graph_instances.add_edge(
                                play_after_link_node_id,
                                link_node_id,
                                dict(type="previous"),
                            )
                            break

        parent_edges = [
            (a, b)
            for (a, b, data) in section_graph_instances.weighted_edge_list()
            if data["type"] != "previous"
        ]

        if len(parent_edges) > 0:
            section_graph_instances_parents = rx.PyDiGraph.edge_subgraph(
                section_graph_instances, parent_edges
            )
        else:
            section_graph_instances_parents = copy.deepcopy(section_graph_instances)

        root_nodes = (
            node
            for node in section_graph_instances_parents.node_indices()
            if section_graph_instances_parents.out_degree(node) == 0
        )

        path_dict: Dict[Tuple[int, ...], str] = {}
        for root_node in root_nodes:
            path_dict[(root_node,)] = section_graph_instances[root_node]
            for section in experiment_dao.sections():
                node_id = node_ids[section]
                paths = rx.all_simple_paths(
                    section_graph_instances_parents, from_=node_id, to=root_node
                )

                if len(paths) == 1:
                    path_dict[tuple(paths[0])] = section
                else:
                    for i, path in enumerate(paths):
                        path_dict[tuple(path)] = f"{section}_{i}"

        for k, section in path_dict.items():
            _logger.debug("path_dict %s : %s", k, section)

        section_instance_tree = rx.PyDiGraph(multigraph=False)
        section_instance_tree_node_ids = {}

        for k, section in path_dict.items():
            section_info: SectionInfo = copy.deepcopy(
                section_infos[section_graph_instances_parents[k[0]]]
            )
            section_info.section_display_name = section_info.section_id
            section_info.section_id = section
            section_infos[section] = section_info
            if len(k) > 1:
                section_link_id = k[1]
            else:
                section_link_id = k[0]

            node_id = section_instance_tree.add_node(
                dict(
                    section_id=section,
                    section_link_id=section_link_id,
                )
            )
            assert node_id not in node_ids
            section_instance_tree_node_ids[section] = node_id

        for k, section in path_dict.items():
            cur_node = section_instance_tree_node_ids[section]
            path_list = list(k)
            while len(path_list) > 2:
                rest_key = tuple(path_list[2:])
                rest_path_entry = section_instance_tree_node_ids[path_dict[rest_key]]

                section_instance_tree.add_edge(
                    rest_path_entry, cur_node, dict(type="parent")
                )

                path_list = path_list[2:]
                cur_node = rest_path_entry

        for node_id in section_instance_tree.node_indices():
            try:
                parent = next(
                    u
                    for u, _, data in section_instance_tree.in_edges(node_id)
                    if data["type"] == "parent"
                )
            except StopIteration:
                siblings = {}
            else:
                siblings = {
                    section_instance_tree[e[1]]["section_link_id"]: e[1]
                    for e in section_instance_tree.out_edges(parent)
                }

            _logger.debug("node_id=%i siblings=%s", node_id, siblings)
            section_link_id = section_instance_tree[node_id]["section_link_id"]

            for edge in section_graph_instances.out_edges(section_link_id):
                node_from, node_to, edge_data = edge
                if edge_data["type"] == "previous":
                    linked_sibling = siblings[node_to]

                    _logger.debug(
                        "node_id=%i section_link_id=%i edge=%s linked_sibling=%i",
                        node_id,
                        section_link_id,
                        edge,
                        linked_sibling,
                    )
                    section_instance_tree.add_edge(
                        node_id, linked_sibling, dict(type="previous")
                    )

        near_time_sections = [
            n
            for n in section_instance_tree.node_indices()
            if section_infos[section_instance_tree[n]["section_id"]].execution_type
            == "controller"
        ]
        # Note: we cannot use PyDiGraph.subgraph(), as it invalidates all node indices.
        # (Node indices in the new graph are a contiguous range starting from 0.)
        section_graph_rt = section_instance_tree.copy()
        section_graph_rt.remove_nodes_from(near_time_sections)

        if len(section_graph_rt.edges()) > 0:
            section_graph_parents: rx.PyDiGraph = section_graph_rt.copy()
            section_graph_parents.remove_edges_from(
                [
                    (u, v)
                    for (u, v, data) in section_graph_rt.weighted_edge_list()
                    if data["type"] != "parent"
                ]
            )

            assert len(section_graph_rt.nodes()) == len(section_graph_parents.nodes())
        else:
            section_graph_parents = copy.deepcopy(section_graph_rt)

        root_rt_sections = SectionGraph._compute_root_sections(
            section_instance_tree, section_infos, section_instance_tree_node_ids
        )

        return SectionGraph(
            section_graph=section_graph_rt,
            section_graph_parents=section_graph_parents,
            node_ids=section_instance_tree_node_ids,
            section_infos=section_infos,
            root_sections=root_rt_sections,
        )

    def json_node_link_data(self):
        return json.loads(rx.node_link_json(self._section_graph))

    def as_primitive(self):
        section_graph = [
            {
                "from": self._section_graph[edge[0]]["section_id"],
                "to": self._section_graph[edge[1]]["section_id"],
                **edge[2],
            }
            for edge in self._section_graph.weighted_edge_list()
        ]

        def sort_key(d):
            return d["from"], d["to"]

        section_graph.sort(key=sort_key)
        return section_graph

    def visualization_dict(self):
        out_graph = {"nodes": [], "links": []}
        link_data = self.json_node_link_data()
        for node in link_data["nodes"]:
            out_graph["nodes"].append({"id": node["id"], "key": node["id"]})
        for link in link_data["links"]:
            out_graph["links"].append(
                {
                    "source": link["source"],
                    "target": link["target"],
                    "relation": link["type"],
                }
            )

        return out_graph

    def log_graph(self):
        _logger.debug("++++Section Graph")

        for node in self._section_graph.node_indices():
            for parent, child, data in self._section_graph.out_edges(node):
                _logger.debug(
                    "%s, %s, %s",
                    self._section_graph[parent]["section_id"],
                    self._section_graph[child]["section_id"],
                    data,
                )

        for section in reversed(self.topologically_sorted_sections()):
            section_children = self.section_children(section)
            _logger.debug("Children of %s: %s", section, section_children)
        _logger.debug("++++ END ++++Section Graph")

    @staticmethod
    def from_json_node_link(d):
        graph = rx.PyDiGraph()
        node_ids = {}
        section_infos = {}
        for node in d["nodes"]:
            section_id = node["id"]
            section_infos[section_id] = SectionInfo(**node["section"])
            node_ids[section_id] = graph.add_node(dict(section_id=section_id))

        for edge_data in d["links"]:
            graph.add_edge(
                node_ids[edge_data["source"]], node_ids[edge_data["target"]], edge_data
            )

        section_graph_parent: rx.PyDiGraph = graph.copy()
        for u, v, data in graph.weighted_edge_list():
            if data.get("type") != "parent":
                section_graph_parent.remove_edge(u, v)

        # Note: We only use this function for internal unit testing. Before this can be
        # used in production code, we need to handle the RT subgraph properly.

        return SectionGraph(
            section_graph=graph,
            section_graph_parents=section_graph_parent,
            node_ids=node_ids,
            section_infos=section_infos,
            root_sections=SectionGraph._compute_root_sections(
                graph, section_infos, node_ids
            ),
        )

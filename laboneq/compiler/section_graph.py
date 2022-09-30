# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from operator import itemgetter
import networkx as nx
from networkx.readwrite import json_graph
import logging
import copy

from .fastlogging import NullLogger

_logger = logging.getLogger(__name__)
if _logger.getEffectiveLevel() == logging.DEBUG:
    _dlogger = _logger
else:
    _logger.info("Debug logging disabled for %s", __name__)
    _dlogger = NullLogger()


class SectionGraph:
    def __init__(
        self,
        section_instance_tree,
        section_graph,
        section_graph_parents,
        section_infos=None,
    ):

        self._section_graph = section_graph
        self._section_instance_tree = section_instance_tree
        self._section_graph_parents = section_graph_parents
        if section_infos is None:
            self._section_infos = {
                k: v["section"] for k, v in section_instance_tree.nodes(data=True)
            }
        else:
            self._section_infos = section_infos

    def section_info(self, section_id):
        return self._section_infos[section_id]

    def json_node_link_data(self):
        return json_graph.node_link_data(self._section_graph)

    def topologically_sorted_sections(self):
        return nx.topological_sort(self._section_graph)

    def cycles(self):
        return nx.simple_cycles(self._section_graph)

    def parent(self, section):
        edges_list = self._section_graph.in_edges([section], data=True)
        parent = None
        for edge in edges_list:
            if edge[2]["type"] == "parent":
                parent = edge[0]
        return parent

    def root_section(self):
        try:
            return next(v for v, d in self._section_instance_tree.in_degree() if d == 0)
        except StopIteration:
            return None

    def depth_map(self):
        root_section = self.root_section()
        if root_section is None:
            return {}
        return {
            item[0]: item[1]
            for item in nx.shortest_path_length(
                self._section_instance_tree, root_section
            ).items()
        }

    def subsection_map(self):
        return {
            node: list(nx.descendants(self._section_graph_parents, node))
            for node in self._section_graph_parents.nodes
        }

    def as_section_graph(self):
        section_graph = [
            dict({"from": edge[0], "to": edge[1]}, **edge[2])
            for edge in self._section_instance_tree.edges(data=True)
        ]
        return section_graph

    def followers(self, section):
        return [
            edge[1]
            for edge in self._section_graph.out_edges(nbunch=section, data=True)
            if edge[2]["type"] == "previous"
        ]

    def preorder_map(self):
        newgraph = nx.DiGraph()
        newgraph.add_nodes_from(self._section_instance_tree.nodes())
        for node in self._section_instance_tree.nodes():
            prio_edge = None
            for edge in self._section_instance_tree.in_edges(node, data=True):
                if edge[2]["type"] == "previous":
                    prio_edge = (edge[1], edge[0])
                else:
                    if prio_edge is None:
                        prio_edge = (edge[1], edge[0])
            if prio_edge is not None:
                newgraph.add_edges_from([(prio_edge[1], prio_edge[0])])
        new_edges = []
        for node in newgraph.nodes:
            for edge in self._section_instance_tree.in_edges(node, data=True):
                if edge[2]["type"] == "parent":
                    _dlogger.debug("Found parent relation %s for %s", edge, node)
                    for sequential_edge in newgraph.out_edges(edge[0]):
                        if sequential_edge[1] != node:
                            _dlogger.debug("sequential_edge=%s", sequential_edge)
                            if not self._section_graph_parents.has_edge(
                                sequential_edge[0], sequential_edge[1]
                            ):
                                _dlogger.debug(
                                    "descendants of %s : %s",
                                    sequential_edge[0],
                                    nx.descendants(
                                        self._section_graph_parents, sequential_edge[0]
                                    ),
                                )
                                for descendant in nx.descendants(
                                    self._section_graph_parents, sequential_edge[0]
                                ):
                                    if sequential_edge[1] != descendant:
                                        _dlogger.debug(
                                            "adding edge %s",
                                            (descendant, sequential_edge[1]),
                                        )
                                        new_edges.append((node, sequential_edge[1]))
        new_edges = list(set(new_edges))
        newgraph.add_edges_from(new_edges)
        toposorted = list(nx.topological_sort(newgraph))
        current_level = 0
        preorder_map = {}
        depth_map = self.depth_map()
        for first, second in list(zip(toposorted, toposorted[1:] + [None])):
            _dlogger.debug(
                "first=%s second=%s current_level=%d", first, second, current_level
            )
            preorder_map[first] = current_level
            if second is not None:
                in_edges = self._section_instance_tree.in_edges(second, data=True)
                _dlogger.debug("%s in_edges=%s", second, in_edges)

            is_independent = True
            if second is not None:
                if self._section_instance_tree.has_edge(first, second):
                    if self._section_instance_tree.has_edge(first, second):
                        if (
                            self._section_instance_tree.edges()[(first, second)]["type"]
                            == "previous"
                        ):
                            is_independent = False
            if (
                second is not None
                and first in depth_map
                and second in depth_map
                and (depth_map[first] != depth_map[second] or is_independent)
            ):
                current_level += 1
        return preorder_map

    def parent_sections(self, section):
        parent_edges = [
            e
            for e in self._section_graph.in_edges(nbunch=section, data=True)
            if e[2]["type"] == "parent"
        ]
        return [parent_edge[0] for parent_edge in parent_edges]

    def sections(self):
        return list(self._section_graph.nodes())

    def section_children(self, section_name):
        return nx.algorithms.dag.descendants(self._section_graph_parents, section_name)

    def log_graph(self):

        _dlogger.debug("++++Section Graph")

        for node in self._section_graph.nodes:
            for edge in self._section_graph.out_edges(nbunch=[node], data=True):
                _dlogger.debug(edge)

        for node in list(
            reversed(list(nx.topological_sort(self._section_instance_tree)))
        ):
            section_children = self.section_children(node)
            _dlogger.debug("Children of %s: %s", node, section_children)
        _dlogger.debug("++++ END ++++Section Graph")

    def root_sections(self):

        root_sections = []
        for section_id in list(nx.topological_sort(self._section_graph_parents)):
            section_info = self.section_info(section_id)
            _dlogger.debug("Section: %s", section_info)
            if section_info["has_repeat"] and section_info["count"] < 1:
                raise Exception(
                    f"Repeat count must be at least 1, but section {section_id} has count={section_info['count']}"
                )

            if section_info["execution_type"] != "controller":
                # first non-controller section
                if len(root_sections) == 0:
                    _dlogger.debug(
                        "section %s is the first real-time section",
                        section_info["section_id"],
                    )
                    root_sections = [section_id]

        if len(root_sections) > 0:
            parent_edges = self._section_graph_parents.in_edges(
                nbunch=[root_sections[0]], data=True
            )
            if len(parent_edges) > 0:
                root_parent = list(parent_edges)[0][0]
                root_sections = []
                child_edges = self._section_graph_parents.out_edges(
                    nbunch=[root_parent], data=True
                )
                for edge in child_edges:
                    child_section_id = edge[1]
                    root_sections.append(child_section_id)

        root_real_time_sections = []
        root_near_time_sections = []
        for section_id in root_sections:
            section_info = self.section_info(section_id)
            (root_real_time_sections, root_near_time_sections)[
                section_info["execution_type"] == "controller"
            ].append(section_id)
        if not (len(root_real_time_sections) == 0 or len(root_near_time_sections) == 0):
            raise Exception(
                f"Root sections {root_sections} contains both real time child sections {root_real_time_sections} and near-time sections {root_near_time_sections}. Only one kind is allowed."
            )
        return root_sections

    @staticmethod
    def from_dao(experiment_dao):
        section_infos = {}
        section_graph_instances = nx.DiGraph()
        for section_id in experiment_dao.sections():
            section_info_0 = experiment_dao.section_info(section_id)
            section_infos[section_id] = section_info_0
            section_graph_instances.add_nodes_from([section_id], section=section_info_0)
        for section_id in experiment_dao.sections():
            previous_instance_signals = {}
            direct_section_children = experiment_dao.direct_section_children(section_id)
            for i, section_ref_id in enumerate(direct_section_children):
                link_node_id = section_ref_id + "_" + section_id + "_" + str(i)
                _dlogger.debug(
                    "  Child section of %s : %s link_node_id=%s",
                    section_id,
                    section_ref_id,
                    link_node_id,
                )

                ref_section_info = experiment_dao.section_info(section_ref_id)
                section_graph_instances.add_nodes_from(
                    [link_node_id], section=ref_section_info
                )
                section_graph_instances.add_edges_from(
                    [(section_ref_id, link_node_id)], type="referenced_through"
                )
                section_graph_instances.add_edges_from(
                    [(link_node_id, section_id)], type="referenced_by"
                )
                current_signals = experiment_dao.section_signals_with_children(
                    section_ref_id
                )

                for previous_node_id, signals in previous_instance_signals.items():
                    common_signals = signals.intersection(current_signals)
                    if len(common_signals) > 0:
                        section_graph_instances.add_edges_from(
                            [(previous_node_id, link_node_id)], type="previous"
                        )
                previous_instance_signals[link_node_id] = current_signals

                if ref_section_info is not None and "play_after" in ref_section_info:
                    play_after = ref_section_info["play_after"]
                    if play_after is not None and play_after != "" and play_after != []:
                        if isinstance(play_after, str):
                            play_after = [play_after]
                        for pa in play_after:
                            for j, sibling_section_ref_id in enumerate(
                                direct_section_children
                            ):
                                if j == i:  # That's us
                                    raise ValueError(
                                        f"Could not find section {pa} mentioned "
                                        + f"in play_after of {section_ref_id}."
                                    )
                                if sibling_section_ref_id == pa:
                                    play_after_link_node_id = (
                                        sibling_section_ref_id
                                        + "_"
                                        + section_id
                                        + "_"
                                        + str(j)
                                    )
                                    section_graph_instances.add_edges_from(
                                        [(play_after_link_node_id, link_node_id)],
                                        type="previous",
                                    )
                                    break

        parent_edges = [
            (y[0], y[1])
            for y in filter(
                lambda x: x[2]["type"] != "previous",
                section_graph_instances.edges(data=True),
            )
        ]
        if len(parent_edges) > 0:
            section_graph_instances_parents = nx.edge_subgraph(
                section_graph_instances, parent_edges
            )
        else:
            section_graph_instances_parents = copy.deepcopy(section_graph_instances)

        try:
            root_node = next(
                v for v, d in section_graph_instances_parents.out_degree() if d == 0
            )
        except StopIteration:
            root_node = None

        path_dict = {}
        for node in experiment_dao.sections():
            paths = list(
                nx.all_simple_paths(
                    section_graph_instances_parents, source=node, target=root_node
                )
            )

            if len(paths) == 1:
                path_dict[tuple(paths[0])] = node
            else:
                for i, path in enumerate(paths):
                    path_dict[tuple(path)] = node + "_" + str(i)
        if root_node is not None:
            path_dict[(root_node,)] = root_node
        for k, v in path_dict.items():
            _dlogger.debug("path_dict %s : %s", k, v)

        section_instance_tree = nx.DiGraph()

        for k, v in path_dict.items():
            section_info = copy.deepcopy(section_infos[k[0]])
            section_info["section_display_name"] = section_info["section_id"]
            section_info["section_id"] = v
            section_infos[v] = section_info
            if len(k) > 1:
                section_link_id = k[1]
            else:
                section_link_id = k[0]

            section_instance_tree.add_nodes_from(
                [v], section={"section_link_id": section_link_id}
            )
        for k, v in path_dict.items():
            cur_node = v
            path_list = list(k)
            while len(path_list) > 2:
                rest_key = tuple(path_list[2:])
                rest_path_entry = path_dict[rest_key]
                section_instance_tree.add_edge(rest_path_entry, cur_node, type="parent")

                section_link_id = section_instance_tree.nodes(data=True)[cur_node][
                    "section"
                ]["section_link_id"]
                path_list = path_list[2:]
                cur_node = rest_path_entry

        for node_id in section_instance_tree.nodes:

            siblings = {}
            try:
                parent = next(e[0] for e in section_instance_tree.in_edges(node_id))
                siblings = {
                    section_instance_tree.nodes(data=True)[e[1]]["section"][
                        "section_link_id"
                    ]: e[1]
                    for e in section_instance_tree.out_edges(parent)
                }
            except StopIteration:
                pass

            _dlogger.debug("node_id=%s siblings=%s", node_id, siblings)
            section_link_id = section_instance_tree.nodes(data=True)[node_id][
                "section"
            ]["section_link_id"]

            for edge in section_graph_instances.out_edges(section_link_id, data=True):
                if edge[2]["type"] == "previous":
                    linked_sibling = siblings[edge[1]]

                    _dlogger.debug(
                        "node_id=%s section_link_id=%s edge=%s linked_sibling=%s",
                        node_id,
                        section_link_id,
                        edge,
                        linked_sibling,
                    )
                    section_instance_tree.add_edge(
                        node_id, linked_sibling, type="previous"
                    )

        if len(section_instance_tree.edges) > 0:

            section_graph_parents = nx.edge_subgraph(
                section_instance_tree,
                [
                    (y[0], y[1])
                    for y in filter(
                        lambda x: x[2]["type"] == "parent",
                        section_instance_tree.edges(data=True),
                    )
                ],
            )

        else:
            section_graph_parents = copy.deepcopy(section_instance_tree)

        _dlogger.debug("***** Section graph parents:")
        for node in section_graph_parents.nodes(data=True):
            _dlogger.debug("  %s", node)
            for edge in section_graph_parents.out_edges(node[0], data=True):
                _dlogger.debug("      %s", edge)

        _dlogger.debug("END ***** Section graph parents")

        section_graph = nx.subgraph(
            section_instance_tree,
            map(
                itemgetter(0),
                filter(
                    lambda x: section_infos[x[0]]["execution_type"] != "controller",
                    section_instance_tree.nodes(data=True),
                ),
            ),
        )

        return SectionGraph(
            section_instance_tree,
            section_graph,
            section_graph_parents,
            section_infos=section_infos,
        )

    def visualization_dict(self):
        node_dir = {}
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

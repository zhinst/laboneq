# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

import copy
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

from laboneq.compiler.common.event_type import EventType
from laboneq.compiler.experiment_access.section_graph import SectionGraph
from laboneq.compiler.scheduler.event_graph import EventGraph, EventRelation

_logger = logging.getLogger(__name__)


@dataclass
class ChainElement:
    id: str
    start_type: EventType = None
    end_type: Optional[EventType] = None
    length: float = None
    attributes: Optional[Dict] = None
    start_attributes: Optional[Dict] = None
    end_attributes: Optional[Dict] = None


class EventGraphBuilder:
    @staticmethod
    def add_time_link(
        event_graph,
        later_id: int,
        earlier_id: int,
        length: Optional,
        reversed: bool,
        use_exactly_relation: bool,
    ):
        if not reversed:
            if length is not None:
                if use_exactly_relation:
                    event_graph.after_exactly(later_id, earlier_id, length)
                else:
                    event_graph.after_at_least(later_id, earlier_id, length)
            else:
                event_graph.after_or_at(later_id, earlier_id)
        else:
            if length is not None:
                event_graph.before_at_least(earlier_id, later_id, length)
            else:
                event_graph.before_or_at(earlier_id, later_id)

    @staticmethod
    def add_linked_spans(
        event_graph: EventGraph,
        boundary_start_node_id: Optional[int],
        boundary_end_node_id: Optional[int],
        spans: List[ChainElement],
        follows_graph,
        reversed=False,
        link_last_to_boundary=False,
        use_exactly_relation=False,
    ):
        span_node_id_map = {}
        for i, element in enumerate(spans):
            span_node_ids = {}
            attributes = {"chain_element_id": element.id}

            if element.attributes is not None:
                attributes.update(element.attributes)

            start_attributes = attributes

            if element.start_attributes is not None:
                start_attributes = {**attributes, **element.start_attributes}

            end_attributes = attributes
            if element.end_attributes is not None:
                end_attributes = {**attributes, **element.end_attributes}

            start_node_id = event_graph.add_node(
                event_type=element.start_type, **start_attributes
            )
            span_node_ids["start_node_id"] = start_node_id
            if element.end_type is not None:
                end_node_id = event_graph.add_node(
                    **{**{"event_type": element.end_type}, **end_attributes}
                )
                span_node_ids["end_node_id"] = end_node_id
                length = None
                if hasattr(element, "length"):
                    length = element.length

                if i == len(spans) - 1 and link_last_to_boundary:
                    EventGraphBuilder.add_time_link(
                        event_graph,
                        boundary_end_node_id,
                        start_node_id,
                        length,
                        reversed,
                        use_exactly_relation=use_exactly_relation,
                    )
                else:
                    EventGraphBuilder.add_time_link(
                        event_graph,
                        end_node_id,
                        start_node_id,
                        length,
                        reversed,
                        use_exactly_relation,
                    )

            span_node_id_map[i] = span_node_ids

        for first, second in follows_graph:
            span_objects = []
            for span_index in [first, second]:

                if span_index >= len(spans):
                    curspan = "LAST"
                elif span_index == -1:
                    curspan = "FIRST"
                else:
                    curspan = spans[first]
                span_objects.append(curspan)

            if first == -1:
                first_end_node_id = boundary_start_node_id
            else:
                span_node_ids_first = span_node_id_map[first]
                if "end_node_id" in span_node_ids_first:
                    first_end_node_id = span_node_ids_first["end_node_id"]
                else:
                    first_end_node_id = span_node_ids_first["start_node_id"]

            if second >= len(spans):
                second_start_node_id = boundary_end_node_id
            else:
                span_node_ids_second = span_node_id_map[second]
                second_start_node_id = span_node_ids_second["start_node_id"]

            if second_start_node_id is None:
                raise Exception("second_start_node_id is None")
            if first_end_node_id is None:
                raise Exception("first_end_node_id is None")

            EventGraphBuilder.add_time_link(
                event_graph,
                second_start_node_id,
                first_end_node_id,
                None,
                reversed,
                False,
            )
        return span_node_id_map

    @staticmethod
    def add_chain(
        event_graph: EventGraph,
        boundary_start_node_id,
        boundary_end_node_id,
        chain,
        reversed=False,
        link_last=True,
    ):
        if len(chain) == 0:
            return {}
        # Form a chain [(-1, 0), (0, 1), ..., (n-1, n)] (or without the last entry if
        # not link_last); -1 is boundary_start_node_id, n is boundary_end_node_id
        follows_graph = list(zip(range(-1, len(chain)), range(0, len(chain) + 1)))
        if not link_last:
            follows_graph = follows_graph[:-1]
        return EventGraphBuilder.add_linked_spans(
            event_graph,
            boundary_start_node_id,
            boundary_end_node_id,
            chain,
            follows_graph,
            reversed,
        )

    @staticmethod
    def add_right_aligned_chain(
        event_graph: EventGraph,
        boundary_start_node_id,
        boundary_end_node_id,
        terminal_node_id,
        chain,
        skeleton_type=EventType.SKELETON,
    ):
        if len(chain) == 0:
            return
        if boundary_end_node_id is None:
            raise Exception("boundary_end_node_id must not be None")
        skeleton_chain = EventGraphBuilder.skeletonize_chain(chain, skeleton_type)

        follows_graph = list(zip(range(-1, len(chain)), range(0, len(chain) + 1)))

        follows_graph_reversed = list(zip(range(len(chain)), range(1, len(chain) + 1)))
        retval = {}
        reversed_nodes = EventGraphBuilder.add_linked_spans(
            event_graph,
            boundary_start_node_id,
            boundary_end_node_id,
            chain,
            follows_graph_reversed,
            True,
        )
        for reversed_node in reversed_nodes.values():
            event_graph.after_or_at(
                reversed_node["start_node_id"], boundary_start_node_id
            )
            if "end_node_id" in reversed_node:
                event_graph.after_or_at(
                    reversed_node["end_node_id"], boundary_start_node_id
                )
            if terminal_node_id is not None:
                event_graph.after_or_at(
                    terminal_node_id, reversed_node["start_node_id"]
                )
                if "end_node_id" in reversed_node:
                    event_graph.after_or_at(
                        terminal_node_id, reversed_node["end_node_id"]
                    )

        retval["reversed_nodes"] = reversed_nodes

        forward_nodes = EventGraphBuilder.add_linked_spans(
            event_graph,
            boundary_start_node_id,
            boundary_end_node_id,
            skeleton_chain,
            follows_graph,
            False,
        )
        retval["forward_nodes"] = forward_nodes

        return retval

    @staticmethod
    def find_right_aligned_collector(event_graph: EventGraph, section_name):
        events = event_graph.find_section_events_by_type(
            section_name, EventType.RIGHT_ALIGNED_COLLECTOR
        )
        if len(events) > 0:
            return events[0]["id"]
        return None

    @staticmethod
    def find_or_add_right_aligned_collector(
        event_graph: EventGraph, section_name, section_span, length
    ) -> int:
        right_aligned_collector_id = EventGraphBuilder.find_right_aligned_collector(
            event_graph, section_name
        )
        if right_aligned_collector_id is None:
            right_aligned_collector_id = event_graph.add_node(
                section_name=section_name, event_type=EventType.RIGHT_ALIGNED_COLLECTOR
            )

            event_graph.after_or_at(section_span.end, right_aligned_collector_id)
            if length is not None:
                event_graph.after_exactly(
                    right_aligned_collector_id, section_span.start, length
                )

        return right_aligned_collector_id

    @staticmethod
    def add_right_aligned_collector_for_section(
        event_graph: EventGraph, section_graph: SectionGraph, section_name
    ):
        assert section_graph.section_info(section_name).align == "right"

        cur_section_span = event_graph.find_section_start_end(section_name)
        length = section_graph.section_info(section_name).length
        return EventGraphBuilder.find_or_add_right_aligned_collector(
            event_graph, section_name, cur_section_span, length
        )

    @staticmethod
    def build_section_structure(
        event_graph: EventGraph, section_graph: SectionGraph, start_node_ids
    ):
        section_name_to_span_index: Dict[str, int]
        parent_sections: Dict[str, List[ChainElement]] = {}
        root_sections: List[ChainElement] = []

        # Create a chain of spans for all subsections of a parent section and
        # find root sections (without parent)
        for i, section_name in enumerate(section_graph.topologically_sorted_sections()):
            section_info = section_graph.section_info(section_name)
            length = section_info.length

            attributes = {"section_name": section_name}
            if section_info.trigger_output:
                attributes["trigger_output"] = section_info.trigger_output
            if section_info.handle:
                attributes["handle"] = section_info.handle
            if section_info.state is not None:
                attributes["state"] = section_info.state
            if section_info.local is not None:
                attributes["local"] = section_info.local
            section_span = ChainElement(
                section_info.section_id,
                start_type=EventType.SECTION_START,
                end_type=EventType.SECTION_END,
                length=length,
                attributes=attributes,
            )

            parent_section_name = section_graph.parent(section_name)

            if parent_section_name:
                if parent_section_name not in parent_sections:
                    parent_sections[parent_section_name] = []
                parent_sections[parent_section_name].append(section_span)

            if len(parent_sections) == 0:
                root_sections.append(section_span)

        EventGraphBuilder.add_linked_spans(
            event_graph=event_graph,
            boundary_start_node_id=None,
            boundary_end_node_id=None,
            spans=root_sections,
            follows_graph=[],
            reversed=False,
            use_exactly_relation=True,
        )

        # Add support for right-aligned root sections and let root sections follow after
        # init events
        for section_name in [s.attributes["section_name"] for s in root_sections]:
            if section_graph.section_info(section_name).align == "right":
                EventGraphBuilder.add_right_aligned_collector_for_section(
                    event_graph, section_graph, section_name
                )
            root_span = event_graph.find_section_start_end(section_name)
            for start_node_id in start_node_ids:
                event_graph.after_or_at(root_span.start, start_node_id)

        for parent_section, spans in parent_sections.items():
            parent_span = event_graph.find_section_start_end(parent_section)
            parent_right_aligned = (
                section_graph.section_info(parent_section).align == "right"
            )
            section_name_to_span_index = {
                span.attributes["section_name"]: i for i, span in enumerate(spans)
            }
            follows_graph = set()
            for span in spans:
                section_name = span.attributes["section_name"]
                for follower in section_graph.followers(section_name):
                    follows_graph.add(
                        (
                            section_name_to_span_index[section_name],
                            section_name_to_span_index[follower],
                        )
                    )

            # create unlinked subsections
            real_subsection_references = EventGraphBuilder.add_linked_spans(
                event_graph=event_graph,
                boundary_start_node_id=None,
                boundary_end_node_id=None,
                spans=spans,
                follows_graph=[],
                reversed=False,
                use_exactly_relation=True,
            )

            subsection_spans = copy.deepcopy(spans)
            for span in subsection_spans:
                span.start_type = EventType.SUBSECTION_START
                span.end_type = EventType.SUBSECTION_END
                span.attributes["subsection_name"] = span.attributes["section_name"]
                span.attributes["section_name"] = parent_section

            if parent_right_aligned:

                right_aligned_collector_id = (
                    EventGraphBuilder.add_right_aligned_collector_for_section(
                        event_graph, section_graph, parent_section
                    )
                )

                assert (
                    right_aligned_collector_id is not None
                ), f"No right aligned collector node found for {parent_section}"

                skeleton_chain = EventGraphBuilder.skeletonize_chain(
                    subsection_spans, EventType.SECTION_SKELETON
                )

                subsection_refs_follows_graph = follows_graph.copy()
                for i, _ in enumerate(spans):
                    subsection_refs_follows_graph.add((-1, i))
                    subsection_refs_follows_graph.add((i, len(spans) + 1))

                subsection_references = EventGraphBuilder.add_linked_spans(
                    event_graph=event_graph,
                    boundary_start_node_id=parent_span.start,
                    boundary_end_node_id=right_aligned_collector_id,
                    spans=skeleton_chain,
                    follows_graph=subsection_refs_follows_graph,
                    reversed=False,
                )

                subsection_refs_reversed_follows_graph = follows_graph.copy()
                for i, _ in enumerate(spans):
                    subsection_refs_reversed_follows_graph.add((i, len(spans) + 1))

                subsection_references_reversed = EventGraphBuilder.add_linked_spans(
                    event_graph=event_graph,
                    boundary_start_node_id=parent_span.start,
                    boundary_end_node_id=right_aligned_collector_id,
                    spans=subsection_spans,
                    follows_graph=subsection_refs_reversed_follows_graph,
                    reversed=True,
                )
                for i, subsection_span in subsection_references_reversed.items():
                    event_graph.after_or_at(
                        subsection_span["start_node_id"], parent_span.start
                    )
                for key, real_ref in real_subsection_references.items():
                    event_graph.after_or_at(parent_span.end, real_ref["end_node_id"])

                    skeleton_reference = subsection_references[key]
                    subsection_reference = subsection_references_reversed[key]
                    relative_timing_node_id = event_graph.add_node(
                        section_name=parent_section,
                        event_type=EventType.RELATIVE_TIMING,
                        subsection=spans[key].id,
                    )

                    event_graph.before_or_at(
                        subsection_reference["start_node_id"], relative_timing_node_id
                    )

                    event_graph.relative_before(
                        relative_timing_node_id, subsection_reference["end_node_id"]
                    )

                    event_graph.uses_early_reference(
                        relative_timing_node_id, skeleton_reference["start_node_id"]
                    )
                    event_graph.uses_late_reference(
                        relative_timing_node_id, skeleton_reference["end_node_id"]
                    )

            else:  # parent is left aligned

                for i in range(len(spans)):
                    # start all subsections after the parent
                    follows_graph.add((-1, i))

                    # end parent after all subsections
                    follows_graph.add((i, len(spans) + 1))

                subsection_references = EventGraphBuilder.add_linked_spans(
                    event_graph=event_graph,
                    boundary_start_node_id=parent_span.start,
                    boundary_end_node_id=parent_span.end,
                    spans=subsection_spans,
                    follows_graph=follows_graph,
                    reversed=False,
                    link_last_to_boundary=True,
                )

            if parent_right_aligned:
                subsection_reference_items = subsection_references_reversed.items()
            else:
                subsection_reference_items = subsection_references.items()

            for k, v in subsection_reference_items:
                current_subsection_name = spans[k].attributes["section_name"]

                subsection_span = event_graph.find_section_start_end(
                    current_subsection_name
                )

                subsection_reference_end_node_id = v["end_node_id"]
                subsection_reference_start_node_id = v["start_node_id"]

                event_graph.after_or_at(
                    subsection_span.start, subsection_reference_start_node_id
                )

                if not parent_right_aligned:
                    event_graph.after_or_at(
                        subsection_reference_end_node_id, subsection_span.end
                    )

        for section_name in section_graph.sections():

            section_span = event_graph.find_section_start_end(section_name)

            if section_graph.section_info(section_name).acquisition_types is not None:
                if (
                    "spectroscopy"
                    in section_graph.section_info(section_name).acquisition_types
                ):
                    spectroscopy_end_id = event_graph.add_node(
                        section_name=section_name, event_type=EventType.SPECTROSCOPY_END
                    )
                    event_graph.after_or_at(section_span.end, spectroscopy_end_id)

            if section_graph.section_info(section_name).align == "right":
                EventGraphBuilder.find_or_add_right_aligned_collector(
                    event_graph,
                    section_name,
                    section_span,
                    section_graph.section_info(section_name).length,
                )

    @staticmethod
    def complete_section_structure(
        event_graph: EventGraph, section_graph: SectionGraph
    ):
        for section_name in reversed(
            list(section_graph.topologically_sorted_sections())
        ):
            if section_graph.section_info(section_name).align == "right":
                subsection_start_nodes = event_graph.find_section_events_by_type(
                    section_name, event_type=EventType.SUBSECTION_START
                )
                for subsection_start_node in subsection_start_nodes:
                    subsection_name = subsection_start_node["subsection_name"]
                    skeleton_start_node = next(
                        n
                        for n in event_graph.find_section_events_by_type(
                            section_name,
                            event_type=EventType.SECTION_SKELETON,
                            properties={
                                "subsection_name": subsection_name,
                                "skeleton_of": "SUBSECTION_START",
                            },
                        )
                    )
                    skeleton_end_node = next(
                        n
                        for n in event_graph.find_section_events_by_type(
                            section_name,
                            event_type=EventType.SECTION_SKELETON,
                            properties={
                                "subsection_name": subsection_name,
                                "skeleton_of": "SUBSECTION_END",
                            },
                        )
                    )

                    subsection_span = event_graph.find_section_start_end(
                        subsection_name
                    )
                    subsection_event_ids = event_graph.descendants(
                        subsection_span.end
                    ).difference(event_graph.descendants(subsection_span.start))
                    subsection_event_ids.add(subsection_span.end)

                    subsection_events = [
                        event_graph.node(node_id) for node_id in subsection_event_ids
                    ]
                    event_map = {}
                    for event in subsection_events:
                        new_id = event_graph.add_node(
                            section_name=event["section_name"],
                            event_type=EventType.SKELETON,
                            skeleton_of=event["event_type"],
                        )
                        event_map[event["id"]] = new_id
                        if event["event_type"] == "SECTION_START":
                            event_graph.after_or_at(new_id, skeleton_start_node["id"])
                        if event["event_type"] == "SECTION_END":
                            event_graph.after_or_at(skeleton_end_node["id"], new_id)

                    for event in subsection_events:
                        new_node_id = event_map[event["id"]]

                        for edge in event_graph.out_edges(event["id"]):
                            other_event = event_graph.node(edge[1])
                            new_other_event = None
                            if other_event["id"] in event_map:
                                new_other_event = event_map[other_event["id"]]
                            if new_other_event is not None:
                                relation = edge[2]["relation"]
                                if relation == EventRelation.AFTER_EXACTLY:
                                    relation = EventRelation.AFTER_OR_AT

                                event_graph.add_edge(
                                    new_node_id, new_other_event, relation=relation
                                )
                                for k, v in edge[2].items():
                                    if not k == "relation":
                                        event_graph.set_edge_attribute(
                                            new_node_id, new_other_event, k, v
                                        )

    @staticmethod
    def skeletonize_chain(chain, skeleton_type=EventType.SKELETON):
        skeleton_chain = []
        for element in chain:
            skeleton_element = copy.deepcopy(element)
            skeleton_element.start_type = skeleton_type
            if element.end_type is not None:
                skeleton_element.end_type = skeleton_type
            skeleton_element.start_attributes = {"skeleton_of": element.start_type}
            skeleton_element.end_attributes = {"skeleton_of": element.end_type}
            skeleton_chain.append(skeleton_element)
        return skeleton_chain

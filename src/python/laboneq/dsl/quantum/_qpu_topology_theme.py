# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

ZI_DARK_BLUE = "#172983"
ZI_LIGHT_BLUE = "#009EE0"


def zi_draw_nx_theme(feature: str) -> dict:
    if feature == "nodes":
        formatting_dict = {
            "node_color": ZI_LIGHT_BLUE,
            "alpha": 0.2,
        }
    elif feature == "labels":
        formatting_dict = {
            "font_color": ZI_DARK_BLUE,
        }
    elif feature == "edges":
        formatting_dict = {
            "edge_color": ZI_DARK_BLUE,
            "alpha": 0.2,
        }
    elif feature == "edge_labels":
        formatting_dict = {
            "label_pos": 0.3,
            "font_color": ZI_DARK_BLUE,
            "bbox": {"alpha": 0},
            "font_size": 10,
        }
    else:
        raise ValueError("Feature must be one of [nodes, labels, edges, edge_labels].")

    return formatting_dict

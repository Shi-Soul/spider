# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Iterable

import numpy as np

MeshPart = tuple[np.ndarray, np.ndarray]


def flatten_base(
    hulls: Iterable[MeshPart],
    thickness: float = 0.01,
    R_world_local: np.ndarray | None = None,
    obj_world_pos: np.ndarray | None = None,
    floor_z: float = 0.0,
    pad: float = 0.05,
    well_below_offset: float = 0.0,
) -> list[MeshPart]:
    """Append a thin plate that supports the object resting on a world-frame floor.

    If ``R_world_local`` and ``obj_world_pos`` are given, the plate is added in
    the object's *local* frame so that, when the free joint is set to
    (obj_world_pos, R_world_local), the plate's TOP sits at ``floor_z`` (minus
    ``well_below_offset``) and the plate extends ``thickness`` downward from
    there. With ``well_below_offset=0.0`` (the default) the plate's top is
    flush with the object's lowest world-frame vertex.

    Otherwise (the legacy behavior), the plate is placed at the convex hull's
    local-frame ``min_z``.

    The plate's XY extent is the object's world-frame bbox padded by ``pad``
    so it is large enough to support the object even if it tilts slightly.
    """
    hull_list = list(hulls)
    if not hull_list:
        return hull_list

    all_vertices = np.vstack([vertices for vertices, _ in hull_list])

    if R_world_local is not None and obj_world_pos is not None:
        v_world = (R_world_local @ all_vertices.T).T + obj_world_pos
        wx_min, wx_max = v_world[:, 0].min() - pad, v_world[:, 0].max() + pad
        wy_min, wy_max = v_world[:, 1].min() - pad, v_world[:, 1].max() + pad
        z_top = floor_z - well_below_offset
        z_bot = z_top - thickness
        corners_world_top = np.array(
            [
                [wx_min, wy_min, z_top],
                [wx_max, wy_min, z_top],
                [wx_max, wy_max, z_top],
                [wx_min, wy_max, z_top],
            ]
        )
        corners_world_bot = corners_world_top.copy()
        corners_world_bot[:, 2] = z_bot
        corners_world = np.vstack([corners_world_bot, corners_world_top])
        plate_vertices = (R_world_local.T @ (corners_world - obj_world_pos).T).T
    else:
        min_x, max_x = np.min(all_vertices[:, 0]), np.max(all_vertices[:, 0])
        min_y, max_y = np.min(all_vertices[:, 1]), np.max(all_vertices[:, 1])
        min_z = np.min(all_vertices[:, 2])
        z0 = min_z
        z1 = min_z + thickness
        plate_vertices = np.array(
            [
                [min_x, min_y, z0],
                [max_x, min_y, z0],
                [max_x, max_y, z0],
                [min_x, max_y, z0],
                [min_x, min_y, z1],
                [max_x, min_y, z1],
                [max_x, max_y, z1],
                [min_x, max_y, z1],
            ]
        )

    plate_faces = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ],
        dtype=int,
    )

    hull_list.append((plate_vertices, plate_faces))
    return hull_list

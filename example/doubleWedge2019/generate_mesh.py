#!/usr/bin/env python3
"""Generate a 3-block parabola-top mesh for the double-wedge case."""

from __future__ import annotations

import argparse
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_DIR = Path(__file__).resolve().parents[2]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from tools.read_grid import read_grid

CASE_DIR = Path(__file__).resolve().parent
WORK_DIR = CASE_DIR / "input"
PLOT3D_NAME = "doubleWedge2019.xyz"
BOUNDARY_NAME = "doubleWedge2019.inp"
SUMMARY_NAME = "mesh_summary.txt"

BC_WALL = 2
BC_INFLOW = 5
BC_OUTFLOW = 6
BC_INTERFACE = -1

POINT_O = np.array([0.0, 0.0], dtype=float)
POINT_1 = np.array([0.0440, 0.0254], dtype=float)
POINT_2 = np.array([0.0586, 0.0462], dtype=float)
POINT_3 = np.array([0.0776, 0.0462], dtype=float)

# The upper boundary is a right-opening parabola with the prescribed left
# vertex and right point.
PARABOLA_VERTEX = np.array([-0.0100, 0.0000], dtype=float)
PARABOLA_RIGHT_POINT = np.array([0.0700, 0.0700], dtype=float)

TARGET_WALL_TANGENT = 1.0e-5
FIRST_INTERFACE_TANGENT_FACTOR = 1.0
RIGHT_TANGENT_FACTOR = 2.0
FIRST_WALL_HEIGHT = 2.93e-6
N_WALL_NORMAL = 506
INTERFACE_RELAX_CELLS = 96

FREESTREAM_CASES = {
    "low_enthalpy": {
        "Mach": 7.11,
        "T": 191.0,
        "p": 391.0,
        "U": 1972.0,
        "rho": 0.007,
        "Re_unit": 1.1e6,
        "test_time": 3.27e-4,
    },
    "high_enthalpy": {
        "Mach": 7.14,
        "T": 710.0,
        "p": 780.0,
        "U": 3812.0,
        "rho": 0.004,
        "Re_unit": 4.4e5,
        "test_time": 2.42e-4,
    },
}


@dataclass
class BoundaryEntry:
    source: tuple[int, int, int, int]
    label: int
    target: tuple[int, int, int, int] | None = None
    target_block: int | None = None


@dataclass
class Block:
    name: str
    x: np.ndarray
    y: np.ndarray
    entries: list[BoundaryEntry]

    @property
    def ni(self) -> int:
        return int(self.x.shape[0])

    @property
    def nj(self) -> int:
        return int(self.x.shape[1])


def parabola_coefficient(vertex: np.ndarray = PARABOLA_VERTEX, point: np.ndarray = PARABOLA_RIGHT_POINT) -> float:
    dy = point[1] - vertex[1]
    if abs(dy) < 1e-14:
        raise ValueError("Parabola point and vertex must not have the same y coordinate.")
    return float((point[0] - vertex[0]) / (dy * dy))


PARABOLA_A = parabola_coefficient()


def parabola_x(y: np.ndarray | float) -> np.ndarray | float:
    return PARABOLA_VERTEX[0] + PARABOLA_A * (y - PARABOLA_VERTEX[1]) ** 2


def parabola_dxdy(y: np.ndarray | float) -> np.ndarray | float:
    return 2.0 * PARABOLA_A * (y - PARABOLA_VERTEX[1])


def parabola_point(y: float) -> np.ndarray:
    return np.array([float(parabola_x(y)), float(y)], dtype=float)


def parabola_tangent(y: float) -> np.ndarray:
    tangent = np.array([float(parabola_dxdy(y)), 1.0], dtype=float)
    return tangent / np.linalg.norm(tangent)


def foot_on_parabola(point: np.ndarray) -> np.ndarray:
    """Closest point on the upper parabola branch by bounded golden search."""
    lo = float(PARABOLA_VERTEX[1])
    hi = float(PARABOLA_RIGHT_POINT[1])
    gr = (math.sqrt(5.0) - 1.0) * 0.5

    def distance_sq(y: float) -> float:
        q = parabola_point(y)
        d = q - point
        return float(np.dot(d, d))

    c = hi - gr * (hi - lo)
    d = lo + gr * (hi - lo)
    fc = distance_sq(c)
    fd = distance_sq(d)
    for _ in range(160):
        if fc > fd:
            lo = c
            c = d
            fc = fd
            d = lo + gr * (hi - lo)
            fd = distance_sq(d)
        else:
            hi = d
            d = c
            fd = fc
            c = hi - gr * (hi - lo)
            fc = distance_sq(c)
    return parabola_point(0.5 * (lo + hi))


def arc_length_between(y0: float, y1: float, n_sample: int = 4097) -> float:
    ys = np.linspace(y0, y1, n_sample, dtype=float)
    integrand = np.sqrt(1.0 + parabola_dxdy(ys) ** 2)
    return float(np.trapezoid(integrand, ys))


def parabola_segment_by_arclength(y0: float, y1: float, n_cell: int) -> np.ndarray:
    sample_count = max(10001, 8 * n_cell + 1)
    ys = np.linspace(y0, y1, sample_count, dtype=float)
    integrand = np.sqrt(1.0 + parabola_dxdy(ys) ** 2)
    ds = 0.5 * (integrand[1:] + integrand[:-1]) * np.diff(ys)
    s = np.empty(sample_count, dtype=float)
    s[0] = 0.0
    s[1:] = np.cumsum(ds)
    targets = np.linspace(0.0, s[-1], n_cell + 1, dtype=float)
    y = np.interp(targets, s, ys)
    x = parabola_x(y)
    pts = np.column_stack((x, y))
    pts[0] = parabola_point(y0)
    pts[-1] = parabola_point(y1)
    return pts


def line_segment(start: np.ndarray, end: np.ndarray, n_cell: int) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n_cell + 1, dtype=float)[:, None]
    pts = (1.0 - t) * start[None, :] + t * end[None, :]
    pts[0] = start
    pts[-1] = end
    return pts


def smooth_widths_with_endpoints(length: float, n_cell: int, start_width: float, end_width: float) -> np.ndarray:
    """Smooth cell widths with fixed first/last widths and exact total length."""
    if n_cell <= 0:
        raise ValueError("n_cell must be positive")
    if length <= 0.0:
        raise ValueError("length must be positive")
    if start_width <= 0.0 or end_width <= 0.0:
        raise ValueError("endpoint widths must be positive")
    if n_cell == 1:
        return np.array([length], dtype=float)

    idx = np.arange(n_cell, dtype=float)
    linear = start_width + (end_width - start_width) * idx / (n_cell - 1)
    bubble = np.sin(np.pi * idx / (n_cell - 1))
    bubble_sum = float(bubble.sum())
    correction = (length - float(linear.sum())) / bubble_sum
    widths = linear + correction * bubble

    if np.any(widths <= 0.0):
        # Fall back to a positive affine rescaling around the exact endpoints.
        interior_count = n_cell - 2
        if interior_count <= 0:
            widths = np.array([start_width, end_width], dtype=float)
            widths *= length / widths.sum()
            return widths
        remaining = length - start_width - end_width
        if remaining <= 0.0:
            raise ValueError("endpoint widths are too large for this segment")
        interior = np.full(interior_count, remaining / interior_count, dtype=float)
        widths = np.concatenate(([start_width], interior, [end_width]))

    widths *= length / widths.sum()
    widths[0] = start_width
    widths[-1] = end_width
    if n_cell > 2:
        residual = length - float(widths.sum())
        widths[1:-1] += residual / (n_cell - 2)
    else:
        widths *= length / widths.sum()
    if np.any(widths <= 0.0):
        raise ValueError("failed to build positive smooth widths")
    return widths


def line_segment_with_widths(start: np.ndarray, end: np.ndarray, widths: np.ndarray) -> np.ndarray:
    direction = end - start
    length = float(np.linalg.norm(direction))
    if length <= 0.0:
        raise ValueError("line segment length must be positive")
    direction /= length
    s = np.empty(widths.size + 1, dtype=float)
    s[0] = 0.0
    s[1:] = np.cumsum(widths)
    s[-1] = length
    pts = start[None, :] + s[:, None] * direction[None, :]
    pts[0] = start
    pts[-1] = end
    return pts


def parabola_segment_with_widths(y0: float, y1: float, widths: np.ndarray) -> np.ndarray:
    sample_count = max(10001, 8 * widths.size + 1)
    ys = np.linspace(y0, y1, sample_count, dtype=float)
    integrand = np.sqrt(1.0 + parabola_dxdy(ys) ** 2)
    ds = 0.5 * (integrand[1:] + integrand[:-1]) * np.diff(ys)
    s = np.empty(sample_count, dtype=float)
    s[0] = 0.0
    s[1:] = np.cumsum(ds)
    targets = np.empty(widths.size + 1, dtype=float)
    targets[0] = 0.0
    targets[1:] = np.cumsum(widths)
    targets[-1] = s[-1]
    y = np.interp(targets, s, ys)
    pts = np.column_stack((parabola_x(y), y))
    pts[0] = parabola_point(y0)
    pts[-1] = parabola_point(y1)
    return pts


def geometric_widths(length: float, n_cell: int, first_width: float) -> np.ndarray:
    if n_cell <= 0:
        raise ValueError("n_cell must be positive")
    if first_width <= 0.0:
        raise ValueError("first_width must be positive")
    if length <= 0.0:
        raise ValueError("length must be positive")
    if first_width * n_cell >= 0.985 * length:
        return np.full(n_cell, length / n_cell, dtype=float)

    def total(ratio: float) -> float:
        if abs(ratio - 1.0) < 1e-12:
            return first_width * n_cell
        return first_width * (ratio**n_cell - 1.0) / (ratio - 1.0)

    lo = 1.0
    hi = 1.01
    while total(hi) < length:
        hi = 1.0 + 2.0 * (hi - 1.0)
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if total(mid) < length:
            lo = mid
        else:
            hi = mid
    ratio = 0.5 * (lo + hi)
    widths = first_width * ratio ** np.arange(n_cell, dtype=float)
    widths *= length / widths.sum()
    return widths


def eta_for_column(length: float, n_cell: int, first_width: float) -> np.ndarray:
    widths = geometric_widths(length, n_cell, min(first_width, 0.25 * length))
    eta = np.empty(n_cell + 1, dtype=float)
    eta[0] = 0.0
    eta[1:] = np.cumsum(widths) / length
    eta[-1] = 1.0
    return eta


def build_patch(bottom: np.ndarray, top: np.ndarray, n_normal: int, first_height: float) -> tuple[np.ndarray, np.ndarray]:
    if bottom.shape != top.shape:
        raise ValueError("bottom and top must have matching node counts")
    ni = bottom.shape[0]
    x = np.empty((ni, n_normal + 1), dtype=float)
    y = np.empty((ni, n_normal + 1), dtype=float)
    for i in range(ni):
        vec = top[i] - bottom[i]
        length = float(np.linalg.norm(vec))
        eta = eta_for_column(length, n_normal, first_height)
        pts = bottom[i][None, :] + eta[:, None] * vec[None, :]
        x[i, :] = pts[:, 0]
        y[i, :] = pts[:, 1]
    return x, y


def separator_segment(start: np.ndarray, end: np.ndarray, n_normal: int, first_height: float) -> np.ndarray:
    length = float(np.linalg.norm(end - start))
    widths = geometric_widths(length, n_normal, min(first_height, 0.25 * length))
    return line_segment_with_widths(start, end, widths)


def build_row_patch(
    left_separator: np.ndarray,
    right_separator: np.ndarray,
    top_y0: float,
    top_y1: float,
    n_tangent: int,
    left_width: float,
    right_width: float,
) -> tuple[np.ndarray, np.ndarray]:
    if left_separator.shape != right_separator.shape:
        raise ValueError("left and right separator shapes must match")

    n_normal = left_separator.shape[0] - 1
    x = np.empty((n_tangent + 1, n_normal + 1), dtype=float)
    y = np.empty((n_tangent + 1, n_normal + 1), dtype=float)
    for j in range(n_normal + 1):
        if j == n_normal:
            length = arc_length_between(top_y0, top_y1)
            widths = smooth_widths_with_endpoints(length, n_tangent, left_width, right_width)
            row = parabola_segment_with_widths(top_y0, top_y1, widths)
        else:
            start = left_separator[j]
            end = right_separator[j]
            length = float(np.linalg.norm(end - start))
            widths = smooth_widths_with_endpoints(length, n_tangent, left_width, right_width)
            row = line_segment_with_widths(start, end, widths)
        x[:, j] = row[:, 0]
        y[:, j] = row[:, 1]
    return x, y


def relax_interface_columns(
    left_x: np.ndarray,
    left_y: np.ndarray,
    right_x: np.ndarray,
    right_y: np.ndarray,
    relax_cells: int,
) -> None:
    """Make the first cell width on both sides of an interface identical.

    The physical bottom/top boundaries are left untouched.  Only interior rows
    are adjusted over a short i-direction band, with the far edge of the band
    held fixed so the change blends back into the original block interior.
    """
    n_left = left_x.shape[0]
    n_right = right_x.shape[0]
    n_row = left_x.shape[1]
    k = min(relax_cells, n_left - 2, n_right - 2)
    if k < 3:
        return

    left_orig_x = left_x.copy()
    left_orig_y = left_y.copy()
    right_orig_x = right_x.copy()
    right_orig_y = right_y.copy()
    left_anchor = n_left - 1 - k

    for j in range(1, n_row - 1):
        interface = np.array([left_orig_x[-1, j], left_orig_y[-1, j]], dtype=float)
        left_anchor_point = np.array([left_orig_x[left_anchor, j], left_orig_y[left_anchor, j]], dtype=float)
        right_anchor_point = np.array([right_orig_x[k, j], right_orig_y[k, j]], dtype=float)

        left_length = float(np.linalg.norm(interface - left_anchor_point))
        right_length = float(np.linalg.norm(right_anchor_point - interface))
        if left_length <= 0.0 or right_length <= 0.0:
            continue

        left_first = float(np.hypot(left_orig_x[-1, j] - left_orig_x[-2, j], left_orig_y[-1, j] - left_orig_y[-2, j]))
        right_first = float(np.hypot(right_orig_x[1, j] - right_orig_x[0, j], right_orig_y[1, j] - right_orig_y[0, j]))
        shared_first = 0.5 * (left_first + right_first)

        left_far = float(
            np.hypot(
                left_orig_x[left_anchor + 1, j] - left_orig_x[left_anchor, j],
                left_orig_y[left_anchor + 1, j] - left_orig_y[left_anchor, j],
            )
        )
        right_far = float(np.hypot(right_orig_x[k, j] - right_orig_x[k - 1, j], right_orig_y[k, j] - right_orig_y[k - 1, j]))

        left_widths = smooth_widths_with_endpoints(left_length, k, left_far, shared_first)
        right_widths = smooth_widths_with_endpoints(right_length, k, shared_first, right_far)

        left_direction = (interface - left_anchor_point) / left_length
        right_direction = (right_anchor_point - interface) / right_length

        left_s = np.empty(k + 1, dtype=float)
        left_s[0] = 0.0
        left_s[1:] = np.cumsum(left_widths)
        left_s[-1] = left_length
        left_points = left_anchor_point[None, :] + left_s[:, None] * left_direction[None, :]

        right_s = np.empty(k + 1, dtype=float)
        right_s[0] = 0.0
        right_s[1:] = np.cumsum(right_widths)
        right_s[-1] = right_length
        right_points = interface[None, :] + right_s[:, None] * right_direction[None, :]

        left_x[left_anchor:, j] = left_points[:, 0]
        left_y[left_anchor:, j] = left_points[:, 1]
        right_x[: k + 1, j] = right_points[:, 0]
        right_y[: k + 1, j] = right_points[:, 1]

    # Preserve exact shared interface coordinates after all adjustments.
    right_x[0, :] = left_x[-1, :]
    right_y[0, :] = left_y[-1, :]


def face_range(ni: int, nj: int, face: str) -> tuple[int, int, int, int]:
    if face == "imin":
        return (1, 1, 1, nj)
    if face == "imax":
        return (ni, ni, 1, nj)
    if face == "jmin":
        return (1, ni, 1, 1)
    if face == "jmax":
        return (1, ni, nj, nj)
    raise ValueError(face)


def build_blocks(
    target_tangent: float,
    first_interface_tangent_factor: float,
    right_tangent_factor: float,
    n_normal: int,
    first_height: float,
    interface_relax_cells: int,
) -> tuple[list[Block], dict[str, np.ndarray]]:
    q1 = foot_on_parabola(POINT_1)
    q2 = foot_on_parabola(POINT_2)
    if not (PARABOLA_VERTEX[1] < q1[1] < q2[1] < PARABOLA_RIGHT_POINT[1]):
        raise RuntimeError("Normal feet from points 1 and 2 do not split the parabola in order.")
    if right_tangent_factor < 1.0:
        raise ValueError("right_tangent_factor should be >= 1.0")
    if not (1.0 <= first_interface_tangent_factor <= right_tangent_factor):
        raise ValueError("first_interface_tangent_factor should be between 1.0 and right_tangent_factor")

    wall_points = [POINT_O, POINT_1, POINT_2, POINT_3]
    top_points = [PARABOLA_VERTEX, q1, q2, PARABOLA_RIGHT_POINT]
    vertex_tangent = np.array(
        [
            target_tangent,
            target_tangent * first_interface_tangent_factor,
            target_tangent * right_tangent_factor,
            target_tangent * right_tangent_factor,
        ],
        dtype=float,
    )
    n_cells: list[int] = []
    wall_lengths: list[float] = []
    top_lengths: list[float] = []
    for i in range(3):
        wall_length = float(np.linalg.norm(wall_points[i + 1] - wall_points[i]))
        top_length = arc_length_between(float(top_points[i][1]), float(top_points[i + 1][1]))
        wall_lengths.append(wall_length)
        top_lengths.append(top_length)
        segment_tangent = 0.5 * (vertex_tangent[i] + vertex_tangent[i + 1])
        n_cells.append(max(4, int(math.ceil(max(wall_length, top_length) / segment_tangent))))

    bottom_widths = [
        smooth_widths_with_endpoints(wall_lengths[i], n_cells[i], vertex_tangent[i], vertex_tangent[i + 1]) for i in range(3)
    ]
    top_widths = [
        smooth_widths_with_endpoints(top_lengths[i], n_cells[i], vertex_tangent[i], vertex_tangent[i + 1]) for i in range(3)
    ]

    bottoms = [
        line_segment_with_widths(POINT_O, POINT_1, bottom_widths[0]),
        line_segment_with_widths(POINT_1, POINT_2, bottom_widths[1]),
        line_segment_with_widths(POINT_2, POINT_3, bottom_widths[2]),
    ]
    tops = [
        parabola_segment_with_widths(PARABOLA_VERTEX[1], q1[1], top_widths[0]),
        parabola_segment_with_widths(q1[1], q2[1], top_widths[1]),
        parabola_segment_with_widths(q2[1], PARABOLA_RIGHT_POINT[1], top_widths[2]),
    ]

    block_arrays = [build_patch(bottoms[i], tops[i], n_normal, first_height) for i in range(3)]
    b1_x, b1_y = block_arrays[0]
    b2_x, b2_y = block_arrays[1]
    b3_x, b3_y = block_arrays[2]

    # Force shared interface coordinates to be bitwise identical, then adjust
    # a short band on both sides so the first i-cell width is matched for every
    # j row while the change blends back into the block interiors.
    b2_x[0, :] = b1_x[-1, :]
    b2_y[0, :] = b1_y[-1, :]
    b3_x[0, :] = b2_x[-1, :]
    b3_y[0, :] = b2_y[-1, :]
    relax_interface_columns(b1_x, b1_y, b2_x, b2_y, interface_relax_cells)
    relax_interface_columns(b2_x, b2_y, b3_x, b3_y, interface_relax_cells)

    blocks = [
        Block(
            "B1_O_to_1",
            b1_x,
            b1_y,
            [
                BoundaryEntry(face_range(b1_x.shape[0], b1_x.shape[1], "imin"), BC_INFLOW),
                BoundaryEntry(face_range(b1_x.shape[0], b1_x.shape[1], "imax"), BC_INTERFACE, face_range(b2_x.shape[0], b2_x.shape[1], "imin"), 2),
                BoundaryEntry(face_range(b1_x.shape[0], b1_x.shape[1], "jmin"), BC_WALL),
                BoundaryEntry(face_range(b1_x.shape[0], b1_x.shape[1], "jmax"), BC_INFLOW),
            ],
        ),
        Block(
            "B2_1_to_2",
            b2_x,
            b2_y,
            [
                BoundaryEntry(face_range(b2_x.shape[0], b2_x.shape[1], "imin"), BC_INTERFACE, face_range(b1_x.shape[0], b1_x.shape[1], "imax"), 1),
                BoundaryEntry(face_range(b2_x.shape[0], b2_x.shape[1], "imax"), BC_INTERFACE, face_range(b3_x.shape[0], b3_x.shape[1], "imin"), 3),
                BoundaryEntry(face_range(b2_x.shape[0], b2_x.shape[1], "jmin"), BC_WALL),
                BoundaryEntry(face_range(b2_x.shape[0], b2_x.shape[1], "jmax"), BC_INFLOW),
            ],
        ),
        Block(
            "B3_2_to_3",
            b3_x,
            b3_y,
            [
                BoundaryEntry(face_range(b3_x.shape[0], b3_x.shape[1], "imin"), BC_INTERFACE, face_range(b2_x.shape[0], b2_x.shape[1], "imax"), 2),
                BoundaryEntry(face_range(b3_x.shape[0], b3_x.shape[1], "imax"), BC_OUTFLOW),
                BoundaryEntry(face_range(b3_x.shape[0], b3_x.shape[1], "jmin"), BC_WALL),
                BoundaryEntry(face_range(b3_x.shape[0], b3_x.shape[1], "jmax"), BC_INFLOW),
            ],
        ),
    ]

    geometry = {
        "Q1": q1,
        "Q2": q2,
        "n_cells": np.array(n_cells, dtype=int),
        "wall_lengths": np.array(wall_lengths, dtype=float),
        "top_lengths": np.array(top_lengths, dtype=float),
        "vertex_tangent": vertex_tangent,
        "bottom_width_minmax": np.array([(width.min(), width.max()) for width in bottom_widths], dtype=float),
        "top_width_minmax": np.array([(width.min(), width.max()) for width in top_widths], dtype=float),
        "interface_relax_cells": np.array([interface_relax_cells], dtype=int),
        "wall_points": np.array(wall_points, dtype=float),
        "top_points": np.array(top_points, dtype=float),
    }
    return blocks, geometry


def write_plot3d(path: Path, blocks: list[Block]) -> None:
    with path.open("wb") as handle:
        np.array([len(blocks)], dtype=np.int32).tofile(handle)
        np.array([(b.ni, b.nj, 1) for b in blocks], dtype=np.int32).tofile(handle)
        for block in blocks:
            z = np.zeros_like(block.x)
            block.x[:, :, None].ravel(order="F").tofile(handle)
            block.y[:, :, None].ravel(order="F").tofile(handle)
            z[:, :, None].ravel(order="F").tofile(handle)


def write_boundary(path: Path, blocks: list[Block]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# double-wedge 3-block parabola-top boundary file\n")
        handle.write("# generated by generate_mesh.py\n")
        for idx, block in enumerate(blocks, start=1):
            handle.write(f"Block {idx}\n")
            handle.write("Boundary conditions\n")
            handle.write(f"{len(block.entries)}\n")
            for entry in block.entries:
                handle.write(" ".join(f"{value:6d}" for value in (*entry.source, entry.label)) + "\n")
                if entry.target is not None and entry.target_block is not None:
                    handle.write(" ".join(f"{value:6d}" for value in (*entry.target, entry.target_block)) + "\n")


def face_nodes(block: Block, face: str) -> np.ndarray:
    if face == "imin":
        return np.column_stack((block.x[0, :], block.y[0, :]))
    if face == "imax":
        return np.column_stack((block.x[-1, :], block.y[-1, :]))
    if face == "jmin":
        return np.column_stack((block.x[:, 0], block.y[:, 0]))
    if face == "jmax":
        return np.column_stack((block.x[:, -1], block.y[:, -1]))
    raise ValueError(face)


def face_tangent_width(block: Block, face: str) -> np.ndarray:
    nodes = face_nodes(block, face)
    return np.linalg.norm(np.diff(nodes, axis=0), axis=1)


def interface_stats(blocks: list[Block]) -> list[dict[str, float | str | int]]:
    pairs = [(0, "imax", 1, "imin"), (1, "imax", 2, "imin")]
    stats: list[dict[str, float | str | int]] = []
    for left_id, left_face, right_id, right_face in pairs:
        left = blocks[left_id]
        right = blocks[right_id]
        left_nodes = face_nodes(left, left_face)
        right_nodes = face_nodes(right, right_face)
        mismatch = float(np.linalg.norm(left_nodes - right_nodes, axis=1).max())

        left_width = np.linalg.norm(left_nodes - np.column_stack((left.x[-2, :], left.y[-2, :])), axis=1)
        right_width = np.linalg.norm(np.column_stack((right.x[1, :], right.y[1, :])) - right_nodes, axis=1)
        ratio = np.maximum(left_width / right_width, right_width / left_width)

        tangent_left = np.linalg.norm(np.diff(left_nodes, axis=0), axis=1)
        tangent_right = np.linalg.norm(np.diff(right_nodes, axis=0), axis=1)
        stats.append(
            {
                "left": left.name,
                "right": right.name,
                "node_mismatch": mismatch,
                "normal_ratio_max": float(ratio.max()),
                "normal_ratio_p95": float(np.percentile(ratio, 95.0)),
                "normal_ratio_mean": float(ratio.mean()),
                "tangent_mismatch": float(np.abs(tangent_left - tangent_right).max()),
            }
        )
    return stats


def signed_cell_areas(block: Block) -> np.ndarray:
    x00 = block.x[:-1, :-1]
    y00 = block.y[:-1, :-1]
    x10 = block.x[1:, :-1]
    y10 = block.y[1:, :-1]
    x11 = block.x[1:, 1:]
    y11 = block.y[1:, 1:]
    x01 = block.x[:-1, 1:]
    y01 = block.y[:-1, 1:]
    return 0.5 * (
        x00 * y10
        - y00 * x10
        + x10 * y11
        - y10 * x11
        + x11 * y01
        - y11 * x01
        + x01 * y00
        - y01 * x00
    )


def wall_normal_vectors() -> list[np.ndarray]:
    walls = [(POINT_O, POINT_1), (POINT_1, POINT_2), (POINT_2, POINT_3)]
    normals: list[np.ndarray] = []
    for start, end in walls:
        tangent = end - start
        tangent /= np.linalg.norm(tangent)
        normals.append(np.array([-tangent[1], tangent[0]], dtype=float))
    return normals


def wall_spacing_stats(blocks: list[Block]) -> dict[str, float]:
    tangent_widths: list[np.ndarray] = []
    top_tangent_widths: list[np.ndarray] = []
    first_widths: list[np.ndarray] = []
    normal_projection: list[np.ndarray] = []
    normals = wall_normal_vectors()
    for block_id, block in enumerate(blocks):
        wall = face_nodes(block, "jmin")
        top = face_nodes(block, "jmax")
        first = np.column_stack((block.x[:, 1], block.y[:, 1]))
        delta = first - wall
        tangent_widths.append(np.linalg.norm(np.diff(wall, axis=0), axis=1))
        top_tangent_widths.append(np.linalg.norm(np.diff(top, axis=0), axis=1))
        first_widths.append(np.linalg.norm(delta, axis=1))
        normal_projection.append(delta @ normals[block_id])
    tang = np.concatenate(tangent_widths)
    top_tang = np.concatenate(top_tangent_widths)
    first = np.concatenate(first_widths)
    proj = np.concatenate(normal_projection)

    def max_adjacent_ratio(width_sets: list[np.ndarray]) -> float:
        ratios: list[np.ndarray] = []
        for widths in width_sets:
            local = widths[1:] / widths[:-1]
            ratios.append(np.maximum(local, 1.0 / local))
        for left, right in zip(width_sets[:-1], width_sets[1:]):
            edge = np.array([right[0] / left[-1]], dtype=float)
            ratios.append(np.maximum(edge, 1.0 / edge))
        return float(np.concatenate(ratios).max())

    return {
        "wall_tangent_min": float(tang.min()),
        "wall_tangent_max": float(tang.max()),
        "wall_tangent_ratio": float(tang.max() / tang.min()),
        "wall_tangent_adjacent_ratio_max": max_adjacent_ratio(tangent_widths),
        "top_tangent_min": float(top_tang.min()),
        "top_tangent_max": float(top_tang.max()),
        "top_tangent_ratio": float(top_tang.max() / top_tang.min()),
        "top_tangent_adjacent_ratio_max": max_adjacent_ratio(top_tangent_widths),
        "first_height_min": float(first.min()),
        "first_height_max": float(first.max()),
        "first_projected_min": float(proj.min()),
        "first_projected_max": float(proj.max()),
    }


def wall_s_locations(blocks: list[Block]) -> tuple[np.ndarray, np.ndarray]:
    s_centers: list[np.ndarray] = []
    ds_values: list[np.ndarray] = []
    s0 = 0.0
    for block in blocks:
        wall = face_nodes(block, "jmin")
        ds = np.linalg.norm(np.diff(wall, axis=0), axis=1)
        centers = s0 + np.cumsum(ds) - 0.5 * ds
        s_centers.append(centers)
        ds_values.append(ds)
        s0 += float(ds.sum())
    return np.concatenate(s_centers), np.concatenate(ds_values)


def yplus_stats(blocks: list[Block]) -> dict[str, dict[str, float]]:
    s_centers, ds_values = wall_s_locations(blocks)
    wall_stats = wall_spacing_stats(blocks)
    y_center = 0.5 * wall_stats["first_projected_min"]
    out: dict[str, dict[str, float]] = {}
    for name, case in FREESTREAM_CASES.items():
        rho = case["rho"]
        u = case["U"]
        mu = rho * u / case["Re_unit"]
        re_x = np.maximum(case["Re_unit"] * s_centers, 1.0)
        cf = 0.664 / np.sqrt(re_x)
        u_tau = u * np.sqrt(0.5 * cf)
        yplus = rho * u_tau * y_center / mu
        dsplus = rho * u_tau * ds_values / mu
        out[name] = {
            "mu": float(mu),
            "max_yplus": float(yplus.max()),
            "min_yplus": float(yplus.min()),
            "max_wall_parallel_plus": float(dsplus.max()),
            "min_wall_parallel_plus": float(dsplus.min()),
        }
    return out


def normal_growth_stats(blocks: list[Block]) -> dict[str, float]:
    ratios: list[np.ndarray] = []
    for block in blocks:
        dx = np.diff(block.x, axis=1)
        dy = np.diff(block.y, axis=1)
        widths = np.hypot(dx, dy)
        local = widths[:, 1:] / widths[:, :-1]
        ratios.append(local)
    ratio = np.concatenate([r.ravel() for r in ratios])
    return {
        "normal_growth_min": float(ratio.min()),
        "normal_growth_max": float(ratio.max()),
    }


def block_tangent_smoothness(blocks: list[Block]) -> list[dict[str, float | str]]:
    stats: list[dict[str, float | str]] = []
    for block in blocks:
        widths = np.hypot(np.diff(block.x, axis=0), np.diff(block.y, axis=0))
        local = widths[1:, :] / widths[:-1, :]
        ratio = np.maximum(local, 1.0 / local)
        stats.append(
            {
                "name": block.name,
                "min_width": float(widths.min()),
                "max_width": float(widths.max()),
                "max_adjacent_ratio": float(ratio.max()),
                "p99_adjacent_ratio": float(np.percentile(ratio, 99.0)),
            }
        )
    return stats


def write_summary(
    path: Path,
    blocks: list[Block],
    geometry: dict[str, np.ndarray],
    target_tangent: float,
    first_interface_tangent_factor: float,
    right_tangent_factor: float,
    first_height: float,
    n_normal: int,
) -> None:
    q1 = geometry["Q1"]
    q2 = geometry["Q2"]
    n_cells = geometry["n_cells"]
    total_cells = sum((b.ni - 1) * (b.nj - 1) for b in blocks)
    total_nodes = sum(b.ni * b.nj for b in blocks)
    wall_stats = wall_spacing_stats(blocks)
    growth = normal_growth_stats(blocks)
    tangent_smoothness = block_tangent_smoothness(blocks)
    interfaces = interface_stats(blocks)
    min_area = min(float(signed_cell_areas(block).min()) for block in blocks)
    max_area = max(float(signed_cell_areas(block).max()) for block in blocks)
    yplus = yplus_stats(blocks)
    normal_residual_1 = float(abs(np.dot(POINT_1 - q1, parabola_tangent(float(q1[1])))))
    normal_residual_2 = float(abs(np.dot(POINT_2 - q2, parabola_tangent(float(q2[1])))))

    with path.open("w", encoding="utf-8") as handle:
        handle.write("Double-wedge 2019 parabola-top 3-block mesh summary\n\n")
        handle.write("Geometry points [m]:\n")
        for name, point in (
            ("O", POINT_O),
            ("1", POINT_1),
            ("2", POINT_2),
            ("3", POINT_3),
            ("V", PARABOLA_VERTEX),
            ("P", PARABOLA_RIGHT_POINT),
            ("Q1", q1),
            ("Q2", q2),
        ):
            handle.write(f"  {name}: x={point[0]:.12e}, y={point[1]:.12e}\n")
        handle.write(f"  parabola: x = {PARABOLA_VERTEX[0]:.12e} + {PARABOLA_A:.12e} * (y - {PARABOLA_VERTEX[1]:.12e})^2\n")
        handle.write(f"  normal residual at point 1: {normal_residual_1:.12e} m\n")
        handle.write(f"  normal residual at point 2: {normal_residual_2:.12e} m\n\n")

        handle.write("Mesh:\n")
        handle.write(f"  blocks: {len(blocks)}\n")
        handle.write(f"  total structured cells: {total_cells}\n")
        handle.write(f"  total block-counted points: {total_nodes}\n")
        handle.write(f"  left/base target wall-tangent spacing: {target_tangent:.12e} m\n")
        handle.write(f"  first-interface tangent factor: {first_interface_tangent_factor:.12e}\n")
        handle.write(f"  right-block tangent factor: {right_tangent_factor:.12e}\n")
        handle.write(
            "  shared-vertex tangent targets O/1/2/3 and V/Q1/Q2/P: "
            + " / ".join(f"{value:.12e}" for value in geometry["vertex_tangent"])
            + " m\n"
        )
        handle.write(f"  first wall-normal spacing along grid line: {first_height:.12e} m\n")
        handle.write(f"  wall-normal cells: {n_normal}\n")
        handle.write(f"  interface width-matching band on each side: {int(geometry['interface_relax_cells'][0])} cells\n")
        for idx, block in enumerate(blocks):
            handle.write(f"  {block.name}: nodes={block.ni} x {block.nj}, cells={block.ni - 1} x {block.nj - 1}\n")
        handle.write(f"  segment cells O-1 / 1-2 / 2-3: {int(n_cells[0])} / {int(n_cells[1])} / {int(n_cells[2])}\n")
        for key, value in wall_stats.items():
            handle.write(f"  {key}: {value:.12e}\n")
        for key, value in growth.items():
            handle.write(f"  {key}: {value:.12e}\n")
        handle.write(f"  min signed cell area: {min_area:.12e} m^2\n")
        handle.write(f"  max signed cell area: {max_area:.12e} m^2\n")
        handle.write("\nBlock internal i-direction smoothness:\n")
        for entry in tangent_smoothness:
            handle.write(
                f"  {entry['name']}: width=[{entry['min_width']:.12e}, {entry['max_width']:.12e}] m, "
                f"max_adjacent_ratio={entry['max_adjacent_ratio']:.12e}, "
                f"p99_adjacent_ratio={entry['p99_adjacent_ratio']:.12e}\n"
            )
        handle.write("\nBlock interface checks:\n")
        for entry in interfaces:
            handle.write(
                f"  {entry['left']} <-> {entry['right']}: "
                f"node_mismatch={entry['node_mismatch']:.12e} m, "
                f"normal_ratio_max={entry['normal_ratio_max']:.12e}, "
                f"normal_ratio_p95={entry['normal_ratio_p95']:.12e}, "
                f"normal_ratio_mean={entry['normal_ratio_mean']:.12e}, "
                f"tangent_mismatch={entry['tangent_mismatch']:.12e} m\n"
            )
        handle.write("\ny+ and wall-parallel plus estimates using first cell center:\n")
        for name, entry in yplus.items():
            handle.write(
                f"  {name}: max_yplus={entry['max_yplus']:.12e}, "
                f"min_yplus={entry['min_yplus']:.12e}, "
                f"max_wall_parallel_plus={entry['max_wall_parallel_plus']:.12e}, "
                f"min_wall_parallel_plus={entry['min_wall_parallel_plus']:.12e}, "
                f"mu={entry['mu']:.12e} kg/(m s)\n"
            )
        handle.write("  target: first-cell-center y+ < 1.0\n\n")
        handle.write("Boundary labels:\n")
        handle.write("  2 wall: O-1-2-3\n")
        handle.write("  5 inflow/farfield: V-Q1-Q2-P and O-V\n")
        handle.write("  6 outflow: 3-P\n")
        handle.write("  -1 block interface: 1-Q1 and 2-Q2\n")


def plot_geometry(path: Path, geometry: dict[str, np.ndarray]) -> None:
    q1 = geometry["Q1"]
    q2 = geometry["Q2"]
    ys = np.linspace(PARABOLA_VERTEX[1], PARABOLA_RIGHT_POINT[1], 500)
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    wall = np.vstack((POINT_O, POINT_1, POINT_2, POINT_3))
    ax.plot(wall[:, 0], wall[:, 1], color="black", linewidth=2.4, label="wall")
    ax.plot(parabola_x(ys), ys, color="#1b9e77", linewidth=2.4, label="parabola")
    ax.plot([POINT_O[0], PARABOLA_VERTEX[0]], [POINT_O[1], PARABOLA_VERTEX[1]], color="#1b9e77", linewidth=1.8)
    ax.plot([POINT_3[0], PARABOLA_RIGHT_POINT[0]], [POINT_3[1], PARABOLA_RIGHT_POINT[1]], color="#d62728", linewidth=2.0, label="outflow")
    for p0, p1 in ((POINT_1, q1), (POINT_2, q2)):
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color="#202020", linewidth=1.8)
    for label, point in (("O", POINT_O), ("1", POINT_1), ("2", POINT_2), ("3", POINT_3), ("V", PARABOLA_VERTEX), ("P", PARABOLA_RIGHT_POINT), ("Q1", q1), ("Q2", q2)):
        ax.scatter(point[0], point[1], s=26, color="black", zorder=5)
        ax.text(point[0] + 0.0012, point[1] + 0.0012, label, fontsize=10)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, color="0.88", linewidth=0.7)
    ax.legend(loc="upper left")
    ax.set_title("Parabola-top three-block geometry")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_block_layout(path: Path, blocks: list[Block]) -> None:
    colors = ["#d8ecff", "#e4f4d8", "#fde8d7"]
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    for idx, block in enumerate(blocks):
        poly = np.vstack(
            (
                face_nodes(block, "jmin"),
                face_nodes(block, "imax")[1:],
                face_nodes(block, "jmax")[-2::-1],
                face_nodes(block, "imin")[-2:0:-1],
            )
        )
        ax.fill(poly[:, 0], poly[:, 1], facecolor=colors[idx], edgecolor="0.15", linewidth=1.6, alpha=0.85)
        cx = float(block.x.mean())
        cy = float(block.y.mean())
        ax.text(cx, cy, block.name, ha="center", va="center", fontsize=9)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, color="0.88", linewidth=0.7)
    ax.set_title("Three structured blocks")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_mesh(path: Path, blocks: list[Block], closeup: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(10.0, 6.8))
    for block in blocks:
        istep = max(1, (block.ni - 1) // 80)
        jstep = max(1, (block.nj - 1) // 60)
        if closeup:
            istep = max(1, (block.ni - 1) // 120)
            jstep = max(1, (block.nj - 1) // 90)
        for i in range(0, block.ni, istep):
            ax.plot(block.x[i, :], block.y[i, :], color="0.55", linewidth=0.26)
        if (block.ni - 1) % istep != 0:
            ax.plot(block.x[-1, :], block.y[-1, :], color="0.55", linewidth=0.26)
        for j in range(0, block.nj, jstep):
            ax.plot(block.x[:, j], block.y[:, j], color="0.55", linewidth=0.26)
        if (block.nj - 1) % jstep != 0:
            ax.plot(block.x[:, -1], block.y[:, -1], color="0.55", linewidth=0.26)
        for face in ("imin", "imax", "jmin", "jmax"):
            nodes = face_nodes(block, face)
            ax.plot(nodes[:, 0], nodes[:, 1], color="0.06", linewidth=1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, color="0.90", linewidth=0.5)
    if closeup:
        ax.set_xlim(0.038, 0.064)
        ax.set_ylim(0.020, 0.067)
        ax.set_title("Mesh close-up around points 1 and 2")
    else:
        ax.set_title("Parabola-top structured mesh preview")
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)


def clean_generated_dirs(work_dir: Path) -> None:
    for child in ("grid", "boundary_condition"):
        path = work_dir / child
        if path.exists():
            shutil.rmtree(path)


def generate(args: argparse.Namespace) -> None:
    work_dir = Path(args.output_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    clean_generated_dirs(work_dir)

    blocks, geometry = build_blocks(
        args.target_wall_tangent,
        args.first_interface_tangent_factor,
        args.right_tangent_factor,
        args.n_normal,
        args.first_wall_height,
        args.interface_relax_cells,
    )
    write_plot3d(work_dir / PLOT3D_NAME, blocks)
    write_boundary(work_dir / BOUNDARY_NAME, blocks)
    write_summary(
        work_dir / SUMMARY_NAME,
        blocks,
        geometry,
        args.target_wall_tangent,
        args.first_interface_tangent_factor,
        args.right_tangent_factor,
        args.first_wall_height,
        args.n_normal,
    )
    plot_geometry(work_dir / "geometry.png", geometry)
    plot_block_layout(work_dir / "block_layout.png", blocks)
    plot_mesh(work_dir / "mesh_preview.png", blocks, closeup=False)
    plot_mesh(work_dir / "mesh_closeup.png", blocks, closeup=True)

    read_grid(
        gridgen_or_pointwise=0,
        dimension=2,
        grid_file_name=str(work_dir / PLOT3D_NAME),
        boundary_file_name=str(work_dir / BOUNDARY_NAME),
        n_proc=args.n_proc,
        is_binary=True,
        write_binary=True,
        set_z=True,
        z_value=0.0,
        output_dir=str(work_dir),
    )

    wall_stats = wall_spacing_stats(blocks)
    interfaces = interface_stats(blocks)
    yplus = yplus_stats(blocks)
    print(f"Generated {len(blocks)} blocks in {work_dir}")
    print(f"Block cells: {', '.join(str(block.ni - 1) + 'x' + str(block.nj - 1) for block in blocks)}")
    print(f"Parabola vertex: ({PARABOLA_VERTEX[0]:.6e}, {PARABOLA_VERTEX[1]:.6e})")
    print(f"Parabola right point: ({PARABOLA_RIGHT_POINT[0]:.6e}, {PARABOLA_RIGHT_POINT[1]:.6e})")
    print(f"Q1: ({geometry['Q1'][0]:.6e}, {geometry['Q1'][1]:.6e})")
    print(f"Q2: ({geometry['Q2'][0]:.6e}, {geometry['Q2'][1]:.6e})")
    print(f"Shared vertex tangent targets: {', '.join(f'{value:.6e}' for value in geometry['vertex_tangent'])} m")
    print(f"Wall tangent spacing: {wall_stats['wall_tangent_min']:.6e} - {wall_stats['wall_tangent_max']:.6e} m")
    print(f"First projected wall spacing: {wall_stats['first_projected_min']:.6e} - {wall_stats['first_projected_max']:.6e} m")
    print(f"Max interface node mismatch: {max(entry['node_mismatch'] for entry in interfaces):.6e} m")
    print(f"Max interface adjacent-width ratio: {max(entry['normal_ratio_max'] for entry in interfaces):.6f}")
    print(f"Max first-cell-center y+: {max(entry['max_yplus'] for entry in yplus.values()):.6f}")
    print(f"Summary file: {work_dir / SUMMARY_NAME}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(WORK_DIR))
    parser.add_argument("--n-proc", type=int, default=1)
    parser.add_argument("--target-wall-tangent", type=float, default=TARGET_WALL_TANGENT)
    parser.add_argument("--first-interface-tangent-factor", type=float, default=FIRST_INTERFACE_TANGENT_FACTOR)
    parser.add_argument("--right-tangent-factor", type=float, default=RIGHT_TANGENT_FACTOR)
    parser.add_argument("--first-wall-height", type=float, default=FIRST_WALL_HEIGHT)
    parser.add_argument("--n-normal", type=int, default=N_WALL_NORMAL)
    parser.add_argument("--interface-relax-cells", type=int, default=INTERFACE_RELAX_CELLS)
    return parser.parse_args()


if __name__ == "__main__":
    generate(parse_args())

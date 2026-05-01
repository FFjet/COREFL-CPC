#!/usr/bin/env python3
"""Generate a simple rectangular-top multiblock mesh for the double wedge."""

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
BC_SYMMETRY = 3
BC_INFLOW = 5
BC_OUTFLOW = 6
BC_INTERFACE = -1

POINT_V = np.array([-0.0100, 0.0000], dtype=float)
POINT_O = np.array([0.0000, 0.0000], dtype=float)
POINT_1 = np.array([0.0440, 0.0254], dtype=float)
POINT_2 = np.array([0.0586, 0.0462], dtype=float)
POINT_3 = np.array([0.0776, 0.0462], dtype=float)
TOP_HEIGHT = 0.08
POINT_TL = np.array([POINT_V[0], POINT_3[1] + TOP_HEIGHT], dtype=float)
POINT_TR = np.array([POINT_3[0], POINT_3[1] + TOP_HEIGHT], dtype=float)

TARGET_WALL_TANGENT = 1.0e-5
FIRST_WALL_HEIGHT = 2.93e-6
N_WALL_NORMAL = 640

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


def smooth_widths_with_endpoints(length: float, n_cell: int, start_width: float, end_width: float) -> np.ndarray:
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
    correction = (length - float(linear.sum())) / float(bubble.sum())
    widths = linear + correction * bubble
    if np.any(widths <= 0.0):
        widths = np.full(n_cell, length / n_cell, dtype=float)
    widths *= length / float(widths.sum())
    return widths


def geometric_widths(length: float, n_cell: int, first_width: float) -> np.ndarray:
    if n_cell <= 0:
        raise ValueError("n_cell must be positive")
    if length <= 0.0:
        raise ValueError("length must be positive")
    first_width = min(first_width, 0.2 * length)
    if first_width * n_cell >= 0.98 * length:
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
    widths *= length / float(widths.sum())
    return widths


def line_segment_with_widths(start: np.ndarray, end: np.ndarray, widths: np.ndarray) -> np.ndarray:
    direction = end - start
    length = float(np.linalg.norm(direction))
    if length <= 0.0:
        raise ValueError("zero-length segment")
    direction /= length
    s = np.empty(widths.size + 1, dtype=float)
    s[0] = 0.0
    s[1:] = np.cumsum(widths)
    s[-1] = length
    pts = start[None, :] + s[:, None] * direction[None, :]
    pts[0] = start
    pts[-1] = end
    return pts


def straight_segment(start: np.ndarray, end: np.ndarray, n_cell: int) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n_cell + 1, dtype=float)[:, None]
    pts = (1.0 - t) * start[None, :] + t * end[None, :]
    pts[0] = start
    pts[-1] = end
    return pts


def vertical_line(bottom: np.ndarray, n_normal: int, first_height: float) -> np.ndarray:
    top = np.array([bottom[0], POINT_TL[1]], dtype=float)
    length = float(top[1] - bottom[1])
    widths = geometric_widths(length, n_normal, first_height)
    return line_segment_with_widths(bottom, top, widths)


def build_quad_patch(left: np.ndarray, right: np.ndarray, n_tangent: int) -> tuple[np.ndarray, np.ndarray]:
    if left.shape != right.shape:
        raise ValueError("left and right boundary arrays must match")
    n_normal = left.shape[0] - 1
    x = np.empty((n_tangent + 1, n_normal + 1), dtype=float)
    y = np.empty((n_tangent + 1, n_normal + 1), dtype=float)
    for j in range(n_normal + 1):
        row = straight_segment(left[j], right[j], n_tangent)
        x[:, j] = row[:, 0]
        y[:, j] = row[:, 1]
    return x, y


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


def build_blocks(target_tangent: float, n_normal: int, first_height: float) -> tuple[list[Block], dict[str, np.ndarray]]:
    bottom_points = [POINT_V, POINT_O, POINT_1, POINT_2, POINT_3]
    verticals = [vertical_line(point, n_normal, first_height) for point in bottom_points]

    bottom_lengths = [float(np.linalg.norm(bottom_points[i + 1] - bottom_points[i])) for i in range(4)]
    x_lengths = [float(abs(bottom_points[i + 1][0] - bottom_points[i][0])) for i in range(4)]
    n_cells = [max(8, int(round(x_lengths[i] / target_tangent))) for i in range(4)]

    arrays = [build_quad_patch(verticals[i], verticals[i + 1], n_cells[i]) for i in range(4)]
    b1_x, b1_y = arrays[0]
    b2_x, b2_y = arrays[1]
    b3_x, b3_y = arrays[2]
    b4_x, b4_y = arrays[3]

    for left_x, left_y, right_x, right_y in ((b1_x, b1_y, b2_x, b2_y), (b2_x, b2_y, b3_x, b3_y), (b3_x, b3_y, b4_x, b4_y)):
        right_x[0, :] = left_x[-1, :]
        right_y[0, :] = left_y[-1, :]

    blocks = [
        Block(
            "B1_V_to_O",
            b1_x,
            b1_y,
            [
                BoundaryEntry(face_range(b1_x.shape[0], b1_x.shape[1], "imin"), BC_INFLOW),
                BoundaryEntry(face_range(b1_x.shape[0], b1_x.shape[1], "imax"), BC_INTERFACE, face_range(b2_x.shape[0], b2_x.shape[1], "imin"), 2),
                BoundaryEntry(face_range(b1_x.shape[0], b1_x.shape[1], "jmin"), BC_SYMMETRY),
                BoundaryEntry(face_range(b1_x.shape[0], b1_x.shape[1], "jmax"), BC_SYMMETRY),
            ],
        ),
        Block(
            "B2_O_to_1",
            b2_x,
            b2_y,
            [
                BoundaryEntry(face_range(b2_x.shape[0], b2_x.shape[1], "imin"), BC_INTERFACE, face_range(b1_x.shape[0], b1_x.shape[1], "imax"), 1),
                BoundaryEntry(face_range(b2_x.shape[0], b2_x.shape[1], "imax"), BC_INTERFACE, face_range(b3_x.shape[0], b3_x.shape[1], "imin"), 3),
                BoundaryEntry(face_range(b2_x.shape[0], b2_x.shape[1], "jmin"), BC_WALL),
                BoundaryEntry(face_range(b2_x.shape[0], b2_x.shape[1], "jmax"), BC_SYMMETRY),
            ],
        ),
        Block(
            "B3_1_to_2",
            b3_x,
            b3_y,
            [
                BoundaryEntry(face_range(b3_x.shape[0], b3_x.shape[1], "imin"), BC_INTERFACE, face_range(b2_x.shape[0], b2_x.shape[1], "imax"), 2),
                BoundaryEntry(face_range(b3_x.shape[0], b3_x.shape[1], "imax"), BC_INTERFACE, face_range(b4_x.shape[0], b4_x.shape[1], "imin"), 4),
                BoundaryEntry(face_range(b3_x.shape[0], b3_x.shape[1], "jmin"), BC_WALL),
                BoundaryEntry(face_range(b3_x.shape[0], b3_x.shape[1], "jmax"), BC_SYMMETRY),
            ],
        ),
        Block(
            "B4_2_to_3",
            b4_x,
            b4_y,
            [
                BoundaryEntry(face_range(b4_x.shape[0], b4_x.shape[1], "imin"), BC_INTERFACE, face_range(b3_x.shape[0], b3_x.shape[1], "imax"), 3),
                BoundaryEntry(face_range(b4_x.shape[0], b4_x.shape[1], "imax"), BC_OUTFLOW),
                BoundaryEntry(face_range(b4_x.shape[0], b4_x.shape[1], "jmin"), BC_WALL),
                BoundaryEntry(face_range(b4_x.shape[0], b4_x.shape[1], "jmax"), BC_SYMMETRY),
            ],
        ),
    ]

    geometry = {
        "bottom_points": np.array(bottom_points, dtype=float),
        "top_points": np.array([[point[0], POINT_TL[1]] for point in bottom_points], dtype=float),
        "n_cells": np.array(n_cells, dtype=int),
        "bottom_lengths": np.array(bottom_lengths, dtype=float),
        "x_lengths": np.array(x_lengths, dtype=float),
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
        handle.write("# double-wedge rectangular-top boundary file\n")
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


def interface_stats(blocks: list[Block]) -> list[dict[str, float | str]]:
    stats: list[dict[str, float | str]] = []
    for idx in range(len(blocks) - 1):
        left = blocks[idx]
        right = blocks[idx + 1]
        left_nodes = face_nodes(left, "imax")
        right_nodes = face_nodes(right, "imin")
        mismatch = float(np.linalg.norm(left_nodes - right_nodes, axis=1).max())
        left_width = np.abs(left_nodes[:, 0] - left.x[-2, :])
        right_width = np.abs(right.x[1, :] - right_nodes[:, 0])
        ratio = np.maximum(left_width / right_width, right_width / left_width)
        tangent_left = np.linalg.norm(np.diff(left_nodes, axis=0), axis=1)
        tangent_right = np.linalg.norm(np.diff(right_nodes, axis=0), axis=1)
        stats.append(
            {
                "left": left.name,
                "right": right.name,
                "node_mismatch": mismatch,
                "dx_ratio_max": float(ratio.max()),
                "dx_ratio_mean": float(ratio.mean()),
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


def face_tangent_width(block: Block, face: str) -> np.ndarray:
    nodes = face_nodes(block, face)
    return np.linalg.norm(np.diff(nodes, axis=0), axis=1)


def normal_widths(block: Block) -> np.ndarray:
    return np.hypot(np.diff(block.x, axis=1), np.diff(block.y, axis=1))


def adjacent_ratio(values: np.ndarray) -> float:
    local = values[1:] / values[:-1]
    ratio = np.maximum(local, 1.0 / local)
    return float(ratio.max()) if ratio.size else 1.0


def mesh_stats(blocks: list[Block]) -> dict[str, float]:
    x_spacing: list[np.ndarray] = []
    wall_tangent: list[np.ndarray] = []
    top_tangent: list[np.ndarray] = []
    wall_first: list[np.ndarray] = []
    top_first: list[np.ndarray] = []
    tangent_ratios: list[float] = []
    normal_ratios: list[float] = []

    for block in blocks:
        x_spacing.append(np.abs(np.diff(block.x, axis=0)).ravel())
        wall_tangent.append(face_tangent_width(block, "jmin"))
        top_tangent.append(face_tangent_width(block, "jmax"))
        wall_first.append(normal_widths(block)[:, 0])
        top_first.append(normal_widths(block)[:, -1])
        widths_i = np.hypot(np.diff(block.x, axis=0), np.diff(block.y, axis=0))
        tangent_ratios.append(float(np.maximum(widths_i[1:, :] / widths_i[:-1, :], widths_i[:-1, :] / widths_i[1:, :]).max()))
        widths_j = normal_widths(block)
        normal_ratios.append(float(np.maximum(widths_j[:, 1:] / widths_j[:, :-1], widths_j[:, :-1] / widths_j[:, 1:]).max()))

    x_spacing_all = np.concatenate(x_spacing)
    wall_tangent_all = np.concatenate(wall_tangent)
    top_tangent_all = np.concatenate(top_tangent)
    wall_first_all = np.concatenate(wall_first)
    top_first_all = np.concatenate(top_first)
    return {
        "x_spacing_min": float(x_spacing_all.min()),
        "x_spacing_max": float(x_spacing_all.max()),
        "wall_tangent_min": float(wall_tangent_all.min()),
        "wall_tangent_max": float(wall_tangent_all.max()),
        "top_tangent_min": float(top_tangent_all.min()),
        "top_tangent_max": float(top_tangent_all.max()),
        "wall_first_min": float(wall_first_all.min()),
        "wall_first_max": float(wall_first_all.max()),
        "top_first_min": float(top_first_all.min()),
        "top_first_max": float(top_first_all.max()),
        "tangent_adjacent_ratio_max": max(tangent_ratios),
        "normal_adjacent_ratio_max": max(normal_ratios),
    }


def wall_s_locations(blocks: list[Block]) -> tuple[np.ndarray, np.ndarray]:
    s_centers: list[np.ndarray] = []
    ds_values: list[np.ndarray] = []
    s0 = 0.0
    for block in blocks[1:]:
        wall = face_nodes(block, "jmin")
        ds = np.linalg.norm(np.diff(wall, axis=0), axis=1)
        centers = s0 + np.cumsum(ds) - 0.5 * ds
        s_centers.append(centers)
        ds_values.append(ds)
        s0 += float(ds.sum())
    return np.concatenate(s_centers), np.concatenate(ds_values)


def yplus_stats(blocks: list[Block]) -> dict[str, dict[str, float]]:
    s_centers, ds_values = wall_s_locations(blocks)
    stats = mesh_stats(blocks)
    y_center = 0.5 * stats["wall_first_min"]
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


def write_summary(path: Path, blocks: list[Block], geometry: dict[str, np.ndarray], target_tangent: float, first_height: float, n_normal: int) -> None:
    total_cells = sum((b.ni - 1) * (b.nj - 1) for b in blocks)
    total_nodes = sum(b.ni * b.nj for b in blocks)
    stats = mesh_stats(blocks)
    interfaces = interface_stats(blocks)
    min_area = min(float(signed_cell_areas(block).min()) for block in blocks)
    max_area = max(float(signed_cell_areas(block).max()) for block in blocks)
    yplus = yplus_stats(blocks)

    with path.open("w", encoding="utf-8") as handle:
        handle.write("Double-wedge rectangular-top mesh summary\n\n")
        handle.write("Geometry points [m]:\n")
        for name, point in (
            ("V", POINT_V),
            ("O", POINT_O),
            ("1", POINT_1),
            ("2", POINT_2),
            ("3", POINT_3),
            ("TL", POINT_TL),
            ("TR", POINT_TR),
        ):
            handle.write(f"  {name}: x={point[0]:.12e}, y={point[1]:.12e}\n")
        handle.write(f"  top height above point 3: {TOP_HEIGHT:.12e} m\n\n")

        handle.write("Mesh:\n")
        handle.write(f"  blocks: {len(blocks)}\n")
        handle.write(f"  total structured cells: {total_cells}\n")
        handle.write(f"  total block-counted points: {total_nodes}\n")
        handle.write(f"  target x-direction spacing: {target_tangent:.12e} m\n")
        handle.write(f"  first wall-normal spacing: {first_height:.12e} m\n")
        handle.write(f"  normal cells: {n_normal}\n")
        for block in blocks:
            handle.write(f"  {block.name}: nodes={block.ni} x {block.nj}, cells={block.ni - 1} x {block.nj - 1}\n")
        handle.write(f"  segment cells V-O / O-1 / 1-2 / 2-3: {' / '.join(str(int(v)) for v in geometry['n_cells'])}\n")
        for key, value in stats.items():
            handle.write(f"  {key}: {value:.12e}\n")
        handle.write(f"  min signed cell area: {min_area:.12e} m^2\n")
        handle.write(f"  max signed cell area: {max_area:.12e} m^2\n\n")

        handle.write("Block interface checks:\n")
        for entry in interfaces:
            handle.write(
                f"  {entry['left']} <-> {entry['right']}: "
                f"node_mismatch={entry['node_mismatch']:.12e} m, "
                f"adjacent_dx_ratio_max={entry['dx_ratio_max']:.12e}, "
                f"adjacent_dx_ratio_mean={entry['dx_ratio_mean']:.12e}, "
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
        handle.write("  3 symmetry: V-O and full rectangular top\n")
        handle.write("  5 inflow: V-TL\n")
        handle.write("  6 outflow: 3-TR\n")
        handle.write("  -1 block interfaces: vertical lines at O, 1, 2\n")


def plot_geometry(path: Path, geometry: dict[str, np.ndarray]) -> None:
    fig, ax = plt.subplots(figsize=(10, 6.5))
    bottom = geometry["bottom_points"]
    top = geometry["top_points"]
    ax.plot(bottom[:, 0], bottom[:, 1], color="black", linewidth=2.4, label="wall/symmetry")
    ax.plot(top[:, 0], top[:, 1], color="#1b9e77", linewidth=2.4, label="top wall")
    ax.plot([POINT_V[0], POINT_TL[0]], [POINT_V[1], POINT_TL[1]], color="#1b9e77", linewidth=1.8, label="inflow")
    ax.plot([POINT_3[0], POINT_TR[0]], [POINT_3[1], POINT_TR[1]], color="#d62728", linewidth=2.0, label="outflow")
    for i in range(1, len(bottom) - 1):
        ax.plot([bottom[i, 0], top[i, 0]], [bottom[i, 1], top[i, 1]], color="0.20", linewidth=1.3)
    for label, point in (("V", POINT_V), ("O", POINT_O), ("1", POINT_1), ("2", POINT_2), ("3", POINT_3), ("TL", POINT_TL), ("TR", POINT_TR)):
        ax.scatter(point[0], point[1], s=26, color="black", zorder=5)
        ax.text(point[0] + 0.001, point[1] + 0.001, label, fontsize=10)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, color="0.88", linewidth=0.7)
    ax.legend(loc="upper left")
    ax.set_title("Rectangular-top block geometry")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_block_layout(path: Path, blocks: list[Block]) -> None:
    colors = ["#d8ecff", "#e4f4d8", "#fde8d7", "#eadcf8"]
    fig, ax = plt.subplots(figsize=(10, 6.5))
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
        ax.text(float(block.x.mean()), float(block.y.mean()), block.name, ha="center", va="center", fontsize=9)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, color="0.88", linewidth=0.7)
    ax.set_title("Four rectangular-top structured blocks")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_mesh(path: Path, blocks: list[Block], closeup: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(10, 6.8))
    for block in blocks:
        istep = max(1, (block.ni - 1) // (180 if closeup else 140))
        jstep = max(1, (block.nj - 1) // (110 if closeup else 80))
        for i in range(0, block.ni, istep):
            ax.plot(block.x[i, :], block.y[i, :], color="0.55", linewidth=0.26)
        ax.plot(block.x[-1, :], block.y[-1, :], color="0.55", linewidth=0.26)
        for j in range(0, block.nj, jstep):
            ax.plot(block.x[:, j], block.y[:, j], color="0.55", linewidth=0.26)
        ax.plot(block.x[:, -1], block.y[:, -1], color="0.55", linewidth=0.26)
        for face in ("imin", "imax", "jmin", "jmax"):
            nodes = face_nodes(block, face)
            ax.plot(nodes[:, 0], nodes[:, 1], color="0.06", linewidth=1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, color="0.90", linewidth=0.5)
    if closeup:
        ax.set_xlim(-0.003, 0.064)
        ax.set_ylim(-0.002, 0.060)
        ax.set_title("Mesh close-up around wall breaks")
    else:
        ax.set_title("Rectangular-top structured mesh preview")
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

    blocks, geometry = build_blocks(args.target_wall_tangent, args.n_normal, args.first_wall_height)
    write_plot3d(work_dir / PLOT3D_NAME, blocks)
    write_boundary(work_dir / BOUNDARY_NAME, blocks)
    write_summary(work_dir / SUMMARY_NAME, blocks, geometry, args.target_wall_tangent, args.first_wall_height, args.n_normal)
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

    stats = mesh_stats(blocks)
    interfaces = interface_stats(blocks)
    yplus = yplus_stats(blocks)
    print(f"Generated {len(blocks)} blocks in {work_dir}")
    print(f"Block cells: {', '.join(str(block.ni - 1) + 'x' + str(block.nj - 1) for block in blocks)}")
    print(f"Top-left point: ({POINT_TL[0]:.6e}, {POINT_TL[1]:.6e})")
    print(f"Top-right point: ({POINT_TR[0]:.6e}, {POINT_TR[1]:.6e})")
    print(f"X-direction spacing: {stats['x_spacing_min']:.6e} - {stats['x_spacing_max']:.6e} m")
    print(f"Wall tangent spacing: {stats['wall_tangent_min']:.6e} - {stats['wall_tangent_max']:.6e} m")
    print(f"Wall first spacing: {stats['wall_first_min']:.6e} - {stats['wall_first_max']:.6e} m")
    print(f"Top first spacing: {stats['top_first_min']:.6e} - {stats['top_first_max']:.6e} m")
    print(f"Max interface node mismatch: {max(entry['node_mismatch'] for entry in interfaces):.6e} m")
    print(f"Max interface adjacent-dx ratio: {max(entry['dx_ratio_max'] for entry in interfaces):.6f}")
    print(f"Max first-cell-center y+: {max(entry['max_yplus'] for entry in yplus.values()):.6f}")
    print(f"Summary file: {work_dir / SUMMARY_NAME}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(WORK_DIR))
    parser.add_argument("--n-proc", type=int, default=1)
    parser.add_argument("--target-wall-tangent", type=float, default=TARGET_WALL_TANGENT)
    parser.add_argument("--first-wall-height", type=float, default=FIRST_WALL_HEIGHT)
    parser.add_argument("--n-normal", type=int, default=N_WALL_NORMAL)
    return parser.parse_args()


if __name__ == "__main__":
    generate(parse_args())

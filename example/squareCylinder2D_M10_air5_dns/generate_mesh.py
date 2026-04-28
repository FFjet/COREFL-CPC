#!/usr/bin/env python3
"""Generate the 2D structured mesh for the Mach-10 square-cylinder case."""

from __future__ import annotations

import math
import sys
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
PLOT3D_NAME = "squareCylinder.xyz"
BOUNDARY_NAME = "squareCylinder.inp"
SU2_NAME = "squareCylinder.su2"
PREVIEW_NAME = "mesh_preview.png"
SUMMARY_NAME = "mesh_summary.txt"

BC_WALL = 2
BC_SYMMETRY = 3
BC_INFLOW = 5
BC_OUTFLOW = 6
BC_INTERFACE = -1

X_LINES = np.array([0.0, 0.003, 0.004, 0.006, 0.007, 0.015], dtype=float)
Y_LINES = np.array([-0.005, -0.002, -0.001, 0.001, 0.002, 0.005], dtype=float)
X_CELLS = [88, 72, 132, 72, 220]
Y_CELLS = [88, 72, 132, 72, 88]
HOLE_CELL = (2, 2)

P_INF = 1000.0
T_INF = 300.0
MACH_INF = 10.0
GAMMA_INF = 1.4
R_AIR = 287.052874
MU_REF = 1.716e-5
T_REF = 273.15
SUTHERLAND_C = 110.4
MW_AIR = 28.97e-3
BODY_SIZE = 0.002
YPLUS_TARGET = 0.9


def sutherland_viscosity(temperature: float) -> float:
    return MU_REF * (temperature / T_REF) ** 1.5 * (T_REF + SUTHERLAND_C) / (temperature + SUTHERLAND_C)


def estimate_wall_spacing() -> tuple[float, dict[str, float]]:
    rho_inf = P_INF * MW_AIR / (8.314462618 * T_INF)
    a_inf = math.sqrt(GAMMA_INF * R_AIR * T_INF)
    u_inf = MACH_INF * a_inf
    mu_inf = sutherland_viscosity(T_INF)
    re_body = rho_inf * u_inf * BODY_SIZE / mu_inf
    cf_lam = 0.664 / math.sqrt(max(re_body, 1.0))
    u_tau = u_inf * math.sqrt(0.5 * cf_lam)
    y_center = YPLUS_TARGET * mu_inf / (rho_inf * u_tau)
    dy_wall = 2.0 * y_center
    return dy_wall, {
        "rho_inf": rho_inf,
        "a_inf": a_inf,
        "u_inf": u_inf,
        "mu_inf": mu_inf,
        "re_body": re_body,
        "cf_lam": cf_lam,
        "u_tau": u_tau,
        "y_center": y_center,
        "dy_wall": dy_wall,
    }


WALL_NORMAL_DN, WALL_INFO = estimate_wall_spacing()


def geometric_widths(length: float, n_cell: int, first_width: float) -> np.ndarray:
    if n_cell <= 0:
        raise ValueError("n_cell must be positive")
    if first_width <= 0.0:
        raise ValueError("first_width must be positive")
    if abs(first_width * n_cell - length) <= 1e-14 * max(length, 1.0):
        return np.full(n_cell, length / n_cell, dtype=float)

    def total(ratio: float) -> float:
        if abs(ratio - 1.0) < 1e-12:
            return first_width * n_cell
        return first_width * (ratio**n_cell - 1.0) / (ratio - 1.0)

    if first_width * n_cell < length:
        lo, hi = 1.0, 1.01
        while total(hi) < length:
            hi = 1.0 + (hi - 1.0) * 2.0
    else:
        lo, hi = 0.99, 1.0
        while total(lo) > length:
            lo *= 0.5

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


def segment_from_start(x0: float, x1: float, n_cell: int, first_width: float) -> np.ndarray:
    widths = geometric_widths(x1 - x0, n_cell, first_width)
    pts = np.empty(n_cell + 1, dtype=float)
    pts[0] = x0
    pts[1:] = x0 + np.cumsum(widths)
    pts[-1] = x1
    return pts


def segment_to_end(x0: float, x1: float, n_cell: int, last_width: float) -> np.ndarray:
    widths = geometric_widths(x1 - x0, n_cell, last_width)[::-1]
    pts = np.empty(n_cell + 1, dtype=float)
    pts[0] = x0
    pts[1:] = x0 + np.cumsum(widths)
    pts[-1] = x1
    return pts


def uniform_segment(x0: float, x1: float, n_cell: int) -> np.ndarray:
    return np.linspace(x0, x1, n_cell + 1, dtype=float)


def symmetric_segment_from_ends(x0: float, x1: float, n_cell: int, end_width: float) -> np.ndarray:
    if n_cell % 2 != 0:
        raise ValueError("symmetric center segments require an even number of cells")
    mid = 0.5 * (x0 + x1)
    left = segment_from_start(x0, mid, n_cell // 2, end_width)
    right = 2.0 * mid - left[-2::-1]
    pts = np.concatenate((left, right))
    pts[0] = x0
    pts[n_cell // 2] = mid
    pts[-1] = x1
    return pts


def build_x_segments() -> list[np.ndarray]:
    wall_left = segment_to_end(X_LINES[1], X_LINES[2], X_CELLS[1], WALL_NORMAL_DN)
    wall_right = segment_from_start(X_LINES[3], X_LINES[4], X_CELLS[3], WALL_NORMAL_DN)
    left_outer = segment_to_end(X_LINES[0], X_LINES[1], X_CELLS[0], wall_left[1] - wall_left[0])
    center = symmetric_segment_from_ends(X_LINES[2], X_LINES[3], X_CELLS[2], WALL_NORMAL_DN)
    right_outer = segment_from_start(X_LINES[4], X_LINES[5], X_CELLS[4], wall_right[-1] - wall_right[-2])
    return [left_outer, wall_left, center, wall_right, right_outer]


def build_y_segments() -> list[np.ndarray]:
    wall_lower = segment_to_end(Y_LINES[1], Y_LINES[2], Y_CELLS[1], WALL_NORMAL_DN)
    wall_upper = segment_from_start(Y_LINES[3], Y_LINES[4], Y_CELLS[3], WALL_NORMAL_DN)
    lower_outer = segment_to_end(Y_LINES[0], Y_LINES[1], Y_CELLS[0], wall_lower[1] - wall_lower[0])
    center = symmetric_segment_from_ends(Y_LINES[2], Y_LINES[3], Y_CELLS[2], WALL_NORMAL_DN)
    upper_outer = segment_from_start(Y_LINES[4], Y_LINES[5], Y_CELLS[4], wall_upper[-1] - wall_upper[-2])
    return [lower_outer, wall_lower, center, wall_upper, upper_outer]


def build_blocks() -> tuple[list[dict], dict[tuple[int, int], int]]:
    x_segments = build_x_segments()
    y_segments = build_y_segments()
    blocks: list[dict] = []
    lookup: dict[tuple[int, int], int] = {}
    for j_seg, y_seg in enumerate(y_segments):
        for i_seg, x_seg in enumerate(x_segments):
            if (i_seg, j_seg) == HOLE_CELL:
                continue
            x, y = np.meshgrid(x_seg, y_seg, indexing="ij")
            block_id = len(blocks)
            blocks.append(
                {
                    "id": block_id,
                    "i_seg": i_seg,
                    "j_seg": j_seg,
                    "x": x[:, :, None],
                    "y": y[:, :, None],
                    "z": np.zeros_like(x)[:, :, None],
                    "nx": x.shape[0],
                    "ny": y.shape[1],
                }
            )
            lookup[(i_seg, j_seg)] = block_id
    return blocks, lookup


def face_range(block: dict, face: str) -> tuple[int, int, int, int, int]:
    nx, ny = block["nx"], block["ny"]
    if face == "imin":
        return (1, 1, 1, ny, 1)
    if face == "imax":
        return (nx, nx, 1, ny, 1)
    if face == "jmin":
        return (1, nx, 1, 1, 1)
    if face == "jmax":
        return (1, nx, ny, ny, 1)
    raise ValueError(face)


def neighbor_key(i_seg: int, j_seg: int, face: str) -> tuple[int, int]:
    if face == "imin":
        return i_seg - 1, j_seg
    if face == "imax":
        return i_seg + 1, j_seg
    if face == "jmin":
        return i_seg, j_seg - 1
    if face == "jmax":
        return i_seg, j_seg + 1
    raise ValueError(face)


def opposite_face(face: str) -> str:
    return {"imin": "imax", "imax": "imin", "jmin": "jmax", "jmax": "jmin"}[face]


def classify_physical_face(i_seg: int, j_seg: int, face: str) -> int:
    if face == "imin":
        return BC_INFLOW if i_seg == 0 else BC_WALL
    if face == "imax":
        return BC_OUTFLOW if i_seg == len(X_CELLS) - 1 else BC_WALL
    if face == "jmin":
        return BC_SYMMETRY if j_seg == 0 else BC_WALL
    if face == "jmax":
        return BC_SYMMETRY if j_seg == len(Y_CELLS) - 1 else BC_WALL
    raise ValueError(face)


def write_plot3d(path: Path, blocks: list[dict]) -> None:
    with path.open("wb") as handle:
        np.array([len(blocks)], dtype=np.int32).tofile(handle)
        np.array([(b["nx"], b["ny"], 1) for b in blocks], dtype=np.int32).tofile(handle)
        for block in blocks:
            block["x"].ravel(order="F").tofile(handle)
            block["y"].ravel(order="F").tofile(handle)
            block["z"].ravel(order="F").tofile(handle)


def write_boundary(path: Path, blocks: list[dict], lookup: dict[tuple[int, int], int]) -> None:
    faces = ("imin", "imax", "jmin", "jmax")
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# square cylinder multiblock boundary file\n")
        handle.write("# generated by generate_mesh.py\n")
        for block in blocks:
            entries: list[tuple[tuple[int, int, int, int, int], int, tuple[int, int, int, int, int] | None, int | None]] = []
            for face in faces:
                source = face_range(block, face)
                nkey = neighbor_key(block["i_seg"], block["j_seg"], face)
                if nkey in lookup:
                    target = blocks[lookup[nkey]]
                    entries.append((source, BC_INTERFACE, face_range(target, opposite_face(face)), lookup[nkey] + 1))
                else:
                    entries.append((source, classify_physical_face(block["i_seg"], block["j_seg"], face), None, None))
            handle.write(f"Block {block['id'] + 1}\n")
            handle.write("Boundary conditions\n")
            handle.write(f"{len(entries)}\n")
            for source, label, target_range, target_id in entries:
                handle.write(" ".join(f"{value:6d}" for value in (*source[:4], label)) + "\n")
                if target_range is not None and target_id is not None:
                    handle.write(" ".join(f"{value:6d}" for value in (*target_range[:4], target_id)) + "\n")


def write_su2(path: Path, blocks: list[dict]) -> None:
    points: dict[tuple[float, float], int] = {}
    coords: list[tuple[float, float]] = []
    elements: list[tuple[int, int, int, int]] = []
    markers: dict[str, list[tuple[int, int]]] = {"inflow": [], "outflow": [], "symmetry": [], "wall": []}

    def point_id(x: float, y: float) -> int:
        key = (round(x, 14), round(y, 14))
        if key not in points:
            points[key] = len(coords)
            coords.append((x, y))
        return points[key]

    for block in blocks:
        x = block["x"][:, :, 0]
        y = block["y"][:, :, 0]
        pid = np.empty_like(x, dtype=np.int64)
        for i in range(block["nx"]):
            for j in range(block["ny"]):
                pid[i, j] = point_id(float(x[i, j]), float(y[i, j]))
        for i in range(block["nx"] - 1):
            for j in range(block["ny"] - 1):
                elements.append((int(pid[i, j]), int(pid[i + 1, j]), int(pid[i + 1, j + 1]), int(pid[i, j + 1])))
        for face in ("imin", "imax", "jmin", "jmax"):
            if neighbor_key(block["i_seg"], block["j_seg"], face) in block_lookup:
                continue
            label = classify_physical_face(block["i_seg"], block["j_seg"], face)
            marker = {BC_INFLOW: "inflow", BC_OUTFLOW: "outflow", BC_SYMMETRY: "symmetry", BC_WALL: "wall"}[label]
            if face == "imin":
                for j in range(block["ny"] - 1):
                    markers[marker].append((int(pid[0, j]), int(pid[0, j + 1])))
            elif face == "imax":
                for j in range(block["ny"] - 1):
                    markers[marker].append((int(pid[-1, j]), int(pid[-1, j + 1])))
            elif face == "jmin":
                for i in range(block["nx"] - 1):
                    markers[marker].append((int(pid[i, 0]), int(pid[i + 1, 0])))
            else:
                for i in range(block["nx"] - 1):
                    markers[marker].append((int(pid[i, -1]), int(pid[i + 1, -1])))

    with path.open("w", encoding="utf-8") as handle:
        handle.write("NDIME= 2\n")
        handle.write(f"NPOIN= {len(coords)}\n")
        for i, (x, y) in enumerate(coords):
            handle.write(f"{x:.16e} {y:.16e} {i}\n")
        handle.write(f"NELEM= {len(elements)}\n")
        for i, elem in enumerate(elements):
            handle.write(f"9 {' '.join(str(v) for v in elem)} {i}\n")
        active_markers = [(name, edges) for name, edges in markers.items() if edges]
        handle.write(f"NMARK= {len(active_markers)}\n")
        for name, edges in active_markers:
            handle.write(f"MARKER_TAG= {name}\n")
            handle.write(f"MARKER_ELEMS= {len(edges)}\n")
            for a, b in edges:
                handle.write(f"3 {a} {b}\n")


def segment_join_stats(segments: list[np.ndarray]) -> list[dict]:
    stats: list[dict] = []
    for i in range(len(segments) - 1):
        left = float(segments[i][-1] - segments[i][-2])
        right = float(segments[i + 1][1] - segments[i + 1][0])
        stats.append(
            {
                "join": i,
                "left_width": left,
                "right_width": right,
                "ratio": max(left, right) / min(left, right),
            }
        )
    return stats


def segment_internal_stats(segments: list[np.ndarray]) -> list[dict]:
    stats: list[dict] = []
    for i, segment in enumerate(segments):
        widths = np.diff(segment)
        adjacent = widths[1:] / widths[:-1]
        if adjacent.size == 0:
            max_adjacent_ratio = 1.0
        else:
            max_adjacent_ratio = float(max(adjacent.max(), (1.0 / adjacent).max()))
        stats.append(
            {
                "segment": i,
                "n_cell": int(widths.size),
                "min_width": float(widths.min()),
                "max_width": float(widths.max()),
                "max_adjacent_ratio": max_adjacent_ratio,
            }
        )
    return stats


def face_normal_width(block: dict, face: str) -> np.ndarray:
    x = block["x"][:, :, 0]
    y = block["y"][:, :, 0]
    if face == "imin":
        return np.abs(x[1, :] - x[0, :])
    if face == "imax":
        return np.abs(x[-1, :] - x[-2, :])
    if face == "jmin":
        return np.abs(y[:, 1] - y[:, 0])
    if face == "jmax":
        return np.abs(y[:, -1] - y[:, -2])
    raise ValueError(face)


def face_tangent_width(block: dict, face: str) -> np.ndarray:
    x = block["x"][:, :, 0]
    y = block["y"][:, :, 0]
    if face in {"imin", "imax"}:
        i = 0 if face == "imin" else -1
        return np.hypot(np.diff(x[i, :]), np.diff(y[i, :]))
    j = 0 if face == "jmin" else -1
    return np.hypot(np.diff(x[:, j]), np.diff(y[:, j]))


def face_nodes(block: dict, face: str) -> np.ndarray:
    x = block["x"][:, :, 0]
    y = block["y"][:, :, 0]
    if face == "imin":
        return np.column_stack((x[0, :], y[0, :]))
    if face == "imax":
        return np.column_stack((x[-1, :], y[-1, :]))
    if face == "jmin":
        return np.column_stack((x[:, 0], y[:, 0]))
    if face == "jmax":
        return np.column_stack((x[:, -1], y[:, -1]))
    raise ValueError(face)


def block_interface_stats(blocks: list[dict], lookup: dict[tuple[int, int], int]) -> list[dict]:
    stats: list[dict] = []
    for block in blocks:
        for face in ("imin", "imax", "jmin", "jmax"):
            nkey = neighbor_key(block["i_seg"], block["j_seg"], face)
            if nkey not in lookup:
                continue
            target = blocks[lookup[nkey]]
            if block["id"] > target["id"]:
                continue
            target_face = opposite_face(face)
            normal_a = face_normal_width(block, face)
            normal_b = face_normal_width(target, target_face)
            normal_ratio = np.maximum(normal_a / normal_b, normal_b / normal_a)
            tangent_a = face_tangent_width(block, face)
            tangent_b = face_tangent_width(target, target_face)
            nodes_a = face_nodes(block, face)
            nodes_b = face_nodes(target, target_face)
            stats.append(
                {
                    "block": block["id"] + 1,
                    "face": face,
                    "neighbor": target["id"] + 1,
                    "normal_ratio_max": float(normal_ratio.max()),
                    "normal_width_min": float(min(normal_a.min(), normal_b.min())),
                    "normal_width_max": float(max(normal_a.max(), normal_b.max())),
                    "tangent_width_max_abs_diff": float(np.abs(tangent_a - tangent_b).max()),
                    "node_max_abs_diff": float(np.abs(nodes_a - nodes_b).max()),
                }
            )
    return stats


def wall_spacing_stats(blocks: list[dict], lookup: dict[tuple[int, int], int]) -> dict:
    widths: list[float] = []
    for block in blocks:
        for face in ("imin", "imax", "jmin", "jmax"):
            if neighbor_key(block["i_seg"], block["j_seg"], face) in lookup:
                continue
            if classify_physical_face(block["i_seg"], block["j_seg"], face) != BC_WALL:
                continue
            widths.extend(float(value) for value in face_normal_width(block, face))
    if not widths:
        return {"count": 0, "min": 0.0, "max": 0.0, "ratio": 1.0}
    return {
        "count": len(widths),
        "min": min(widths),
        "max": max(widths),
        "ratio": max(widths) / min(widths),
    }


def write_summary(path: Path, blocks: list[dict]) -> None:
    total_cells = sum((b["nx"] - 1) * (b["ny"] - 1) for b in blocks)
    total_nodes = sum(b["nx"] * b["ny"] for b in blocks)
    lookup = {(b["i_seg"], b["j_seg"]): b["id"] for b in blocks}
    x_segments = build_x_segments()
    y_segments = build_y_segments()
    x_join = segment_join_stats(x_segments)
    y_join = segment_join_stats(y_segments)
    x_internal = segment_internal_stats(x_segments)
    y_internal = segment_internal_stats(y_segments)
    interface = block_interface_stats(blocks, lookup)
    wall_spacing = wall_spacing_stats(blocks, lookup)
    max_interface_ratio = max((entry["normal_ratio_max"] for entry in interface), default=1.0)
    max_interface_node_mismatch = max((entry["node_max_abs_diff"] for entry in interface), default=0.0)
    max_interface_tangent_mismatch = max((entry["tangent_width_max_abs_diff"] for entry in interface), default=0.0)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"Blocks: {len(blocks)}\n")
        handle.write(f"Total structured cells: {total_cells}\n")
        handle.write(f"Total grid nodes counted per block: {total_nodes}\n")
        handle.write(f"X_CELLS: {X_CELLS}\n")
        handle.write(f"Y_CELLS: {Y_CELLS}\n")
        handle.write(f"Wall cell width: {WALL_NORMAL_DN:.12e} m\n")
        handle.write(f"First-center y+: {YPLUS_TARGET:.3f}\n")
        handle.write(f"Wall-normal spacing count: {wall_spacing['count']}\n")
        handle.write(f"Wall-normal spacing min: {wall_spacing['min']:.12e} m\n")
        handle.write(f"Wall-normal spacing max: {wall_spacing['max']:.12e} m\n")
        handle.write(f"Wall-normal spacing max/min: {wall_spacing['ratio']:.12e}\n")
        handle.write(f"Max block-interface normal-width ratio: {max_interface_ratio:.12e}\n")
        handle.write(f"Max block-interface node mismatch: {max_interface_node_mismatch:.12e} m\n")
        handle.write(f"Max block-interface tangent-width mismatch: {max_interface_tangent_mismatch:.12e} m\n")
        for key, value in WALL_INFO.items():
            handle.write(f"{key}: {value:.12e}\n")
        handle.write("\nX segment joins:\n")
        for entry in x_join:
            handle.write(
                f"  join {entry['join']}-{entry['join'] + 1}: "
                f"{entry['left_width']:.12e} -> {entry['right_width']:.12e}, "
                f"ratio={entry['ratio']:.12e}\n"
            )
        handle.write("Y segment joins:\n")
        for entry in y_join:
            handle.write(
                f"  join {entry['join']}-{entry['join'] + 1}: "
                f"{entry['left_width']:.12e} -> {entry['right_width']:.12e}, "
                f"ratio={entry['ratio']:.12e}\n"
            )
        handle.write("\nX segment internal ratios:\n")
        for entry in x_internal:
            handle.write(
                f"  seg {entry['segment']}: n={entry['n_cell']}, "
                f"min={entry['min_width']:.12e}, max={entry['max_width']:.12e}, "
                f"max_adjacent_ratio={entry['max_adjacent_ratio']:.12e}\n"
            )
        handle.write("Y segment internal ratios:\n")
        for entry in y_internal:
            handle.write(
                f"  seg {entry['segment']}: n={entry['n_cell']}, "
                f"min={entry['min_width']:.12e}, max={entry['max_width']:.12e}, "
                f"max_adjacent_ratio={entry['max_adjacent_ratio']:.12e}\n"
            )
        handle.write("\nBlock interface checks:\n")
        for entry in interface:
            handle.write(
                f"  block {entry['block']} {entry['face']} -> block {entry['neighbor']}: "
                f"normal_ratio={entry['normal_ratio_max']:.12e}, "
                f"normal_width=[{entry['normal_width_min']:.12e}, {entry['normal_width_max']:.12e}], "
                f"node_mismatch={entry['node_max_abs_diff']:.12e}, "
                f"tangent_width_mismatch={entry['tangent_width_max_abs_diff']:.12e}\n"
            )


def write_preview(path: Path, blocks: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    for block in blocks:
        x = block["x"][:, :, 0]
        y = block["y"][:, :, 0]
        for i in range(block["nx"]):
            ax.plot(x[i, :], y[i, :], color="0.72", linewidth=0.25)
        for j in range(block["ny"]):
            ax.plot(x[:, j], y[:, j], color="0.72", linewidth=0.25)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_xlim(X_LINES[0], X_LINES[-1])
    ax.set_ylim(Y_LINES[0], Y_LINES[-1])
    ax.set_title("Square-cylinder multiblock mesh")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    blocks, block_lookup = build_blocks()
    write_plot3d(WORK_DIR / PLOT3D_NAME, blocks)
    write_boundary(WORK_DIR / BOUNDARY_NAME, blocks, block_lookup)
    write_su2(WORK_DIR / SU2_NAME, blocks)
    write_summary(WORK_DIR / SUMMARY_NAME, blocks)
    write_preview(WORK_DIR / PREVIEW_NAME, blocks)
    read_grid(
        gridgen_or_pointwise=0,
        dimension=2,
        grid_file_name=str(WORK_DIR / PLOT3D_NAME),
        boundary_file_name=str(WORK_DIR / BOUNDARY_NAME),
        n_proc=1,
        is_binary=True,
        write_binary=True,
        set_z=True,
        z_value=0.0,
        output_dir=str(WORK_DIR),
    )
    lookup = {(b["i_seg"], b["j_seg"]): b["id"] for b in blocks}
    interface = block_interface_stats(blocks, lookup)
    wall_spacing = wall_spacing_stats(blocks, lookup)
    print(f"Generated {len(blocks)} blocks in {WORK_DIR}")
    print(f"Total structured cells: {sum((b['nx'] - 1) * (b['ny'] - 1) for b in blocks)}")
    print(f"Wall cell width: {WALL_NORMAL_DN:.6e} m")
    print(f"First-center y+: {YPLUS_TARGET:.3f}")
    print(f"Wall-normal spacing range: {wall_spacing['min']:.6e} - {wall_spacing['max']:.6e} m")
    print(f"Max block-interface normal-width ratio: {max(entry['normal_ratio_max'] for entry in interface):.6f}")
    print(f"Preview image: {WORK_DIR / PREVIEW_NAME}")
    print(f"Summary file : {WORK_DIR / SUMMARY_NAME}")

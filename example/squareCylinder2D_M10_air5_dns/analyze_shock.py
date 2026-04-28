#!/usr/bin/env python3
"""Report centerline bow-shock metrics from the latest square-cylinder Tecplot file."""

from __future__ import annotations

import argparse
import re
import struct
from collections import defaultdict
from pathlib import Path

import numpy as np


CASE_DIR = Path(__file__).resolve().parent
BODY_LEADING_EDGE_X = 0.004
BODY_WINDOW = (0.0025, 0.00398)


def tecplot_string(data: bytes, offset: int) -> tuple[str, int]:
    chars: list[str] = []
    while True:
        value = struct.unpack_from("<i", data, offset)[0]
        offset += 4
        if value == 0:
            return "".join(chars), offset
        chars.append(chr(value))


def parse_solution(path: Path) -> tuple[list[str], list[dict]]:
    data = path.read_bytes()
    offset = 8
    byte_order, file_type = struct.unpack_from("<ii", data, offset)
    offset += 8
    if data[:8] != b"#!TDV112" or byte_order != 1 or file_type != 0:
        raise ValueError(f"{path} is not a supported Tecplot V112 full solution file")

    _, offset = tecplot_string(data, offset)
    n_var = struct.unpack_from("<i", data, offset)[0]
    offset += 4
    names = []
    for _ in range(n_var):
        name, offset = tecplot_string(data, offset)
        names.append(name)

    zones: list[dict] = []
    while True:
        marker = struct.unpack_from("<f", data, offset)[0]
        offset += 4
        if abs(marker - 357.0) < 1e-4:
            break
        if abs(marker - 299.0) > 1e-4:
            raise ValueError(f"bad zone marker {marker} at offset {offset - 4}")
        zone_name, offset = tecplot_string(data, offset)
        offset += 8
        time = struct.unpack_from("<d", data, offset)[0]
        offset += 8
        offset += 20
        mx, my, mz = struct.unpack_from("<iii", data, offset)
        offset += 12
        offset += 4
        zones.append({"name": zone_name, "mx": mx, "my": my, "mz": mz, "time": time})

    for zone in zones:
        offset += 4 + 4 * n_var + 12 + 16 * n_var
        n_point = zone["mx"] * zone["my"] * zone["mz"]
        values: dict[str, np.ndarray] = {}
        keep = {"x", "y", "pressure", "temperature", "mach", "Tve"}
        for i, name in enumerate(names):
            array = np.frombuffer(data, dtype="<f8", count=n_point, offset=offset)
            offset += 8 * n_point
            if name in keep:
                values[name] = array.reshape((zone["mx"], zone["my"], zone["mz"]), order="F").copy()
        zone["values"] = values
    return names, zones


def latest_time_series(case_dir: Path) -> Path:
    files = []
    for path in (case_dir / "output" / "time_series").glob("flowfield_*.plt"):
        if path.name == "flowfield_0.plt":
            continue
        match = re.search(r"flowfield_([0-9.+eE-]+)s\.plt$", path.name)
        if match:
            files.append((float(match.group(1)), path))
    if not files:
        raise FileNotFoundError("no time-series flowfield files found")
    return max(files, key=lambda item: item[0])[1]


def analyze(path: Path) -> dict:
    _, zones = parse_solution(path)
    centerline: dict[float, list[tuple[float, float, float, float]]] = defaultdict(list)
    max_p = 0.0
    max_t = 0.0
    max_tve = 0.0
    min_mach = float("inf")
    max_mach = 0.0
    for zone in zones:
        values = zone["values"]
        pressure = values["pressure"]
        temperature = values["temperature"]
        mach = values["mach"]
        tve = values["Tve"]
        max_p = max(max_p, float(pressure.max()))
        max_t = max(max_t, float(temperature.max()))
        max_tve = max(max_tve, float(tve.max()))
        min_mach = min(min_mach, float(mach.min()))
        max_mach = max(max_mach, float(mach.max()))
        x = values["x"][:, :, 0]
        y = values["y"][:, :, 0]
        mask = (np.abs(y) < 1.0e-10) & (x >= 0.0) & (x < BODY_LEADING_EDGE_X + 1.0e-10)
        for xi, pi, mi, ti, tve_i in zip(x[mask], pressure[:, :, 0][mask], mach[:, :, 0][mask],
                                         temperature[:, :, 0][mask], tve[:, :, 0][mask]):
            centerline[round(float(xi), 12)].append((float(pi), float(mi), float(ti), float(tve_i)))

    xs = np.array(sorted(centerline))
    p = np.array([np.median([entry[0] for entry in centerline[x]]) for x in xs])
    mach = np.array([np.median([entry[1] for entry in centerline[x]]) for x in xs])
    temperature = np.array([np.median([entry[2] for entry in centerline[x]]) for x in xs])
    tve = np.array([np.median([entry[3] for entry in centerline[x]]) for x in xs])
    gradient = np.gradient(np.log(np.maximum(p, 1.0e-30)), xs)
    window = (xs > BODY_WINDOW[0]) & (xs < BODY_WINDOW[1])
    idx = int(np.argmax(np.where(window, gradient, -np.inf)))
    lo = max(idx - 2, 0)
    hi = min(idx + 2, len(xs) - 1)
    p_jump = p[hi] / max(p[lo], 1.0e-30)
    mach_drop = mach[lo] - mach[hi]
    standoff = BODY_LEADING_EDGE_X - xs[idx]
    detected = bool(window[idx] and standoff > 2.0e-5 and p_jump > 1.25 and mach_drop > 0.3)
    return {
        "file": path,
        "time": zones[0]["time"],
        "n_zone": len(zones),
        "max_p": max_p,
        "max_t": max_t,
        "max_tve": max_tve,
        "min_mach": min_mach,
        "max_mach": max_mach,
        "shock_detected": detected,
        "shock_x": float(xs[idx]),
        "standoff": float(standoff),
        "p_jump": float(p_jump),
        "mach_drop": float(mach_drop),
        "mach_up": float(mach[lo]),
        "mach_down": float(mach[hi]),
        "p_candidate": float(p[idx]),
        "t_candidate": float(temperature[idx]),
        "tve_candidate": float(tve[idx]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("file", nargs="?", type=Path, help="Tecplot file; defaults to latest time-series output")
    args = parser.parse_args()
    path = args.file if args.file else latest_time_series(CASE_DIR)
    result = analyze(path)
    print(f"file={result['file'].name} time={result['time']:.7e} zones={result['n_zone']}")
    print(
        "global "
        f"max_p={result['max_p']:.6e} max_T={result['max_t']:.3f} "
        f"max_Tve={result['max_tve']:.3f} min_M={result['min_mach']:.6e} max_M={result['max_mach']:.3f}"
    )
    print(
        "shock "
        f"detected={result['shock_detected']} x={result['shock_x']:.9f} "
        f"standoff={result['standoff']:.9e} p_jump_4pt={result['p_jump']:.3f} "
        f"mach_drop_4pt={result['mach_drop']:.3f} "
        f"M_up~{result['mach_up']:.3f} M_down~{result['mach_down']:.3f}"
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
import contextlib
import io
from pathlib import Path

import numpy as np

from TecplotUtilsGXL import read_tecplot_plt


DEFAULT_VARIABLES = [
    "x", "y", "z", "density", "u", "v", "w", "pressure", "temperature",
    "N2", "O2", "NO", "N", "O", "Eve", "mach", "Tve",
]

KEY_VARIABLES = ["density", "pressure", "temperature", "Tve", "Eve", "N2", "O2", "NO", "N", "O"]


def read_dataset(path, variables):
    # TecplotUtilsGXL is intentionally verbose; keep validation output compact.
    with contextlib.redirect_stdout(io.StringIO()):
        return read_tecplot_plt(str(path), variables=variables)


def variable_stats(dataset):
    stats = {name: {"min": np.inf, "max": -np.inf, "nonfinite": 0, "count": 0}
             for name in dataset["variables"]}
    for zone in dataset["zones"]:
        data = zone["data"]
        for idx, name in enumerate(dataset["variables"]):
            values = data[idx].ravel()
            finite = np.isfinite(values)
            stats[name]["nonfinite"] += int(values.size - np.count_nonzero(finite))
            stats[name]["count"] += int(values.size)
            if np.any(finite):
                finite_values = values[finite]
                stats[name]["min"] = min(stats[name]["min"], float(np.min(finite_values)))
                stats[name]["max"] = max(stats[name]["max"], float(np.max(finite_values)))
    return stats


def difference_norms(left, right):
    variables = [v for v in left["variables"] if v in right["variables"]]
    right_indices = {name: idx for idx, name in enumerate(right["variables"])}
    norms = {}

    if len(left["zones"]) != len(right["zones"]):
        raise ValueError("Datasets have different zone counts")

    for name in variables:
        li = left["variables"].index(name)
        ri = right_indices[name]
        abs_sum = 0.0
        sq_sum = 0.0
        max_abs = 0.0
        count = 0
        nonfinite = 0
        for zl, zr in zip(left["zones"], right["zones"]):
            if zl["dimensions"] != zr["dimensions"]:
                raise ValueError(f"Zone dimension mismatch for {zl['name']} / {zr['name']}")
            diff = zr["data"][ri] - zl["data"][li]
            finite = np.isfinite(diff)
            nonfinite += int(diff.size - np.count_nonzero(finite))
            if np.any(finite):
                d = diff[finite]
                abs_d = np.abs(d)
                abs_sum += float(np.sum(abs_d))
                sq_sum += float(np.sum(d * d))
                max_abs = max(max_abs, float(np.max(abs_d)))
                count += int(d.size)
        norms[name] = {
            "L1_mean": abs_sum / count if count else np.nan,
            "L2_rms": np.sqrt(sq_sum / count) if count else np.nan,
            "Linf": max_abs,
            "nonfinite": nonfinite,
            "count": count,
        }
    return norms


def format_stats(title, stats, variables):
    lines = [title, "variable min max nonfinite count"]
    for name in variables:
        s = stats[name]
        lines.append(f"{name} {s['min']:.16e} {s['max']:.16e} {s['nonfinite']} {s['count']}")
    return lines


def format_norms(title, norms, variables):
    lines = [title, "variable L1_mean L2_rms Linf nonfinite count"]
    for name in variables:
        n = norms[name]
        lines.append(f"{name} {n['L1_mean']:.16e} {n['L2_rms']:.16e} {n['Linf']:.16e} {n['nonfinite']} {n['count']}")
    return lines


def main():
    parser = argparse.ArgumentParser(description="Compare two COREFL Tecplot flowfield outputs.")
    parser.add_argument("reference", type=Path, help="Reference Tecplot .plt file")
    parser.add_argument("candidate", type=Path, help="Candidate Tecplot .plt file")
    parser.add_argument("--summary", type=Path, help="Write the text summary to this file")
    parser.add_argument("--variables", nargs="*", default=DEFAULT_VARIABLES)
    args = parser.parse_args()

    left = read_dataset(args.reference, args.variables)
    right = read_dataset(args.candidate, args.variables)
    left_stats = variable_stats(left)
    right_stats = variable_stats(right)
    norms = difference_norms(left, right)

    key_vars = [name for name in KEY_VARIABLES if name in left_stats and name in right_stats]
    common_vars = [name for name in left["variables"] if name in norms]

    lines = [
        f"reference: {args.reference}",
        f"candidate: {args.candidate}",
        f"zones: {len(left['zones'])}",
        "",
    ]
    lines.extend(format_stats("reference_minmax", left_stats, key_vars))
    lines.append("")
    lines.extend(format_stats("candidate_minmax", right_stats, key_vars))
    lines.append("")
    lines.extend(format_norms("candidate_minus_reference_norms", norms, common_vars))
    text = "\n".join(lines) + "\n"

    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(text)
    print(text)


if __name__ == "__main__":
    main()

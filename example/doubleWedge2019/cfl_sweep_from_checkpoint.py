#!/usr/bin/env python3
import argparse
import datetime as dt
import os
import re
import shutil
import subprocess
from pathlib import Path


CASE_DIR = Path(__file__).resolve().parent
SETUP = CASE_DIR / "input" / "setup.txt"
OUTPUT = CASE_DIR / "output"
CHECKPOINT_TIME = 3.45e-5
CHECKPOINT_STEP = 69000
CHECKPOINT_DIR = CASE_DIR / "output_latest_valid"
CHECKPOINT_FLOW = CHECKPOINT_DIR / "time_series" / "flowfield_3.4500e-05s.plt"
SUMMARY = CASE_DIR / "cfl_sweep_summary.tsv"

BAD_PATTERNS = (
    "Nan occurred",
    "Abort(",
    "BAD TERMINATION",
    "Segmentation fault",
    "Floating point exception",
    "Error:",
)


def stamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def patch_line(text: str, pattern: str, replacement: str) -> str:
    new_text = re.sub(pattern, replacement, text, flags=re.MULTILINE)
    if new_text == text and "set_current_physical_time" in replacement:
        new_text = text.replace(
            f"real total_simulation_time = {TEST_PHYSICAL_TIME:.12e}",
            f"real total_simulation_time = {TEST_PHYSICAL_TIME:.12e}\n{replacement}",
        )
    return new_text


def prepare_case(cfl: float, steps: int, output_screen: int) -> None:
    if OUTPUT.exists():
        shutil.move(str(OUTPUT), str(CASE_DIR / f"output_before_cfl_case_{stamp()}"))
    (OUTPUT / "message").mkdir(parents=True, exist_ok=True)
    shutil.copy2(CHECKPOINT_FLOW, OUTPUT / "flowfield.plt")
    for name in ("residual_scale.txt", "reference_state.txt"):
        src = CHECKPOINT_DIR / "message" / name
        if src.exists():
            shutil.copy2(src, OUTPUT / "message" / name)
    (OUTPUT / "message" / "step.txt").write_text(str(CHECKPOINT_STEP), encoding="utf-8")

    text = SETUP.read_text(encoding="utf-8")
    replacements = {
        r"^\s*int\s+initial\s*=\s*\d+": "int initial = 1",
        r"^\s*int\s+total_step\s*=\s*\d+": f"int total_step = {steps}",
        r"^\s*int\s+output_file\s*=\s*\d+": "int output_file = 100000000",
        r"^\s*int\s+output_screen\s*=\s*\d+": f"int output_screen = {output_screen}",
        r"^\s*int\s+output_time_series\s*=\s*\d+": "int output_time_series = 0",
        r"^\s*real\s+cfl\s*=\s*[0-9.eE+-]+": f"real cfl = {cfl:.12e}",
        r"^\s*bool\s+fixed_time_step\s*=\s*[01]": "bool fixed_time_step = 0",
        r"^\s*real\s+total_simulation_time\s*=\s*[0-9.eE+-]+": f"real total_simulation_time = {TEST_PHYSICAL_TIME:.12e}",
    }
    for pattern, replacement in replacements.items():
        text = patch_line(text, pattern, replacement)
    set_time = f"real set_current_physical_time = {CHECKPOINT_TIME:.12e}"
    if re.search(r"^\s*real\s+set_current_physical_time\s*=", text, re.MULTILINE):
        text = re.sub(r"^\s*real\s+set_current_physical_time\s*=\s*[0-9.eE+-]+", set_time, text, flags=re.MULTILINE)
    else:
        text = text.replace(
            f"real total_simulation_time = {TEST_PHYSICAL_TIME:.12e}",
            f"real total_simulation_time = {TEST_PHYSICAL_TIME:.12e}\n{set_time}",
        )
    SETUP.write_text(text, encoding="utf-8")


def parse_log(path: Path) -> dict[str, str]:
    text = path.read_text(errors="ignore") if path.exists() else ""
    bad = next((p for p in BAD_PATTERNS if p in text), "")
    matches = list(re.finditer(
        r"n=\s*(\d+),\s+dt=([0-9.eE+-]+).*?Current physical\s+time is\s+([0-9.eE+-]+)s",
        text,
        re.S,
    ))
    last_step = matches[-1].group(1) if matches else "0"
    last_dt = matches[-1].group(2) if matches else "nan"
    last_time = matches[-1].group(3) if matches else "nan"
    dt_values = [float(m.group(2)) for m in matches]
    status = "ok" if "Yeah, baby, we are ok now" in text or "reaches specified total step" in text else "running_or_failed"
    if bad:
        status = "bad"
    return {
        "status": status,
        "bad": bad,
        "last_step": last_step,
        "last_dt": last_dt,
        "last_time": last_time,
        "min_printed_dt": f"{min(dt_values):.12e}" if dt_values else "nan",
        "max_printed_dt": f"{max(dt_values):.12e}" if dt_values else "nan",
    }


def run_case(cfl: float, steps: int, output_screen: int) -> dict[str, str]:
    prepare_case(cfl, steps, output_screen)
    tag = f"cfl{cfl:g}".replace(".", "p")
    log = CASE_DIR / f"run_auto_cfl_{tag}_from_t3p45e-05_steps{steps}.log"
    cmd = ["mpirun", "-np", "2", "../../corefl"]
    with log.open("w", encoding="utf-8") as f:
        result = subprocess.run(
            cmd,
            cwd=CASE_DIR,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "0,1"},
            stdout=f,
            stderr=subprocess.STDOUT,
        )
    info = parse_log(log)
    info.update({"cfl": f"{cfl:.12e}", "returncode": str(result.returncode), "log": log.name})
    dest = CASE_DIR / f"output_auto_cfl_{tag}_steps{steps}_{stamp()}"
    if OUTPUT.exists():
        shutil.move(str(OUTPUT), str(dest))
    info["output"] = dest.name
    return info


def append_summary(info: dict[str, str]) -> None:
    header = ["cfl", "status", "bad", "returncode", "last_step", "last_dt", "last_time", "min_printed_dt", "max_printed_dt", "log", "output"]
    write_header = not SUMMARY.exists()
    with SUMMARY.open("a", encoding="utf-8") as f:
        if write_header:
            f.write("\t".join(header) + "\n")
        f.write("\t".join(info.get(k, "") for k in header) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--output-screen", type=int, default=200)
    parser.add_argument("cfl", nargs="+", type=float)
    args = parser.parse_args()
    if not CHECKPOINT_FLOW.exists():
        raise SystemExit(f"missing checkpoint: {CHECKPOINT_FLOW}")
    for cfl in args.cfl:
        info = run_case(cfl, args.steps, args.output_screen)
        append_summary(info)
        print(info, flush=True)
        if info["status"] == "bad":
            break
    return 0


TEST_PHYSICAL_TIME = 1.0e-5

if __name__ == "__main__":
    raise SystemExit(main())

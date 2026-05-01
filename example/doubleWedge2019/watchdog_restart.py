#!/usr/bin/env python3
import argparse
import datetime as dt
import os
import re
import shutil
import signal
import struct
import subprocess
import time
from pathlib import Path


CASE_DIR = Path(__file__).resolve().parent
SETUP = CASE_DIR / "input" / "setup.txt"
OUTPUT = CASE_DIR / "output"
TS_DIR = OUTPUT / "time_series"
WATCHDOG_LOG = CASE_DIR / "watchdog_restart.log"
CORE_LOG = CASE_DIR / "run_high_enthalpy_A8_yplus08_watchdog_current.log"

TARGET_TIME = 2.4e-4
TS_INTERVAL = 5.0e-7
DT_FALLBACKS = [5.0e-10, 4.0e-10, 3.125e-10, 2.5e-10, 2.0e-10, 1.25e-10, 1.0e-10]
BAD_PATTERNS = (
    "Nan occurred",
    "Abort(",
    "BAD TERMINATION",
    "Segmentation fault",
    "Floating point exception",
    "Error:",
)


def log(message: str) -> None:
    stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    with WATCHDOG_LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(cmd, check=False):
    return subprocess.run(cmd, cwd=CASE_DIR, text=True, capture_output=True, check=check)


def current_corefl_pids() -> list[int]:
    result = run(["pgrep", "-f", "mpirun -np 2 ../../corefl"])
    pids: list[int] = []
    for token in result.stdout.split():
        try:
            pids.append(int(token))
        except ValueError:
            pass
    result = run(["pgrep", "-f", "../../corefl"])
    for token in result.stdout.split():
        try:
            pids.append(int(token))
        except ValueError:
            pass
    return sorted(set(pid for pid in pids if pid != os.getpid()))


def current_launcher_pids() -> list[int]:
    result = run(["pgrep", "-f", "run_high_enthalpy_A8_yplus08"])
    pids: list[int] = []
    for token in result.stdout.split():
        try:
            pid = int(token)
        except ValueError:
            continue
        if pid != os.getpid():
            pids.append(pid)
    return sorted(set(pids))


def kill_running_solver() -> None:
    pids = sorted(set(current_corefl_pids() + current_launcher_pids()))
    if not pids:
        return
    log(f"stopping solver pids={pids}")
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    time.sleep(10)
    for pid in pids:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            continue
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def read_setup() -> str:
    return SETUP.read_text(encoding="utf-8")


def setup_value(pattern: str, default: float | int) -> float | int:
    text = read_setup()
    match = re.search(pattern, text, re.MULTILINE)
    if not match:
        return default
    value = match.group(1)
    return float(value) if any(c in value.lower() for c in ".e") else int(value)


def current_dt() -> float:
    return float(setup_value(r"^\s*real\s+dt\s*=\s*([0-9.eE+-]+)", 6.25e-10))


def parse_last_printed_step() -> tuple[int, float]:
    if not CORE_LOG.exists():
        return 0, 0.0
    text = CORE_LOG.read_text(errors="ignore")
    matches = list(re.finditer(r"n=\s*(\d+),\s+dt=([0-9.eE+-]+).*?Current physical\s+time is\s+([0-9.eE+-]+)s", text, re.S))
    if not matches:
        return 0, 0.0
    m = matches[-1]
    return int(m.group(1)), float(m.group(3))


def bad_log_detected(pos: int) -> tuple[bool, int, str]:
    if not CORE_LOG.exists():
        return False, pos, ""
    with CORE_LOG.open("r", errors="ignore") as f:
        f.seek(pos)
        chunk = f.read()
        new_pos = f.tell()
    for pat in BAD_PATTERNS:
        if pat in chunk:
            return True, new_pos, pat
    return False, new_pos, ""


def _read_i4(data: bytes, offset: int) -> tuple[int, int]:
    return struct.unpack_from("<i", data, offset)[0], offset + 4


def _read_f4(data: bytes, offset: int) -> tuple[float, int]:
    return struct.unpack_from("<f", data, offset)[0], offset + 4


def _read_f8(data: bytes, offset: int) -> tuple[float, int]:
    return struct.unpack_from("<d", data, offset)[0], offset + 8


def _read_tecplot_string(data: bytes, offset: int) -> tuple[str, int]:
    chars: list[str] = []
    while True:
        value, offset = _read_i4(data, offset)
        if value == 0:
            break
        if value < 0 or value > 0x10FFFF:
            raise ValueError(f"bad string code {value}")
        chars.append(chr(value))
        if len(chars) > 4096:
            raise ValueError("string too long")
    return "".join(chars), offset


def validate_tecplot_header(path: Path) -> tuple[bool, str]:
    try:
        data = path.read_bytes()
        if len(data) < 1024:
            return False, "file too small"
        if data[:8] != b"#!TDV112":
            return False, "bad magic"
        offset = 8
        byte_order, offset = _read_i4(data, offset)
        file_type, offset = _read_i4(data, offset)
        if byte_order != 1 or file_type != 0:
            return False, f"bad byte_order/file_type {byte_order}/{file_type}"
        _, offset = _read_tecplot_string(data, offset)
        n_var, offset = _read_i4(data, offset)
        if n_var <= 0 or n_var > 256:
            return False, f"bad n_var {n_var}"
        for _ in range(n_var):
            _, offset = _read_tecplot_string(data, offset)
        zone_count = 0
        while True:
            marker, next_offset = _read_f4(data, offset)
            if abs(marker - 357.0) < 1e-6:
                break
            if abs(marker - 299.0) > 1e-6:
                return False, f"bad zone marker {marker} at offset {offset}"
            offset = next_offset
            zone_name, offset = _read_tecplot_string(data, offset)
            if zone_name != f"zone {zone_count}":
                return False, f"bad zone name {zone_name}, expected zone {zone_count}"
            _, offset = _read_i4(data, offset)
            _, offset = _read_i4(data, offset)
            _, offset = _read_f8(data, offset)
            _, offset = _read_i4(data, offset)
            zone_type, offset = _read_i4(data, offset)
            _, offset = _read_i4(data, offset)
            _, offset = _read_i4(data, offset)
            _, offset = _read_i4(data, offset)
            imax, offset = _read_i4(data, offset)
            jmax, offset = _read_i4(data, offset)
            kmax, offset = _read_i4(data, offset)
            aux, offset = _read_i4(data, offset)
            if zone_type != 0 or imax <= 0 or jmax <= 0 or kmax <= 0 or aux != 0:
                return False, f"bad zone {zone_count} metadata"
            zone_count += 1
            if zone_count > 10000:
                return False, "too many zones"
        if zone_count == 0:
            return False, "no zones"
        return True, f"{zone_count} zones"
    except Exception as exc:
        return False, str(exc)


def simulation_finished() -> bool:
    if not CORE_LOG.exists():
        return False
    tail = CORE_LOG.read_text(errors="ignore")[-5000:]
    return "Yeah, baby, we are ok now" in tail or "reaches specified total step" in tail


def latest_timeseries() -> tuple[Path | None, float]:
    best: tuple[Path | None, float] = (None, 0.0)
    if not TS_DIR.exists():
        return best
    expected_size = 0
    template = TS_DIR / "flowfield_0.plt"
    if template.exists():
        expected_size = template.stat().st_size
    for path in TS_DIR.glob("flowfield_*s.plt"):
        match = re.search(r"flowfield_([0-9. eE+-]+)s\.plt$", path.name)
        if not match:
            continue
        try:
            t = float(match.group(1).strip())
        except ValueError:
            continue
        size = path.stat().st_size
        if expected_size and size < 0.99 * expected_size:
            continue
        valid, _ = validate_tecplot_header(path)
        if t > best[1] and size > 100_000_000 and valid:
            best = (path, t)
    return best


def latest_timeseries_problem() -> tuple[bool, str]:
    if not TS_DIR.exists():
        return False, ""
    template = TS_DIR / "flowfield_0.plt"
    expected_size = template.stat().st_size if template.exists() else 0
    newest: tuple[Path | None, float] = (None, 0.0)
    for path in TS_DIR.glob("flowfield_*s.plt"):
        match = re.search(r"flowfield_([0-9. eE+-]+)s\.plt$", path.name)
        if not match:
            continue
        try:
            t = float(match.group(1).strip())
        except ValueError:
            continue
        if expected_size and path.stat().st_size < 0.99 * expected_size:
            continue
        if t > newest[1]:
            newest = (path, t)
    path, _ = newest
    if path is None:
        return False, ""
    valid, reason = validate_tecplot_header(path)
    if not valid:
        return True, f"{path.name}: {reason}"
    return False, ""


def next_dt(old_dt: float) -> float:
    candidates = [dt for dt in DT_FALLBACKS if dt < old_dt * 0.999]
    if candidates:
        return candidates[0]
    return old_dt * 0.8


def steps_for(dt_value: float, physical_time: float) -> tuple[int, int, int, float]:
    restart_step = int(round(physical_time / dt_value))
    ts_steps = max(1, int(round(TS_INTERVAL / dt_value)))
    remaining_time = max(0.0, TARGET_TIME - physical_time)
    remaining_steps = int(round(remaining_time / dt_value))
    target_step = restart_step + remaining_steps
    return restart_step, ts_steps, remaining_steps, target_step


def patch_setup(dt_value: float, restart_time: float, restart_step: int, ts_steps: int, remaining_steps: int,
                target_step: int) -> None:
    text = read_setup()
    remaining_time = max(0.0, TARGET_TIME - restart_time)
    repl = {
        r"^\s*int\s+initial\s*=\s*\d+": "int initial = 1",
        r"^\s*int\s+total_step\s*=\s*\d+": f"int total_step = {remaining_steps}",
        r"^\s*int\s+output_file\s*=\s*\d+": f"int output_file = {target_step}",
        r"^\s*int\s+output_screen\s*=\s*\d+": "int output_screen = 200",
        r"^\s*int\s+output_time_series\s*=\s*\d+": f"int output_time_series = {ts_steps}",
        r"^\s*real\s+dt\s*=\s*[0-9.eE+-]+": f"real dt = {dt_value:.12e}",
        r"^\s*real\s+total_simulation_time\s*=\s*[0-9.eE+-]+": f"real total_simulation_time = {remaining_time:.12e}",
    }
    for pattern, value in repl.items():
        text = re.sub(pattern, value, text, flags=re.MULTILINE)
    line = f"real set_current_physical_time = {restart_time:.12e}"
    if re.search(r"^\s*real\s+set_current_physical_time\s*=", text, re.MULTILINE):
        text = re.sub(r"^\s*real\s+set_current_physical_time\s*=\s*[0-9.eE+-]+", line, text, flags=re.MULTILINE)
    else:
        text = text.replace(
            f"real total_simulation_time = {remaining_time:.12e}",
            f"real total_simulation_time = {remaining_time:.12e}\n{line}",
        )
    SETUP.write_text(text, encoding="utf-8")
    (OUTPUT / "message").mkdir(parents=True, exist_ok=True)
    (OUTPUT / "message" / "step.txt").write_text(str(restart_step), encoding="utf-8")


def backup_output(reason: str) -> Path:
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = CASE_DIR / f"output_watchdog_{reason}_{stamp}"
    if OUTPUT.exists():
        shutil.move(str(OUTPUT), str(dst))
    return dst


def prepare_restart(old_dt: float) -> bool:
    ts_file, ts_time = latest_timeseries()
    if ts_file is None:
        log("no completed timeseries yet; lowering dt and restarting from current output/flowfield.plt")
        source = OUTPUT / "flowfield.plt"
        restart_time = 0.0
    else:
        log(f"latest completed timeseries: {ts_file.name}, t={ts_time:.12e}s")
        source = ts_file
        restart_time = ts_time

    if not source.exists():
        log(f"restart source missing: {source}")
        return False

    tmp_restart = CASE_DIR / "restart_flowfield_from_watchdog.plt"
    shutil.copy2(source, tmp_restart)

    old_output = backup_output("failed")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "message").mkdir(parents=True, exist_ok=True)
    shutil.copy2(old_output / "message" / "residual_scale.txt", OUTPUT / "message" / "residual_scale.txt")
    if (old_output / "message" / "reference_state.txt").exists():
        shutil.copy2(old_output / "message" / "reference_state.txt", OUTPUT / "message" / "reference_state.txt")
    shutil.move(str(tmp_restart), OUTPUT / "flowfield.plt")

    new_dt = next_dt(old_dt)
    restart_step, ts_steps, remaining_steps, target_step = steps_for(new_dt, restart_time)
    patch_setup(new_dt, restart_time, restart_step, ts_steps, remaining_steps, target_step)
    log(
        "restart prepared: "
        f"dt {old_dt:.12e} -> {new_dt:.12e}, t0={restart_time:.12e}, "
        f"step={restart_step}, output_time_series={ts_steps}, "
        f"remaining_steps={remaining_steps}, target_step={target_step}"
    )
    return True


def start_solver() -> None:
    if CORE_LOG.exists():
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        CORE_LOG.rename(CASE_DIR / f"{CORE_LOG.stem}_{stamp}.log")
    cmd = (
        "cd /home/adminqwq/桌面/COREFL-CPC-main/example/doubleWedge2019 && "
        "CUDA_VISIBLE_DEVICES=0,1 mpirun -np 2 ../../corefl "
        "> run_high_enthalpy_A8_yplus08_watchdog_current.log 2>&1"
    )
    subprocess.Popen(["setsid", "bash", "-lc", cmd], cwd=CASE_DIR, stdin=subprocess.DEVNULL,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(5)
    log(f"solver started; pids={current_corefl_pids()}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poll", type=int, default=60)
    args = parser.parse_args()

    log("watchdog started")
    if not current_corefl_pids() and (not CORE_LOG.exists() or CORE_LOG.stat().st_size == 0):
        start_solver()
    pos = 0
    restart_count = 0
    while True:
        if CORE_LOG.exists():
            try:
                size = CORE_LOG.stat().st_size
                if pos > size:
                    pos = 0
            except FileNotFoundError:
                size = 0
        bad, pos, reason = bad_log_detected(pos)
        bad_ts, ts_reason = latest_timeseries_problem()
        if bad_ts:
            bad = True
            reason = f"bad timeseries header: {ts_reason}"
        pids = current_corefl_pids()
        if simulation_finished():
            log("simulation finished; watchdog exiting")
            return 0
        if bad or (CORE_LOG.exists() and not pids and CORE_LOG.stat().st_size > 0):
            last_step, last_time = parse_last_printed_step()
            log(f"failure detected reason={reason or 'process exited'}, last_printed_step={last_step}, last_time={last_time:.12e}")
            kill_running_solver()
            old_dt = current_dt()
            if not prepare_restart(old_dt):
                log("restart preparation failed; watchdog exiting")
                return 2
            restart_count += 1
            start_solver()
            pos = 0
        else:
            last_step, last_time = parse_last_printed_step()
            ts_file, ts_time = latest_timeseries()
            ts_name = ts_file.name if ts_file else "none"
            log(
                f"ok pids={pids} last_printed_step={last_step} "
                f"last_printed_time={last_time:.12e} latest_timeseries={ts_name} "
                f"latest_timeseries_time={ts_time:.12e} restarts={restart_count}"
            )
        time.sleep(args.poll)


if __name__ == "__main__":
    raise SystemExit(main())

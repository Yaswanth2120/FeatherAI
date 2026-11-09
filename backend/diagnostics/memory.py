"""macOS memory diagnostics: system state, per-process peak footprint and stage-level snapshots.

Metrics and their sources:
- peak_memory_footprint_bytes / max_rss_bytes: ``/usr/bin/time -l`` for the whole process lifetime
  (includes startup). Peak footprint is the kernel's phys_footprint high-water mark; with
  ``--load-mode none`` model weights are anonymous memory and are counted. With mmap, clean
  file-backed weight pages are not counted in footprint.
- footprint_bytes at a stage boundary: ``/usr/bin/footprint -p PID`` snapshot (not a peak).
- sampled_rss: psutil RSS samples (may miss short peaks).
- swap / pressure / vm_stat: system-wide, shared with every other process on the machine.
"""
from __future__ import annotations

import platform
import re
import subprocess
import threading
import time

import psutil

PAGE_RE = re.compile(r"page size of (\d+) bytes")


def _run(argv, timeout=10) -> str:
    try:
        return subprocess.run(argv, capture_output=True, text=True, timeout=timeout).stdout
    except (OSError, subprocess.TimeoutExpired):
        return ""


def parse_swapusage(text: str) -> dict | None:
    match = re.search(r"total = ([\d.]+)M\s+used = ([\d.]+)M\s+free = ([\d.]+)M", text)
    if not match:
        return None
    total, used, free = (int(float(v) * 1024 * 1024) for v in match.groups())
    return {"total_bytes": total, "used_bytes": used, "free_bytes": free}


def parse_vm_stat(text: str) -> dict:
    page = int(PAGE_RE.search(text).group(1)) if PAGE_RE.search(text) else 16384
    values = {}
    for line in text.splitlines()[1:]:
        key, _, value = line.partition(":")
        value = value.strip().rstrip(".")
        if value.isdigit():
            values[re.sub(r"[^a-z0-9]+", "_", key.strip().lower().replace('"', "")).strip("_")] = int(value)
    return {"page_size": page, "pages": values}


def parse_time_l(text: str) -> dict:
    """Parse BSD ``/usr/bin/time -l`` output (bytes for RSS/footprint on macOS)."""
    fields = {
        "max_rss_bytes": r"(\d+)\s+maximum resident set size",
        "peak_memory_footprint_bytes": r"(\d+)\s+peak memory footprint",
        "page_reclaims": r"(\d+)\s+page reclaims",
        "page_faults": r"(\d+)\s+page faults",
        "involuntary_context_switches": r"(\d+)\s+involuntary context switches",
        "instructions_retired": r"(\d+)\s+instructions retired",
    }
    result = {}
    for name, pattern in fields.items():
        matches = re.findall(pattern, text)
        if matches:
            result[name] = int(matches[-1])
    real = re.findall(r"([\d.]+)\s+real\s+([\d.]+)\s+user\s+([\d.]+)\s+sys", text)
    if real:
        result.update(real_seconds=float(real[-1][0]), user_seconds=float(real[-1][1]), sys_seconds=float(real[-1][2]))
    return result


def parse_footprint(text: str) -> int | None:
    match = re.search(r"Footprint:\s*([\d.]+)\s*(B|KB|MB|GB)", text)
    if not match:
        return None
    scale = {"B": 1, "KB": 1024, "MB": 1024 ** 2, "GB": 1024 ** 3}[match.group(2)]
    return int(float(match.group(1)) * scale)


def parse_therm(text: str) -> dict:
    values = dict(re.findall(r"(CPU_\w+|Scheduler_Limit|GPU_\w+)\s*=\s*(\d+)", text))
    warning = "No thermal warning level has been recorded" in text or "thermal warning level" not in text.lower()
    return {"limits": {k: int(v) for k, v in values.items()}, "no_thermal_warning_recorded": warning,
            "raw": text.strip()[:400]}


def footprint(pid: int) -> int | None:
    return parse_footprint(_run(["/usr/bin/footprint", "-p", str(pid)], timeout=20))


def system_snapshot() -> dict:
    vm = psutil.virtual_memory()
    pressure = _run(["sysctl", "-n", "kern.memorystatus_vm_pressure_level"]).strip()
    return {"time": time.time(), "physical_ram_bytes": vm.total, "available_bytes": vm.available,
            "swap": parse_swapusage(_run(["sysctl", "vm.swapusage"])),
            "pressure_level": int(pressure) if pressure.isdigit() else None,  # 1 normal, 2 warning, 4 critical
            "vm_stat": parse_vm_stat(_run(["vm_stat"])),
            "thermal": parse_therm(_run(["pmset", "-g", "therm"])),
            "os": platform.platform()}


def delta(before: dict, after: dict) -> dict:
    pages_before, pages_after = before["vm_stat"]["pages"], after["vm_stat"]["pages"]
    page = after["vm_stat"]["page_size"]
    swap_before = (before.get("swap") or {}).get("used_bytes")
    swap_after = (after.get("swap") or {}).get("used_bytes")
    return {"seconds": after["time"] - before["time"],
            "swap_used_delta_bytes": swap_after - swap_before if None not in (swap_before, swap_after) else None,
            "pageins": pages_after.get("pageins", 0) - pages_before.get("pageins", 0),
            "pageouts": pages_after.get("pageouts", 0) - pages_before.get("pageouts", 0),
            "swapins": pages_after.get("swapins", 0) - pages_before.get("swapins", 0),
            "swapouts": pages_after.get("swapouts", 0) - pages_before.get("swapouts", 0),
            "page_size": page,
            "max_pressure_level": max(filter(None, [before.get("pressure_level"), after.get("pressure_level")]), default=None)}


class RssSampler:
    """Background RSS sampling of a process tree, with named phase markers."""

    def __init__(self, pid: int, interval: float = .1):
        self.pid, self.interval = pid, interval
        self.samples: list[tuple[float, int]] = []
        self.marks: list[tuple[float, str]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _sample(self):
        try:
            process = psutil.Process(self.pid)
            total = process.memory_info().rss
            for child in process.children(recursive=True):
                try:
                    total += child.memory_info().rss
                except psutil.Error:
                    pass
            self.samples.append((time.time(), total))
        except psutil.Error:
            pass

    def _loop(self):
        while not self._stop.wait(self.interval):
            self._sample()

    def start(self):
        self._sample()
        self._thread.start()
        return self

    def mark(self, name: str):
        self.marks.append((time.time(), name))

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2)

    def phase_peaks(self) -> dict:
        """Peak sampled RSS between consecutive marks (phase named by its starting mark)."""
        bounds = self.marks + [(float("inf"), "end")]
        peaks = {}
        start_name, start = "startup", 0.0
        for when, name in bounds:
            values = [rss for t, rss in self.samples if start <= t < when]
            if values:
                peaks[start_name] = max(values)
            start_name, start = name, when
        return peaks


def measure_command(argv: list[str], log_path, sample_interval: float = .1, timeout: float = 3600) -> dict:
    """Run a command under ``/usr/bin/time -l`` and return peak memory plus system deltas."""
    before = system_snapshot()
    started = time.monotonic()
    with open(log_path, "w") as log:
        process = subprocess.Popen(["/usr/bin/time", "-l", *argv], stdout=log, stderr=subprocess.STDOUT)
        sampler = RssSampler(process.pid, sample_interval).start()
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            for child in psutil.Process(process.pid).children(recursive=True):
                child.kill()
            process.kill()
            process.wait()
            raise
        finally:
            sampler.stop()
    after = system_snapshot()
    text = open(log_path, errors="replace").read()
    return {"argv": argv, "exit_code": code, "wall_seconds": time.monotonic() - started,
            "time_l": parse_time_l(text), "sampled_peak_rss_bytes": max((r for _, r in sampler.samples), default=None),
            "rss_samples": len(sampler.samples), "system_before": before, "system_after": after,
            "system_delta": delta(before, after)}

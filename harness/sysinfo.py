"""System probes and their parsers.

Every parser takes raw command output (so it can be unit-tested against
fixtures captured on this machine) and every probe stores the raw text
alongside the parsed value in the result JSON.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import platform
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def sh(*cmd: str, check: bool = True) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, check=check).stdout


# ---------------------------------------------------------------- swap / vm
def parse_swapusage(text: str) -> dict:
    """`sysctl vm.swapusage` -> MB floats."""
    out = {}
    for key in ("total", "used", "free"):
        m = re.search(rf"{key} = ([\d.]+)M", text)
        if not m:
            raise ValueError(f"cannot parse swapusage: {text!r}")
        out[f"{key}_mb"] = float(m.group(1))
    return out


def parse_vm_stat(text: str) -> dict:
    """`vm_stat` -> {'page_size': int, '<snake_key>': int, ...}."""
    m = re.search(r"page size of (\d+) bytes", text)
    if not m:
        raise ValueError("vm_stat: no page size line")
    out = {"page_size": int(m.group(1))}
    for line in text.splitlines()[1:]:
        if ":" not in line:
            continue
        k, v = line.rsplit(":", 1)
        k = re.sub(r"[^a-z0-9]+", "_", k.strip().strip('"').lower()).strip("_")
        v = v.strip().rstrip(".")
        if v.isdigit():
            out[k] = int(v)
    for need in ("pageins", "pageouts", "swapins", "swapouts", "pages_free"):
        if need not in out:
            raise ValueError(f"vm_stat: missing {need}")
    return out


def available_bytes(vm: dict) -> int:
    """Pages the kernel can hand out without swapping or compressing:
    free + inactive + speculative + purgeable. A conservative proxy; the
    formula is recorded in every result so it can be audited."""
    pages = sum(vm.get(k, 0) for k in
                ("pages_free", "pages_inactive", "pages_speculative", "pages_purgeable"))
    return pages * vm["page_size"]


AVAILABLE_FORMULA = "(pages_free + pages_inactive + pages_speculative + pages_purgeable) * page_size"


# ---------------------------------------------------------------- power / thermal
_THERM_OK_LINE = re.compile(r"^Note: No .* has been recorded$")


def parse_pmset_therm(text: str) -> dict:
    """Fail-closed: OK only if every non-blank line is a
    'Note: No ... has been recorded' line (the only form observed on this
    machine with no thermal pressure). Anything else -> not OK."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    unexpected = [l for l in lines if not _THERM_OK_LINE.match(l)]
    return {"ok": bool(lines) and not unexpected, "unexpected_lines": unexpected}


def parse_pmset_batt(text: str) -> dict:
    return {"ac_power": "'AC Power'" in text,
            "battery_power": "'Battery Power'" in text}


def parse_lowpowermode(text: str) -> int | None:
    m = re.search(r"^\s*lowpowermode\s+(\d+)", text, re.M)
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------- /usr/bin/time -l
_TIME_FIELDS = {
    "maximum resident set size": "max_rss_bytes",
    "peak memory footprint": "peak_footprint_bytes",
    "page reclaims": "page_reclaims",
    "page faults": "page_faults",
    "swaps": "swaps",
    "block input operations": "block_input_ops",
    "involuntary context switches": "involuntary_csw",
    "voluntary context switches": "voluntary_csw",
}


def parse_time_l(text: str) -> dict:
    """Parse BSD `/usr/bin/time -l` output (it may be mixed with other
    stderr; only the final block is used)."""
    out: dict = {}
    for line in text.splitlines():
        m = re.match(r"^\s*([\d.]+) real\s+([\d.]+) user\s+([\d.]+) sys\s*$", line)
        if m:
            out = {"real_s": float(m.group(1)), "user_s": float(m.group(2)),
                   "sys_s": float(m.group(3))}
            continue
        m = re.match(r"^\s*(\d+)\s+(.+?)\s*$", line)
        if m and out and m.group(2) in _TIME_FIELDS:
            out[_TIME_FIELDS[m.group(2)]] = int(m.group(1))
    if "peak_footprint_bytes" not in out:
        raise ValueError("no /usr/bin/time -l block found")
    return out


# ---------------------------------------------------------------- snapshots
def snapshot() -> dict:
    """Point-in-time system state (raw + parsed)."""
    vm_raw = sh("vm_stat")
    swap_raw = sh("sysctl", "vm.swapusage")
    therm_raw = sh("pmset", "-g", "therm")
    batt_raw = sh("pmset", "-g", "batt")
    pm_raw = sh("pmset", "-g")
    vm = parse_vm_stat(vm_raw)
    return {
        "vm_stat": vm,
        "available_bytes": available_bytes(vm),
        "swap": parse_swapusage(swap_raw),
        "therm": parse_pmset_therm(therm_raw) | {"raw": therm_raw},
        "power": parse_pmset_batt(batt_raw) | {"raw": batt_raw.strip()},
        "lowpowermode": parse_lowpowermode(pm_raw),
    }


def blocking_processes() -> list[str]:
    """Editors/agents that must not run during measurement."""
    found = []
    for label, pattern in (("VS Code", "Visual Studio Code.app"),
                           ("Claude Code", "anthropic.claude-code")):
        if subprocess.run(["pgrep", "-f", pattern], capture_output=True).returncode == 0:
            found.append(label)
    return found


# ---------------------------------------------------------------- fingerprint
def sha256_file(path: Path, bufsize: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(bufsize):
            h.update(chunk)
    return h.hexdigest()


def _git(*args: str, cwd: Path = ROOT) -> str | None:
    r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def _pkg(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def boot_time() -> str:
    return sh("sysctl", "-n", "kern.boottime").strip()


def environment() -> dict:
    llama_src = ROOT / "third_party" / "llama.cpp"
    lock = ROOT / "uv.lock"
    thr = ROOT / "configs" / "thresholds.yaml"
    return {
        "macos_version": sh("sw_vers", "-productVersion").strip(),
        "macos_build": sh("sw_vers", "-buildVersion").strip(),
        "chip": sh("sysctl", "-n", "machdep.cpu.brand_string").strip(),
        "ram_bytes": int(sh("sysctl", "-n", "hw.memsize")),
        "boot_time": boot_time(),
        "llama_cpp_commit": _git("rev-parse", "HEAD", cwd=llama_src),
        "mlx_version": _pkg("mlx"),
        "mlx_lm_version": _pkg("mlx-lm"),
        "python_version": platform.python_version(),
        "uv_lock_sha256": sha256_file(lock) if lock.exists() else None,
        "thresholds_sha256": sha256_file(thr) if thr.exists() else None,
        "prereg_tag_commit": _git("rev-list", "-n", "1", "prereg-v1"),
        "harness_commit": _git("rev-parse", "HEAD"),
        "harness_dirty": bool(_git("status", "--porcelain")),
    }

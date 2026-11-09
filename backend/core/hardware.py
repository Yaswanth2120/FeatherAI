import platform
import subprocess
import psutil


def inspect():
    chip = platform.processor() or platform.machine()
    if platform.system() == "Darwin":
        chip = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    return {"chip": chip, "architecture": platform.machine(), "os": platform.platform(),
            "python": platform.python_version(), "physical_ram_bytes": psutil.virtual_memory().total,
            "available_ram_bytes": psutil.virtual_memory().available,
            "system_swap": psutil.swap_memory()._asdict()}

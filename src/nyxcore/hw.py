"""Apple-silicon GPU telemetry without admin rights.

macOS publishes the GPU driver's own counters in the IORegistry. Reading them needs
no sudo, unlike ``powermetrics``. Utilisation and memory are available this way;
power draw and per-core activity are not.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path

_FIELDS = {"Device Utilization %": "gpu_util", "Renderer Utilization %": "renderer_util", "Tiler Utilization %": "tiler_util",
           "In use system memory": "gpu_mem_bytes", "Alloc system memory": "gpu_alloc_bytes"}


def sample() -> dict[str, float] | None:
    """One reading of the GPU counters, or None when this machine does not expose them."""
    try:
        raw = subprocess.run(["ioreg", "-r", "-d", "1", "-w", "0", "-c", "AGXAccelerator"], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    block = re.search(r'"PerformanceStatistics" = \{([^}]*)\}', raw)
    if not block:
        return None
    values = dict(re.findall(r'"([^"]+)"=(\d+)', block.group(1)))
    return {"t": round(time.time(), 2), **{name: int(values[key]) for key, name in _FIELDS.items() if key in values}}


def watch(out: str | Path, interval: float = 1.0, while_running: str | None = None) -> int:
    """Append one reading per interval to a JSONL file; stop when no process matches ``while_running``."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out.open("a", encoding="utf-8") as fh:
        while True:
            reading = sample()
            if reading:
                fh.write(json.dumps(reading) + "\n")
                fh.flush()
                count += 1
            if while_running:
                found = subprocess.run(["pgrep", "-f", while_running], capture_output=True, text=True).stdout.split()
                if not [pid for pid in found if pid != str(os.getpid())]:  # our own command line can contain the pattern
                    return count
            time.sleep(interval)

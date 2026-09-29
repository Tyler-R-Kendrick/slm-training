"""Host process identity bound to boot, PID and kernel start ticks."""

import os
from pathlib import Path


def process_identity(pid: int | None = None) -> str:
    """Bind a PID to this boot and kernel start ticks (Linux/WSL only)."""
    pid = os.getpid() if pid is None else pid
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    except (FileNotFoundError, ProcessLookupError):
        return ""
    return f"{boot}:{pid}:{fields[19]}"

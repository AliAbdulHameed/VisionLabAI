"""How much CPU this process may really use.

Cloud hosts usually cap a container's CPU time with a quota while still
showing it every core on the machine. Sizing thread pools from the visible
core count makes them burn through the quota in a burst and then get frozen
for the rest of the scheduling period, which shows up as stutter.
"""

from __future__ import annotations

import math
import os
from pathlib import Path


def cgroup_cpu_quota(root: str = "/sys/fs/cgroup", proc_cgroup: str = "/proc/self/cgroup") -> float | None:
    """CPUs this container may use according to its cgroup quota, or None if unlimited.

    Cloud hosts usually cap a container's CPU time with a quota while still
    showing it every core on the machine. Libraries then start one thread per
    visible core, burn through the quota in a burst and get frozen for the rest
    of the scheduling period, which shows up as stutter.
    """
    candidates = [Path(root, "cpu.max")]  # cgroup v2: "<quota> <period>" or "max <period>"
    try:
        for line in Path(proc_cgroup).read_text().splitlines():  # cgroup v1: "N:cpu,cpuacct:/path"
            _, controllers, path = line.split(":", 2)
            if "cpu" in controllers.split(","):
                for mount in ("cpu", "cpu,cpuacct", "cpuacct,cpu"):
                    candidates += [Path(root, mount, path.lstrip("/")), Path(root, mount)]
    except (OSError, ValueError):
        pass

    for candidate in candidates:
        try:
            if candidate.name == "cpu.max":
                quota, period = candidate.read_text().split()
                if quota != "max":
                    return int(quota) / int(period)
                return None
            quota = int((candidate / "cpu.cfs_quota_us").read_text())
            period = int((candidate / "cpu.cfs_period_us").read_text())
            return quota / period if quota > 0 else None
        except (OSError, ValueError):
            continue
    return None


def cpu_budget() -> int:
    """Whole CPUs actually available: visible cores, capped by any cgroup quota."""
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    quota = cgroup_cpu_quota()
    if quota is not None:
        cpus = min(cpus, math.floor(quota))
    return max(1, cpus)


def inference_threads(budget: int | None = None) -> int:
    """Threads for the model: the CPU budget minus one core left for video encoding."""
    budget = cpu_budget() if budget is None else budget
    return max(1, budget - 1)

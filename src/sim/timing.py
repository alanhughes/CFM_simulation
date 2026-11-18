"""Clocks and machine data for the timing block of the run manifest (FEAT-013).

A value that cannot be read is None, with a warning in the log. It never stops a run.
"""

import logging
import os
import platform
import resource
import subprocess
import sys
import time
from typing import Callable, Optional

SLURM_VARIABLES = ("SLURM_JOB_ID", "SLURM_CPUS_PER_TASK", "SLURM_JOB_PARTITION")
THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def clock() -> dict:
    """The wall clock (time.perf_counter) and the CPU time of this process (time.process_time) now, in seconds.

    The same two clocks as in the CFM (firn.step_timing_stages), so the values of the two can be subtracted.
    """
    return {"wall": time.perf_counter(), "cpu": time.process_time()}


def stage_time(start: Optional[dict], end: Optional[dict]) -> Optional[dict]:
    """{"wall_s", "cpu_s"} from the clock() value start to the clock() value end, or None if one of them is None."""
    if start is None or end is None:
        return None
    return {"wall_s": round(end["wall"] - start["wall"], 3), "cpu_s": round(end["cpu"] - start["cpu"], 3)}


def or_none(name: str, function: Callable):
    """function(), or None with a warning if it raises an exception."""
    try:
        return function()
    except Exception as error:
        logging.warning(f"The timing value {name} is not in the manifest: {error!r}")
        return None


def loadavg() -> Optional[list]:
    """The load average of the machine over 1, 5 and 15 minutes (os.getloadavg), or None."""
    return or_none("loadavg", lambda: [round(value, 2) for value in os.getloadavg()])


def cpu_time_total() -> Optional[float]:
    """The user plus system CPU time of this process in seconds (getrusage), or None. In a run it is the process of
    sim.worker, so the wrapper and the CFM are in it."""
    def read():
        usage = resource.getrusage(resource.RUSAGE_SELF)
        return round(usage.ru_utime + usage.ru_stime, 3)

    return or_none("cpu_s_total", read)


def max_rss_mb() -> Optional[float]:
    """The maximum resident set size of this process in MiB (1024 * 1024 bytes), or None. ru_maxrss is in KiB on Linux
    and in bytes on macOS."""
    def read():
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return round(value / (1024 * 1024 if sys.platform == "darwin" else 1024), 1)

    return or_none("max_rss_mb", read)


def cpu_model() -> Optional[str]:
    """The "model name" line of /proc/cpuinfo, else (macOS) "sysctl -n machdep.cpu.brand_string", else
    platform.processor(), or None if all are empty."""
    def read():
        try:
            with open("/proc/cpuinfo") as file:
                for line in file:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
        if sys.platform == "darwin":
            try:
                brand = subprocess.run(
                    ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, timeout=5
                ).stdout.strip()
                if brand:
                    return brand
            except (OSError, subprocess.SubprocessError):
                pass
        return platform.processor() or None

    return or_none("cpu_model", read)


def cpus_allowed() -> Optional[int]:
    """The number of CPUs that this process can use (os.sched_getaffinity), or None where the function does not exist
    (macOS). On a Slurm node this is the allocation, not the size of the node."""
    if not hasattr(os, "sched_getaffinity"):
        logging.warning("The timing value cpus_allowed is not in the manifest: os.sched_getaffinity does not exist")
        return None
    return or_none("cpus_allowed", lambda: len(os.sched_getaffinity(0)))


def environment(names) -> dict:
    """{name: value} of the environment variables in names that are set."""
    return {name: os.environ[name] for name in names if name in os.environ}


def machine_data() -> dict:
    """The CPU model, the CPU counts, the Slurm variables and the thread variables that are set, for the timing block."""
    return {
        "cpu_model": cpu_model(),
        "cpu_count": or_none("cpu_count", os.cpu_count),
        "cpus_allowed": cpus_allowed(),
        "slurm": environment(SLURM_VARIABLES),
        "threads": environment(THREAD_VARIABLES),
    }

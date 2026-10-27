"""What the brain costs: the memory and CPU of its processes, and the size of
what it keeps -- for the window's resources box.

CPU is in % of one core, as `top` shows it: 250% is two and a half cores
busy. Memory is what each process really holds in RAM (its resident set).
"""

import psutil


class Usage:
    """Processes by name, measured each time `measure` is called."""

    def __init__(self):
        self._processes = {}

    def measure(self, pids):
        """{name: {"mb", "cpu"}} for {name: pid}; a pid that is gone is left out.

        CPU is the average since the last call -- so the first call for a
        process reads 0.
        """
        out = {}

        for name, pid in pids.items():
            if pid is None:
                continue

            try:
                process = self._processes.get(pid)

                if process is None:
                    process = self._processes[pid] = psutil.Process(pid)

                with process.oneshot():
                    out[name] = {
                        "mb": round(process.memory_info().rss / 1e6),
                        "cpu": round(process.cpu_percent(None)),
                    }
            except psutil.Error:
                self._processes.pop(pid, None)

        return out


def megabytes(*arrays):
    """Bytes held by some NumPy arrays (None is nothing), in MB."""
    return sum(a.nbytes for a in arrays if a is not None) / 1e6

"""What running DELTA costs, sampled every 2 s on a thread of its own: the
window's own memory and CPU, the whole machine's, and the GPU's.

The brain measures itself and RTAB-Map and sends that with its state
(delta/brain/resources.py); this is the window's half. CPU is in % of one
core, as `top` shows it. The GPU's power draw is what nvidia-smi reports;
the CPU's cannot be read under WSL, so it shows as how busy it is.
"""

import os
import subprocess
import threading
import time

import psutil


class Monitor:
    def __init__(self, every=2.0):
        self.every = every
        self.window = {}           # {"mb", "cpu"}
        self.machine = {}          # {"cpu", "threads", "mb", "total_mb"}: all of WSL
        self.gpu = None            # {"use", "mb", "total_mb", "watts"}, None without nvidia-smi
        threading.Thread(target=self._run, daemon=True, name="monitor").start()

    def _run(self):
        process = psutil.Process(os.getpid())
        process.cpu_percent(None)
        psutil.cpu_percent(None)

        while True:
            time.sleep(self.every)

            with process.oneshot():
                self.window = {"mb": process.memory_info().rss / 1e6, "cpu": process.cpu_percent(None)}

            memory = psutil.virtual_memory()
            self.machine = {
                "cpu": psutil.cpu_percent(None), "threads": psutil.cpu_count(),
                "mb": (memory.total - memory.available) / 1e6, "total_mb": memory.total / 1e6,
            }
            self.gpu = self._gpu()

    @staticmethod
    def _gpu():
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            ).stdout.splitlines()[0]
            use, used, total, watts = (field.strip() for field in out.split(","))
        except (OSError, IndexError, ValueError, subprocess.SubprocessError):
            return None

        def number(text):
            try:
                return float(text)
            except ValueError:            # "[N/A]" on some drivers
                return None

        return {"use": number(use), "mb": number(used), "total_mb": number(total), "watts": number(watts)}

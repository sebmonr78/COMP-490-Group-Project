"""Work done on a thread of its own, so the window's loop never waits for it.

The game-engine way of keeping frames even: whatever is slow and not needed
this very frame -- the map's layers redrawn after the brain sends news -- is
handed to a worker, and the window keeps showing the last finished result
until the next one is ready. Asked again before a job is done, only the
newest request is done next. NumPy and OpenCV let go of Python's lock for
their heavy parts, so the work really does run beside the loop.
"""

import threading
import traceback


class Background:
    def __init__(self, name):
        self._job = None
        self._done = None
        self._wake = threading.Condition()
        threading.Thread(target=self._run, daemon=True, name=name).start()

    def ask(self, work, *args):
        """Run work(*args) when the worker is free; replaces any job not
        yet started."""
        with self._wake:
            self._job = (work, args)
            self._wake.notify()

    def latest(self):
        """The newest finished result, or None before the first."""
        return self._done

    def _run(self):
        while True:
            with self._wake:
                while self._job is None:
                    self._wake.wait()

                (work, args), self._job = self._job, None

            try:
                self._done = work(*args)
            except Exception:          # a bad frame of work must not end the worker
                traceback.print_exc()

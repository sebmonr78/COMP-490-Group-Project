"""The window's end of the cable to the brain (the protocol is delta/wire.py).

Exactly what will connect the real drone to the laptop: the window sends the
sensors' readings, the brain sends back its state -- where it thinks it is,
its maps, its replica.
"""

import queue
import threading
import time

from delta import wire


class BrainLink:
    """The connection, from the sim's side. Receiving runs on its own thread;
    Habitat is only ever touched from the main one."""

    def __init__(self, host, port, hello):
        self.host, self.port, self.hello = host, port, hello
        self.link = None
        self.ready = False
        self.state = None
        self.state_arrays = {}
        self.state_at = None           # when the last state arrived
        self.replica = None            # the brain's 3D replica, when sent
        self.said = queue.Queue()
        self.error = None
        self.closed = False

        threading.Thread(target=self._run, daemon=True, name="brain-link").start()

    def _run(self):
        try:
            self.link = wire.connect(self.host, self.port, timeout=300.0,
                                     should_stop=lambda: self.closed)
            self.link.send("hello", self.hello)

            while not self.closed:
                message = self.link.receive()

                if message.kind == "say":
                    self.said.put((message.fields["kind"], message.fields["text"]))
                elif message.kind == "status":
                    # The glasses spoke; DELTA writes a line in the log.
                    self.said.put(("system", message.fields["text"]))
                elif message.kind == "state":
                    self.state, self.state_arrays = message.fields, message.arrays
                    self.state_at = time.monotonic()
                elif message.kind == "replica":
                    # (cube indices, colours, the fields): a new tuple each
                    # time, which is also how the view knows to redraw.
                    info = dict(message.fields)
                    info["ceiling"] = message.arrays.get("ceiling")
                    self.replica = (message.arrays["index"], message.arrays["colour"], info)
                elif message.kind == "ready":
                    self.ready = True

        except (wire.Closed, OSError) as error:
            if not self.closed:
                self.error = str(error)

    @property
    def connected(self):
        return self.link is not None and self.link.open and self.error is None

    def send(self, kind, fields=None, arrays=None):
        if self.connected:
            try:
                self.link.send(kind, fields, arrays)
            except (wire.Closed, OSError) as error:
                self.error = str(error)

    def close(self):
        if self.connected:
            self.send("bye")

        self.closed = True

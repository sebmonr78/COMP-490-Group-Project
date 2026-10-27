"""Messages between the sim and the brain.

The two run in different conda envs -- Python 3.9 with Habitat, Python 3.10
with torch -- so they cannot import each other. They talk over a local TCP
socket instead, which is also how the real glasses will reach the laptop.

A message is a small JSON header plus any number of numpy arrays, sent raw:

    b"SAG1"  uint32 header length  header JSON  array bytes, in header order

No pickle. The ends run different Pythons and different numpy, pickle is not
portable between those, and it should never be read off a socket anyway.
"""

import json
import socket
import struct
import threading
import time
from dataclasses import dataclass, field

import numpy as np


MAGIC = b"SAG1"

# Fail loudly on a corrupt stream instead of allocating gigabytes.
MAX_HEADER = 1 << 22


@dataclass
class Message:
    kind: str
    fields: dict = field(default_factory=dict)
    arrays: dict = field(default_factory=dict)


class Closed(Exception):
    """The other end hung up."""


class Link:
    """One end of the conversation. Safe to send from several threads."""

    def __init__(self, sock):
        self.sock = sock
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._send_lock = threading.Lock()
        self.open = True

    def send(self, kind, fields=None, arrays=None):
        arrays = {
            name: np.ascontiguousarray(value)
            for name, value in (arrays or {}).items()
        }

        header = json.dumps(
            {
                "kind": kind,
                "fields": fields or {},
                "arrays": [
                    {
                        "name": name,
                        "dtype": value.dtype.str,
                        "shape": list(value.shape),
                    }
                    for name, value in arrays.items()
                ],
            }
        ).encode("utf-8")

        with self._send_lock:
            try:
                self.sock.sendall(MAGIC + struct.pack("<I", len(header)) + header)

                for value in arrays.values():
                    self.sock.sendall(memoryview(value).cast("B"))

            except OSError as error:
                self.open = False
                raise Closed(str(error)) from error

    def receive(self):
        """Block until a whole message has arrived."""
        prefix = self._exactly(8)

        if prefix[:4] != MAGIC:
            raise Closed("stream out of step: bad magic")

        (length,) = struct.unpack("<I", prefix[4:])

        if length > MAX_HEADER:
            raise Closed(f"header of {length} bytes is not believable")

        header = json.loads(self._exactly(length).decode("utf-8"))

        arrays = {}

        for spec in header["arrays"]:
            dtype = np.dtype(spec["dtype"])
            shape = tuple(spec["shape"])
            size = int(np.prod(shape)) * dtype.itemsize

            arrays[spec["name"]] = np.frombuffer(
                self._exactly(size), dtype=dtype
            ).reshape(shape)

        return Message(header["kind"], header["fields"], arrays)

    def _exactly(self, size):
        buffer = bytearray(size)
        view = memoryview(buffer)
        got = 0

        while got < size:
            try:
                count = self.sock.recv_into(view[got:], size - got)

            except OSError as error:
                self.open = False
                raise Closed(str(error)) from error

            if count == 0:
                self.open = False
                raise Closed("connection closed")

            got += count

        return buffer

    def close(self):
        self.open = False

        try:
            self.sock.shutdown(socket.SHUT_RDWR)

        except OSError:
            pass

        self.sock.close()


def serve(host, port):
    """Listen, wait for the one sim that will connect, and return its Link."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((host, port))
    server.listen(1)

    try:
        sock, _ = server.accept()

    finally:
        server.close()

    return Link(sock)


def connect(host, port, timeout=120.0, should_stop=None):
    """Connect to the brain, retrying while it starts up."""
    deadline = time.monotonic() + timeout

    while True:
        try:
            sock = socket.create_connection((host, port), timeout=5.0)

            # The timeout was for connecting only. Left on, every quiet
            # moment longer than it would look like a dropped connection.
            sock.settimeout(None)

            return Link(sock)

        except OSError:
            if time.monotonic() > deadline or (should_stop and should_stop()):
                raise

            time.sleep(0.5)

"""The drone's cameras, drawn by a process of their own.

Under WSL every picture Habitat draws costs ~9-13 ms of CPU, whatever its
size, and the cameras take 30 a second (left, right and, with --depth oracle,
the true depth, ten times a second). Drawn by the window, that was 13-16 ms
of every loop it had to draw the view you steer by. This process loads the
same house, holds nothing but the camera rig, and draws the pictures on
another core while the window gets on with drawing: the window sends where
the rig is, the pictures come back through shared memory.

The pictures are the same: the same house, the same cameras, the same pose.
What is not in this copy of the world is the drone itself -- its lenses sit
at its centre, inside the shell, which never shows in them anyway.
"""

import multiprocessing
from multiprocessing import shared_memory

import numpy as np

# Frames in flight at most: one, and its pictures stay put for SLOTS frames.
SLOTS = 3


def _layout(width, height, true_depth):
    """(name, shape, dtype) of each picture, in order in a slot."""
    pictures = [("left", (height, width, 3), np.uint8), ("right", (height, width, 3), np.uint8)]

    if true_depth:
        pictures.append(("true_depth", (height, width), np.float32))

    return pictures


def _slot_views(buffer, layout):
    """Each slot's pictures as NumPy arrays over the shared buffer."""
    size = sum(int(np.prod(shape)) * np.dtype(dtype).itemsize for _, shape, dtype in layout)
    slots = []

    for slot in range(SLOTS):
        offset, pictures = slot * size, {}

        for name, shape, dtype in layout:
            count = int(np.prod(shape))
            pictures[name] = np.ndarray(shape, dtype, buffer, offset)
            offset += count * np.dtype(dtype).itemsize

        slots.append(pictures)

    return size, slots


class RigProcess:
    """Start it, ask it for a frame, collect the frame a loop or two later."""

    def __init__(self, stage, true_depth):
        from delta import rig as rig_module

        width, height = rig_module.load().resolution(stage)
        self.layout = _layout(width, height, true_depth)
        size = sum(int(np.prod(shape)) * np.dtype(dtype).itemsize for _, shape, dtype in self.layout)

        self._memory = shared_memory.SharedMemory(create=True, size=size * SLOTS)
        _, self._slots = _slot_views(self._memory.buf, self.layout)

        context = multiprocessing.get_context("spawn")
        self._connection, child = context.Pipe()
        self._process = context.Process(
            target=_serve, args=(child, self._memory.name, stage, true_depth),
            daemon=True, name="delta-cameras",
        )
        self._process.start()

        self.ready = False
        self.failed = None          # why it stopped, if it did
        self.busy = None            # the frame id it is drawing

    def poll(self):
        """A finished frame as {name: picture}, or None. Also notices the
        process being ready, or having died (then `failed` says why)."""
        try:
            while self._connection.poll():
                message = self._connection.recv()

                if message == "ready":
                    self.ready = True
                elif isinstance(message, str):
                    self.failed = message
                    self.ready = False
                else:
                    frame_id, slot = message
                    self.busy = None
                    # Copied out: the slot is drawn into again SLOTS frames on.
                    return {name: picture.copy() for name, picture in self._slots[slot].items()}
        except (EOFError, OSError) as error:
            self.failed = f"the camera process stopped ({error!r})"
            self.ready = False

        return None

    def request(self, frame_id, translation, rotation_wxyz):
        """Draw the rig's pictures from here (the rig agent's pose)."""
        self.busy = frame_id
        self._connection.send((frame_id, frame_id % SLOTS, list(translation), list(rotation_wxyz)))

    def close(self):
        try:
            self._connection.send(None)
            self._process.join(timeout=10)
        except (OSError, EOFError):
            pass

        if self._process.is_alive():
            self._process.terminate()

        self._slots = None          # the views first: memory with views on it cannot close
        self._memory.close()
        self._memory.unlink()


def _serve(connection, memory_name, stage, true_depth):
    """The camera process: the house, the rig, and nothing else."""
    try:
        import magnum as mn

        from delta import rig as rig_module
        from delta.sim import SENSORS, create_simulator, load_project_config, resolve_dataset_paths

        config = load_project_config()
        _, dataset_config, scene_file, _ = resolve_dataset_paths(config)
        rig = rig_module.load()
        simulator = create_simulator(dataset_config, scene_file, rig, spectator_size=(16, 16),
                                     stage=stage, true_depth=true_depth)

        width, height = rig.resolution(stage)
        memory = shared_memory.SharedMemory(name=memory_name)
        _, slots = _slot_views(memory.buf, _layout(width, height, true_depth))

        left, right = rig.stereo_pair
        sources = {"left": rig.rgb_uuid(left), "right": rig.rgb_uuid(right), "true_depth": rig.depth_uuid(left)}
        node = simulator.agents[SENSORS].scene_node
        connection.send("ready")
    except Exception as error:      # anything at all: say so, and the window draws them itself
        connection.send(f"the camera process could not start: {error!r}")
        return

    while True:
        try:
            message = connection.recv()
        except (EOFError, OSError):
            break

        if message is None:
            break

        frame_id, slot, translation, (w, x, y, z) = message
        node.translation = mn.Vector3(*translation)
        node.rotation = mn.Quaternion(mn.Vector3(x, y, z), w)

        observations = simulator.get_sensor_observations(agent_ids=SENSORS)

        for name, picture in slots[slot].items():
            seen = observations[sources[name]]
            picture[...] = np.squeeze(seen) if name == "true_depth" else seen[..., :3]

        connection.send((frame_id, slot))

    simulator.close()
    memory.close()

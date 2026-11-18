"""Saving a finished simulation run into the database.

A DELTA run leaves a "memory" folder behind (see simulation/delta/brain/loop.py).
The map files are not in it directly but in a folder for the scene and the pose
source, for example memory/102816036/rtabmap/:

    grid.npz     the 2D map: three layers, with its resolution and origin inside
    replica.ply  the 3D replica (only written by the final save)
    meta.json    scene, pose source, saved time

(Starting a run with --fresh, which the website does, moves the earlier map to
a folder ending in .previous. Those are never saved.)

import_run() turns one of those folders into rows for one user:

    Session      one row, on the user's drone, from the run's start to its end
    File         one row each for grid.npz, replica.ply and map.png (type 'map_export')
    MapRecord    the map's resolution (metres per cell) and origin
    ChangeRecord what is different from the user's previous saved map (see below)

map.png is a small picture of the map for the dashboard, drawn here from the
"solid" layer of grid.npz (the map as you would draw it: walls, tables and
sofas solid). Drawing it is best effort: a map that can't be drawn is still saved.

The files are copied out of the run's folder first, because the website reuses
that folder for the next run. Objects are not saved here: the simulation does
not produce them yet (item 7 of its ROADMAP).

Changes are worked out here instead, by comparing the new map with the same
user's previous saved map (find_changes()). The first scan has nothing to be
compared with, so it is the baseline and gets no changes. Only cells that
both maps have seen are compared: free floor that is now solid "appeared",
solid that is now free "disappeared". Nearby cells are grouped into one change,
and specks and thin lines (a wall that moved a cell between two scans) are
ignored. Two maps are only compared when they line up, meaning the same
resolution, size and origin (the origin is where the drone started), because
otherwise every wall would look like a change. Like the picture, this is best
effort: if comparing fails, the scan is still saved, just without changes.

Everything is read with the standard library, so the app needs no numpy or
Pillow. An .npz is a zip of .npy files, and a PNG is a few small chunks.

    import import_run
    session_id = import_run.import_run(user_id, memory_folder, started_at, ended_at)
"""

import array
import ast
import shutil
import struct
import sys
import zipfile
import zlib
from datetime import datetime, timezone
from pathlib import Path

import db

MAP_FILE = "grid.npz"
REPLICA_FILE = "replica.ply"
PICTURE_FILE = "map.png"
SAVED_FOLDER = "saved_runs"      # next to the database file; not committed
CLOCK_SLACK = 2                  # seconds the map's file time may be older than the run's start

# The same thresholds the simulation uses (simulation/config/brain.json,
# occupancy.free_below and occupied_above): a cell is free below the first,
# solid above the second, and not seen yet in between.
FREE_BELOW = -0.4
OCCUPIED_ABOVE = 1.2

PICTURE_MARGIN = 10              # cells of "not seen" shown around the explored part
PICTURE_WIDTH = 700              # about this many pixels across (whole-number zoom)
NOT_SEEN, FREE, SOLID = 0, 1, 2
COLOURS = {NOT_SEEN: b"\xd9\xdb\xe0", FREE: b"\xff\xff\xff", SOLID: b"\x1f\x29\x37"}

# Comparing two maps. These are first guesses; check them against real runs.
ALIGN_TOLERANCE = 0.25           # two origins line up if they are within this part of a cell
MIN_CHANGE_CELLS = 12            # a change smaller than this (12 cells is about 17 x 17 cm) is a speck
MAX_CHANGES = 30                 # keep the biggest ones; a long list means the maps are off, not the house


def saved_runs_dir():
    """Where copies of saved maps go: a saved_runs folder next to the database file."""
    return Path(db.db_path()).resolve().parent / SAVED_FOLDER


def find_map_folder(memory_dir):
    """The folder inside a run's memory folder that holds the map.

    DELTA writes it to memory/<scene>/<pose source>/, so that is looked at
    first, but a map directly in memory/ or in memory/<scene>/ is found too.
    Folders ending in .previous (the map of the run before, kept by --fresh)
    are skipped.

    Args:
        memory_dir (Path): the run's memory folder.

    Returns:
        Path | None: the folder with the newest grid.npz in it, or None if
        there is no map anywhere in there.
    """
    memory_dir = Path(memory_dir)
    if not memory_dir.is_dir():
        return None

    def folders_in(folder):
        return [p for p in sorted(folder.iterdir())
                if p.is_dir() and not p.name.endswith(".previous")]

    candidates = [memory_dir]
    for scene in folders_in(memory_dir):
        candidates.append(scene)
        candidates.extend(folders_in(scene))

    with_a_map = [c for c in candidates if (c / MAP_FILE).is_file()]
    if not with_a_map:
        return None
    return max(with_a_map, key=lambda c: (c / MAP_FILE).stat().st_mtime)


def read_array(npz_path, name):
    """Reads one float array out of a .npz file, with no numpy needed.

    An .npz is a zip of .npy files, and an .npy is a short header followed by
    the raw numbers.

    Args:
        npz_path (Path): the .npz file.
        name (str): the array's name, e.g. "solid_layer".

    Returns:
        tuple[array.array, tuple]: the numbers, flat and row by row, and the
        array's shape, e.g. (1200, 1200). A single number has shape ().

    Raises:
        ValueError: the array is not a plain little-endian float array.
        KeyError, zipfile.BadZipFile, OSError: the name or file is not there.
    """
    with zipfile.ZipFile(npz_path) as archive:
        data = archive.read(name + ".npy")

    if data[:6] != b"\x93NUMPY":
        raise ValueError(f"{name}.npy is not a numpy array")
    if data[6] == 1:
        header_length, start = struct.unpack("<H", data[8:10])[0], 10
    else:
        header_length, start = struct.unpack("<I", data[8:12])[0], 12

    header = ast.literal_eval(data[start:start + header_length].decode("latin1").strip())
    kinds = {"<f8": "d", "<f4": "f"}
    if header["descr"] not in kinds or header["fortran_order"]:
        raise ValueError(f"{name}.npy holds {header['descr']}, not plain floats")

    values = array.array(kinds[header["descr"]])
    values.frombytes(data[start + header_length:])
    if sys.byteorder == "big":
        values.byteswap()                 # the file is little-endian
    shape = tuple(header["shape"])
    expected = 1
    for size in shape:
        expected *= size
    if len(values) != expected:
        raise ValueError(f"{name}.npy is cut short")
    return values, shape


def read_floats(npz_path, name):
    """A small float array out of a .npz as a list: one number, or an origin's two."""
    return list(read_array(npz_path, name)[0])


def png_bytes(width, height, rows):
    """A PNG file from rows of RGB bytes (8 bits a colour, no filtering).

    Args:
        width, height (int): size in pixels.
        rows (list[bytes]): `height` rows of `width * 3` bytes each.

    Returns:
        bytes: the whole .png file.
    """
    def chunk(kind, body):
        crc = zlib.crc32(kind + body) & 0xFFFFFFFF
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", crc)

    raw = b"".join(b"\x00" + row for row in rows)       # each row starts with filter type 0
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 6))
            + chunk(b"IEND", b""))


def classify_cells(values):
    """Turns a layer's numbers into NOT_SEEN, FREE or SOLID, one byte per cell.

    Args:
        values (array.array): a layer of grid.npz, flat.

    Returns:
        bytearray: the same length as `values`.
    """
    cells = bytearray(len(values))
    for index, value in enumerate(values):
        if value < FREE_BELOW:
            cells[index] = FREE
        elif value > OCCUPIED_ABOVE:
            cells[index] = SOLID
    return cells


def draw_map(grid_path):
    """A picture (PNG) of a saved map: grey not seen, white free floor, dark solid.

    Draws the "solid" layer, cropped to the part the drone explored plus a
    small margin (the grid itself covers 60 m), zoomed in by a whole number
    so it is about PICTURE_WIDTH pixels across. Up in the picture is +y.

    Args:
        grid_path (Path): a grid.npz.

    Returns:
        bytes | None: the .png file, or None if nothing was seen yet.

    Raises:
        ValueError, KeyError, zipfile.BadZipFile, OSError: the layer is missing
        or not what the simulation writes.
    """
    values, shape = read_array(grid_path, "solid_layer")
    if len(shape) != 2:
        raise ValueError("solid_layer is not a grid")
    rows, columns = shape

    cells = classify_cells(values)

    # The part that was seen
    first_row = first_column = None
    last_row = last_column = None
    for row in range(rows):
        line = cells[row * columns:(row + 1) * columns]
        if line.count(NOT_SEEN) == columns:
            continue
        seen = [i for i, cell in enumerate(line) if cell != NOT_SEEN]
        if first_row is None:
            first_row = row
        last_row = row
        first_column = seen[0] if first_column is None else min(first_column, seen[0])
        last_column = seen[-1] if last_column is None else max(last_column, seen[-1])
    if first_row is None:
        return None

    first_row, first_column = max(0, first_row - PICTURE_MARGIN), max(0, first_column - PICTURE_MARGIN)
    last_row, last_column = min(rows - 1, last_row + PICTURE_MARGIN), min(columns - 1, last_column + PICTURE_MARGIN)
    width, height = last_column - first_column + 1, last_row - first_row + 1
    zoom = max(1, PICTURE_WIDTH // max(width, height))

    picture_rows = []
    for row in range(last_row, first_row - 1, -1):      # +y up: the highest row is drawn first
        line = b"".join(COLOURS[cell] * zoom
                        for cell in cells[row * columns + first_column:row * columns + last_column + 1])
        picture_rows.extend([line] * zoom)
    return png_bytes(width * zoom, height * zoom, picture_rows)


def maps_line_up(first, second):
    """True if two maps use the same cells, so their cells can be compared one to one.

    Args:
        first, second (tuple): (resolution, origin_x, origin_y) of each map,
            as in MapRecord.

    Returns:
        bool: same resolution, and origins no more than ALIGN_TOLERANCE of a
        cell apart.
    """
    (resolution, first_x, first_y), (other, second_x, second_y) = first, second
    if abs(resolution - other) > 1e-9:
        return False
    limit = ALIGN_TOLERANCE * resolution
    return abs(first_x - second_x) <= limit and abs(first_y - second_y) <= limit


def clusters(cells):
    """Groups cells that touch (sideways or diagonally) into separate regions.

    Args:
        cells (set[tuple]): (row, column) of each cell.

    Returns:
        list[list[tuple]]: one list of cells per region.
    """
    left = set(cells)
    regions = []
    while left:
        start = left.pop()
        region, queue = [start], [start]
        while queue:
            row, column = queue.pop()
            for next_row in (row - 1, row, row + 1):
                for next_column in (column - 1, column, column + 1):
                    neighbour = (next_row, next_column)
                    if neighbour in left:
                        left.remove(neighbour)
                        region.append(neighbour)
                        queue.append(neighbour)
        regions.append(region)
    return regions


def has_core(region):
    """True if some cell of the region has all 8 of its neighbours in it too.

    That means the region is at least 3 x 3 cells somewhere, so a thin line
    (the edge of a wall that moved a cell between two scans) does not count.
    """
    cells = set(region)
    return any(all((row + dr, column + dc) in cells
                   for dr in (-1, 0, 1) for dc in (-1, 0, 1))
               for row, column in region)


def find_changes(old_grid, new_grid):
    """What is different between two saved maps of the same place.

    Only cells that both maps have seen are compared. A cell that was free
    and is now solid is "appeared", one that was solid and is now free is
    "disappeared". Regions that are too small or too thin are dropped.

    Args:
        old_grid, new_grid (Path): the earlier and the later grid.npz.

    Returns:
        list[dict] | None: None if the two maps do not line up (see
        maps_line_up()), otherwise the changes, biggest first, at most
        MAX_CHANGES, each with change_type ('appeared' or 'disappeared'),
        x and y (world position of the middle, in metres), and width and
        height (metres, along x and along y).

    Raises:
        ValueError, KeyError, zipfile.BadZipFile, OSError: a layer is missing
        or not what the simulation writes.
    """
    old_values, old_shape = read_array(old_grid, "solid_layer")
    new_values, new_shape = read_array(new_grid, "solid_layer")
    if len(new_shape) != 2 or old_shape != new_shape:
        return None
    resolution = read_floats(new_grid, "resolution")[0]
    origin_x, origin_y = read_floats(new_grid, "origin")[:2]
    old_resolution = read_floats(old_grid, "resolution")[0]
    old_x, old_y = read_floats(old_grid, "origin")[:2]
    if not maps_line_up((old_resolution, old_x, old_y), (resolution, origin_x, origin_y)):
        return None

    columns = new_shape[1]
    before, after = classify_cells(old_values), classify_cells(new_values)
    appeared, disappeared = set(), set()
    for index, (was, now) in enumerate(zip(before, after)):
        if was and now and was != now:
            (appeared if now == SOLID else disappeared).add(divmod(index, columns))

    changes = []
    for kind, cells in (("appeared", appeared), ("disappeared", disappeared)):
        for region in clusters(cells):
            if len(region) < MIN_CHANGE_CELLS or not has_core(region):
                continue
            rows = [row for row, _ in region]
            cols = [column for _, column in region]
            changes.append({
                "change_type": kind,
                "cells": len(region),
                "x": origin_x + (sum(cols) / len(cols) + 0.5) * resolution,
                "y": origin_y + (sum(rows) / len(rows) + 0.5) * resolution,
                "width": (max(cols) - min(cols) + 1) * resolution,
                "height": (max(rows) - min(rows) + 1) * resolution,
            })
    changes.sort(key=lambda change: -change["cells"])
    return changes[:MAX_CHANGES]


def describe_change(change):
    """A short sentence for the dashboard, like "Something appeared here, about 0.6 x 0.4 m"."""
    what = "appeared here" if change["change_type"] == "appeared" else "is gone from here"
    return f"Something {what}, about {change['width']:.1f} x {change['height']:.1f} m"


def previous_saved_scan(conn, user_id, session_id, saved_dir):
    """The user's latest earlier scan that has a saved map file.

    Args:
        conn: an open database connection.
        user_id (int): whose scans to look at (only their own drones').
        session_id (int): the new scan; only earlier ones count.
        saved_dir (Path): where saved maps are.

    Returns:
        tuple[int, Path] | None: that scan's session_id and its grid.npz, or
        None if this is the first one (the baseline).
    """
    rows = conn.execute(
        "SELECT s.session_id FROM Session s JOIN Drone d ON d.drone_id = s.drone_id "
        "WHERE d.owner_id = ? AND s.session_id < ? ORDER BY s.session_id DESC",
        (user_id, session_id),
    ).fetchall()
    for row in rows:
        path = saved_dir / f"session-{row['session_id']}" / MAP_FILE
        if path.is_file():
            return row["session_id"], path
    return None


def find_or_make_drone(conn, user_id):
    """The drone a run is saved under: the user's own, or a new simulated one.

    Args:
        conn: an open database connection.
        user_id (int): who the run belongs to.

    Returns:
        int: a drone_id. An active drone is preferred. A user with no drone
        at all gets a new active one called DELTA-SIM-U<user id>.
    """
    row = conn.execute(
        "SELECT drone_id FROM Drone WHERE owner_id = ? "
        "ORDER BY (status = 'active') DESC, drone_id LIMIT 1",
        (user_id,),
    ).fetchone()
    if row:
        return row["drone_id"]
    return conn.execute(
        "INSERT INTO Drone (serial_number, owner_id, status) VALUES (?, ?, 'active')",
        (f"DELTA-SIM-U{user_id:04d}", user_id),
    ).lastrowid


def import_run(user_id, memory_dir, started_at, ended_at, conn=None, saved_dir=None):
    """Saves one finished run's map as a new scan for a user.

    Args:
        user_id (int): the user who ran it.
        memory_dir (str | Path): the run's memory folder (the map itself is
            found inside it, see find_map_folder()).
        started_at (str): when the run started, "YYYY-MM-DD HH:MM:SS" UTC.
        ended_at (str): when it ended, same format.
        conn: an open connection to use (the tests pass one); default opens the
            database from db.connect() and closes it afterwards.
        saved_dir (Path): where to copy the files; default saved_runs_dir().

    Returns:
        int | None: the new session_id (its changes since the user's previous
        saved scan are saved too, see find_changes()), or None if there was nothing to save
        (no map in the folder, or a map older than this run: that folder is
        reused, so an old file there belongs to somebody else's run).

    Raises:
        sqlite3.Error: the database refused a row (nothing is saved then).
        OSError: a file could not be copied (nothing is saved then).
    """
    folder = find_map_folder(memory_dir)
    if folder is None:
        return None
    grid = folder / MAP_FILE

    started = datetime.strptime(started_at, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    if grid.stat().st_mtime < started.timestamp() - CLOCK_SLACK:
        return None

    resolution = read_floats(grid, "resolution")[0]
    origin_x, origin_y = read_floats(grid, "origin")[:2]

    saved_dir = Path(saved_dir) if saved_dir else saved_runs_dir()
    own_connection = conn is None
    conn = conn or db.connect()
    destination = None

    try:
        with conn:  # one transaction: all of it or none of it
            drone_id = find_or_make_drone(conn, user_id)
            session_id = conn.execute(
                "INSERT INTO Session (drone_id, started_at, ended_at) VALUES (?, ?, ?)",
                (drone_id, started_at, ended_at),
            ).lastrowid

            destination = saved_dir / f"session-{session_id}"
            destination.mkdir(parents=True, exist_ok=True)

            def add_file(name, mime_type):
                return conn.execute(
                    "INSERT INTO File (session_id, drone_id, owner_id, file_type, filename, "
                    "storage_url, mime_type, size_bytes, captured_at, uploaded_at) "
                    "VALUES (?, ?, ?, 'map_export', ?, ?, ?, ?, ?, ?)",
                    (session_id, drone_id, user_id, name,
                     f"{saved_dir.name}/session-{session_id}/{name}", mime_type,
                     (destination / name).stat().st_size, ended_at, db.now()),
                ).lastrowid

            map_file_id = None
            for name in (MAP_FILE, REPLICA_FILE):
                source = folder / name
                if not source.is_file():
                    continue
                shutil.copy2(source, destination / name)
                file_id = add_file(name, "application/octet-stream")
                if name == MAP_FILE:
                    map_file_id = file_id

            try:
                picture = draw_map(grid)
            except Exception:
                picture = None      # a map is worth saving even if it can't be drawn
            if picture:
                (destination / PICTURE_FILE).write_bytes(picture)
                add_file(PICTURE_FILE, "image/png")

            conn.execute(
                "INSERT INTO MapRecord (session_id, file_id, resolution, origin_x, origin_y) "
                "VALUES (?, ?, ?, ?, ?)",
                (session_id, map_file_id, resolution, origin_x, origin_y),
            )

            # What changed since this user's last saved scan (none for the first one)
            earlier = previous_saved_scan(conn, user_id, session_id, saved_dir)
            changes = []
            if earlier:
                try:
                    changes = find_changes(earlier[1], grid) or []
                except Exception:
                    changes = []    # a map is worth saving even if it can't be compared
            for change in changes:
                conn.execute(
                    "INSERT INTO ChangeRecord (object_id, previous_session_id, current_session_id, "
                    "change_type, x, y, z, description, detected_at) "
                    "VALUES (NULL, ?, ?, ?, ?, ?, NULL, ?, ?)",
                    (earlier[0], session_id, change["change_type"], change["x"], change["y"],
                     describe_change(change), db.now()),
                )
        return session_id
    except Exception:
        if destination is not None:
            shutil.rmtree(destination, ignore_errors=True)  # no half-saved copies
        raise
    finally:
        if own_connection:
            conn.close()

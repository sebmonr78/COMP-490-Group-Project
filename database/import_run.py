"""Saving a finished simulation run into the database.

A DELTA run leaves a "memory" folder behind (see simulation/delta/brain/loop.py):

    grid.npz     the 2D map, with its resolution and origin inside
    replica.ply  the 3D replica (only written by the final save)
    meta.json    scene, pose source, saved time

import_run() turns one of those folders into rows for one user:

    Session      one row, on the user's drone, from the run's start to its end
    File         one row each for grid.npz and replica.ply (file type 'map_export')
    MapRecord    the map's resolution (metres per cell) and origin

The files are copied out of the run's folder first, because the website reuses
that folder for the next run. Objects and changes are not saved here: the
simulation does not produce them yet (items 6 and 7 of its ROADMAP).

    import import_run
    session_id = import_run.import_run(user_id, memory_folder, started_at, ended_at)
"""

import ast
import shutil
import struct
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import db

MAP_FILE = "grid.npz"
REPLICA_FILE = "replica.ply"
SAVED_FOLDER = "saved_runs"      # next to the database file; not committed
CLOCK_SLACK = 2                  # seconds the map's file time may be older than the run's start


def saved_runs_dir():
    """Where copies of saved maps go: a saved_runs folder next to the database file."""
    return Path(db.db_path()).resolve().parent / SAVED_FOLDER


def read_floats(npz_path, name):
    """Reads one small float array out of a .npz file, with no numpy needed.

    An .npz is a zip of .npy files, and an .npy is a short header followed by
    the raw numbers.

    Args:
        npz_path (Path): the .npz file.
        name (str): the array's name, e.g. "resolution".

    Returns:
        list[float]: its values (one for a single number, two for an origin).

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
    kinds = {"<f8": ("d", 8), "<f4": ("f", 4)}
    if header["descr"] not in kinds:
        raise ValueError(f"{name}.npy holds {header['descr']}, not floats")

    code, size = kinds[header["descr"]]
    body = data[start + header_length:]
    return list(struct.unpack(f"<{len(body) // size}{code}", body))


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
        memory_dir (str | Path): the run's memory folder.
        started_at (str): when the run started, "YYYY-MM-DD HH:MM:SS" UTC.
        ended_at (str): when it ended, same format.
        conn: an open connection to use (the tests pass one); default opens the
            database from db.connect() and closes it afterwards.
        saved_dir (Path): where to copy the files; default saved_runs_dir().

    Returns:
        int | None: the new session_id, or None if there was nothing to save
        (no map in the folder, or a map older than this run: that folder is
        reused, so an old file there belongs to somebody else's run).

    Raises:
        sqlite3.Error: the database refused a row (nothing is saved then).
        OSError: a file could not be copied (nothing is saved then).
    """
    memory_dir = Path(memory_dir)
    grid = memory_dir / MAP_FILE
    if not grid.is_file():
        return None

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

            map_file_id = None
            for name in (MAP_FILE, REPLICA_FILE):
                source = memory_dir / name
                if not source.is_file():
                    continue
                shutil.copy2(source, destination / name)
                file_id = conn.execute(
                    "INSERT INTO File (session_id, drone_id, owner_id, file_type, filename, "
                    "storage_url, mime_type, size_bytes, captured_at, uploaded_at) "
                    "VALUES (?, ?, ?, 'map_export', ?, ?, 'application/octet-stream', ?, ?, ?)",
                    (session_id, drone_id, user_id, name,
                     f"{saved_dir.name}/session-{session_id}/{name}",
                     (destination / name).stat().st_size, ended_at, db.now()),
                ).lastrowid
                if name == MAP_FILE:
                    map_file_id = file_id

            conn.execute(
                "INSERT INTO MapRecord (session_id, file_id, resolution, origin_x, origin_y) "
                "VALUES (?, ?, ?, ?, ?)",
                (session_id, map_file_id, resolution, origin_x, origin_y),
            )
        return session_id
    except Exception:
        if destination is not None:
            shutil.rmtree(destination, ignore_errors=True)  # no half-saved copies
        raise
    finally:
        if own_connection:
            conn.close()

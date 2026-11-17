"""Checks for import_run.py: saving a finished run into the database.

    python test_import_run.py

Each test builds a fresh temporary database and a fake run folder, so nothing
here touches delta.db or the simulation.
"""

import os
import shutil
import sqlite3
import struct
import tempfile
import time
import unittest
import zipfile
import zlib
from pathlib import Path

import db
import import_run
import seed

# The fake map: 40 rows x 50 columns, 5 cm a cell, with an origin of (-10, -7.5).
# Rows 10..29 and columns 15..44 were explored: free floor, with a wall along
# the top row (29, the highest y) and along the left column (15).
ROWS, COLUMNS = 40, 50
FIRST_ROW, LAST_ROW, FIRST_COLUMN, LAST_COLUMN = 10, 29, 15, 44


def npy(values, shape=None, descr="<f8"):
    """The bytes of an .npy file holding floats (little-endian), like numpy writes them."""
    shape = shape if shape is not None else ((len(values),) if len(values) > 1 else ())
    shape_text = "()" if shape == () else "(" + ", ".join(str(n) for n in shape) + ("," if len(shape) == 1 else "") + ")"
    text = f"{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape_text}, }}"
    padding = (64 - (10 + len(text) + 1) % 64) % 64      # numpy pads its header to 64 bytes
    header = (text + " " * padding + "\n").encode("latin1")
    code = "d" if descr == "<f8" else "f"
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header \
        + struct.pack(f"<{len(values)}{code}", *values)


def solid_layer():
    """The fake map's solid layer, row by row: 0 not seen, -1 free, 2 solid."""
    cells = [0.0] * (ROWS * COLUMNS)
    for row in range(FIRST_ROW, LAST_ROW + 1):
        for column in range(FIRST_COLUMN, LAST_COLUMN + 1):
            cells[row * COLUMNS + column] = -1.0
    for column in range(FIRST_COLUMN, LAST_COLUMN + 1):
        cells[LAST_ROW * COLUMNS + column] = 2.0
    for row in range(FIRST_ROW, LAST_ROW + 1):
        cells[row * COLUMNS + FIRST_COLUMN] = 2.0
    return cells


def fake_memory_folder(folder, replica=True, layer="explored"):
    """What a DELTA run leaves behind. `layer`: "explored", "empty" (nothing seen) or None (no layer)."""
    folder.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(folder / "grid.npz", "w") as archive:
        archive.writestr("resolution.npy", npy([0.05]))
        archive.writestr("origin.npy", npy([-10.0, -7.5]))
        if layer == "explored":
            archive.writestr("solid_layer.npy", npy(solid_layer(), (ROWS, COLUMNS), "<f4"))
        elif layer == "empty":
            archive.writestr("solid_layer.npy", npy([0.0] * (ROWS * COLUMNS), (ROWS, COLUMNS), "<f4"))
    if replica:
        (folder / "replica.ply").write_text("ply\nend_header\n")
    (folder / "meta.json").write_text('{"scene": "demo"}')


def decode_png(data):
    """Width, height and a pixel(x, y) -> (r, g, b) for the PNGs import_run draws (8-bit RGB, no filtering)."""
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    width, height = struct.unpack(">II", data[16:24])
    position, packed = 8, b""
    while position < len(data):
        length, kind = struct.unpack(">I4s", data[position:position + 8])
        if kind == b"IDAT":
            packed += data[position + 8:position + 8 + length]
        position += 12 + length
    raw = zlib.decompress(packed)
    stride = 1 + width * 3
    rows = [raw[i * stride + 1:(i + 1) * stride] for i in range(height)]
    return width, height, lambda x, y: tuple(rows[y][x * 3:x * 3 + 3])


class SavingARun(unittest.TestCase):
    def setUp(self):
        self.folder = Path(tempfile.mkdtemp())
        self.conn = db.init_db(self.folder / "test.db")
        seed.seed(self.conn)
        self.memory = self.folder / "memory"
        self.saved = self.folder / "saved_runs"
        self.started = db.now()
        fake_memory_folder(self.memory)
        self.demo_id = self.conn.execute(
            "SELECT user_id FROM User WHERE email = 'demo@example.com'").fetchone()["user_id"]

    def tearDown(self):
        self.conn.close()
        shutil.rmtree(self.folder, ignore_errors=True)

    def save(self, user_id=None, memory=None):
        return import_run.import_run(
            user_id or self.demo_id, memory or self.memory,
            self.started, db.now(), conn=self.conn, saved_dir=self.saved)

    def count(self, table):
        return self.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

    def names(self, session_id):
        return [r["filename"] for r in self.conn.execute(
            "SELECT filename FROM File WHERE session_id = ? ORDER BY filename", (session_id,))]

    def test_reads_resolution_and_origin_without_numpy(self):
        grid = self.memory / "grid.npz"
        self.assertEqual(import_run.read_floats(grid, "resolution"), [0.05])
        self.assertEqual(import_run.read_floats(grid, "origin"), [-10.0, -7.5])

    def test_reads_a_whole_grid_without_numpy(self):
        values, shape = import_run.read_array(self.memory / "grid.npz", "solid_layer")
        self.assertEqual(shape, (ROWS, COLUMNS))
        self.assertEqual(list(values), solid_layer())

    def test_saves_a_session_three_files_and_a_map(self):
        before = self.count("Session")
        session_id = self.save()
        self.assertEqual(self.count("Session"), before + 1)

        session = self.conn.execute("SELECT * FROM Session WHERE session_id = ?", (session_id,)).fetchone()
        drone = self.conn.execute("SELECT * FROM Drone WHERE drone_id = ?", (session["drone_id"],)).fetchone()
        self.assertEqual(drone["serial_number"], "DELTA-SIM-0001")   # the demo user's own drone
        self.assertEqual(session["started_at"], self.started)

        files = self.conn.execute(
            "SELECT * FROM File WHERE session_id = ? ORDER BY filename", (session_id,)).fetchall()
        self.assertEqual([f["filename"] for f in files], ["grid.npz", "map.png", "replica.ply"])
        for f in files:
            self.assertEqual(f["file_type"], "map_export")
            self.assertEqual(f["owner_id"], self.demo_id)
            self.assertTrue((self.saved / f"session-{session_id}" / f["filename"]).is_file())
            self.assertEqual(f["storage_url"], f"saved_runs/session-{session_id}/{f['filename']}")
        self.assertEqual(files[1]["mime_type"], "image/png")

        record = self.conn.execute("SELECT * FROM MapRecord WHERE session_id = ?", (session_id,)).fetchone()
        self.assertEqual((record["resolution"], record["origin_x"], record["origin_y"]), (0.05, -10.0, -7.5))
        self.assertEqual(record["file_id"], files[0]["file_id"])      # the map file, not the picture

    def test_the_picture_shows_what_was_seen(self):
        session_id = self.save()
        width, height, pixel = decode_png((self.saved / f"session-{session_id}" / "map.png").read_bytes())

        # explored part: 20 rows x 30 columns, plus 10 cells of margin where the grid has room
        # (rows 0..39 = 40, columns 5..49 = 45), zoomed 700 // 45 = 15 times
        self.assertEqual((width, height), (45 * 15, 40 * 15))

        grey, white, dark = (0xD9, 0xDB, 0xE0), (0xFF, 0xFF, 0xFF), (0x1F, 0x29, 0x37)
        self.assertEqual(pixel(5, 5), grey)                       # corner: never seen
        self.assertEqual(pixel(375, 285), white)                  # row 20, column 30: free floor
        self.assertEqual(pixel(230, 155), dark)                   # row 29, column 20: the wall
        # up is +y: the wall (row 29, the highest y) is drawn ABOVE the floor (row 20)
        self.assertLess(155, 285)

    def test_the_saved_copy_survives_the_run_folder_being_wiped(self):
        session_id = self.save()
        shutil.rmtree(self.memory)               # the next run in that slot starts fresh
        self.assertTrue((self.saved / f"session-{session_id}" / "grid.npz").is_file())

    def test_a_user_with_no_drone_gets_a_simulated_one(self):
        self.assertEqual(self.conn.execute(
            "INSERT INTO User (username, email, password_hash) VALUES ('new', 'new@example.com', 'x')"
        ).rowcount, 1)
        new_id = self.conn.execute("SELECT user_id FROM User WHERE username = 'new'").fetchone()["user_id"]
        session_id = self.save(user_id=new_id)
        drone = self.conn.execute(
            "SELECT d.* FROM Drone d JOIN Session s ON s.drone_id = d.drone_id WHERE s.session_id = ?",
            (session_id,)).fetchone()
        self.assertEqual(drone["owner_id"], new_id)
        self.assertEqual(drone["status"], "active")
        self.assertEqual(drone["serial_number"], f"DELTA-SIM-U{new_id:04d}")

    def test_the_replica_is_optional(self):
        shutil.rmtree(self.memory)
        fake_memory_folder(self.memory, replica=False)
        self.assertEqual(self.names(self.save()), ["grid.npz", "map.png"])

    def test_a_map_that_cannot_be_drawn_is_still_saved(self):
        shutil.rmtree(self.memory)
        fake_memory_folder(self.memory, layer=None)      # no solid layer to draw
        session_id = self.save()
        self.assertEqual(self.names(session_id), ["grid.npz", "replica.ply"])
        self.assertEqual(self.count("MapRecord"), 3)     # the 2 demo scans, plus this one

    def test_a_map_where_nothing_was_seen_has_no_picture(self):
        shutil.rmtree(self.memory)
        fake_memory_folder(self.memory, layer="empty")
        self.assertEqual(self.names(self.save()), ["grid.npz", "replica.ply"])

    def test_a_run_with_no_map_saves_nothing(self):
        (self.memory / "grid.npz").unlink()
        before = self.count("Session")
        self.assertIsNone(self.save())
        self.assertEqual(self.count("Session"), before)

    def test_a_map_left_by_an_earlier_run_is_not_saved(self):
        long_ago = time.time() - 3600
        os.utime(self.memory / "grid.npz", (long_ago, long_ago))
        before = self.count("Session")
        self.assertIsNone(self.save())
        self.assertEqual(self.count("Session"), before)

    def test_a_database_error_saves_nothing(self):
        before = (self.count("Session"), self.count("Drone"), self.count("File"))
        with self.assertRaises(sqlite3.IntegrityError):
            self.save(user_id=9999)              # no such user
        self.assertEqual((self.count("Session"), self.count("Drone"), self.count("File")), before)
        self.assertFalse(self.saved.exists() and any(self.saved.iterdir()))


if __name__ == "__main__":
    unittest.main()

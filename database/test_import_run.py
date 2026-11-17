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
from pathlib import Path

import db
import import_run
import seed


def npy(values):
    """The bytes of a little-endian float64 .npy file: one number, or a list."""
    shape = "()" if len(values) == 1 else f"({len(values)},)"
    text = f"{{'descr': '<f8', 'fortran_order': False, 'shape': {shape}, }}"
    padding = (64 - (10 + len(text) + 1) % 64) % 64      # numpy pads its header to 64 bytes
    header = (text + " " * padding + "\n").encode("latin1")
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header \
        + struct.pack(f"<{len(values)}d", *values)


def fake_memory_folder(folder, replica=True):
    """What a DELTA run leaves behind: a map (0.05 m cells, origin -10 / -7.5) and a replica."""
    folder.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(folder / "grid.npz", "w") as archive:
        archive.writestr("resolution.npy", npy([0.05]))
        archive.writestr("origin.npy", npy([-10.0, -7.5]))
    if replica:
        (folder / "replica.ply").write_text("ply\nend_header\n")
    (folder / "meta.json").write_text('{"scene": "demo"}')


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

    def test_reads_resolution_and_origin_without_numpy(self):
        grid = self.memory / "grid.npz"
        self.assertEqual(import_run.read_floats(grid, "resolution"), [0.05])
        self.assertEqual(import_run.read_floats(grid, "origin"), [-10.0, -7.5])

    def test_saves_a_session_two_files_and_a_map(self):
        before = self.count("Session")
        session_id = self.save()
        self.assertEqual(self.count("Session"), before + 1)

        session = self.conn.execute("SELECT * FROM Session WHERE session_id = ?", (session_id,)).fetchone()
        drone = self.conn.execute("SELECT * FROM Drone WHERE drone_id = ?", (session["drone_id"],)).fetchone()
        self.assertEqual(drone["serial_number"], "DELTA-SIM-0001")   # the demo user's own drone
        self.assertEqual(session["started_at"], self.started)

        files = self.conn.execute(
            "SELECT * FROM File WHERE session_id = ? ORDER BY filename", (session_id,)).fetchall()
        self.assertEqual([f["filename"] for f in files], ["grid.npz", "replica.ply"])
        for f in files:
            self.assertEqual(f["file_type"], "map_export")
            self.assertEqual(f["owner_id"], self.demo_id)
            self.assertTrue((self.saved / f"session-{session_id}" / f["filename"]).is_file())
            self.assertEqual(f["storage_url"], f"saved_runs/session-{session_id}/{f['filename']}")

        record = self.conn.execute("SELECT * FROM MapRecord WHERE session_id = ?", (session_id,)).fetchone()
        self.assertEqual((record["resolution"], record["origin_x"], record["origin_y"]), (0.05, -10.0, -7.5))
        self.assertEqual(record["file_id"], files[0]["file_id"])      # the map file, not the replica

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
        session_id = self.save()
        names = [r["filename"] for r in self.conn.execute(
            "SELECT filename FROM File WHERE session_id = ?", (session_id,))]
        self.assertEqual(names, ["grid.npz"])

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

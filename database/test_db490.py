"""Checks for the DELTA database.

    python test_db.py

Every test builds a fresh in-memory database, so nothing here touches
delta.db. The ones worth reading are the constraint tests: a database that
silently accepts bad data does not fail loudly, it just fills up with it.
"""

import os
import sqlite3
import tempfile
import unittest
from pathlib import Path

import db
import seed


def fresh():
    return db.init_db(db.MEMORY)


class Structure(unittest.TestCase):
    def test_all_seven_tables_exist(self):
        conn = fresh()
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
        self.assertEqual({r["name"] for r in rows}, set(db.TABLES))

    def test_foreign_keys_are_switched_on(self):
        conn = fresh()
        self.assertEqual(conn.execute("PRAGMA foreign_keys").fetchone()[0], 1)

    def test_foreign_key_list_matches_the_design(self):
        conn = fresh()

        def targets(table):
            rows = conn.execute(f"PRAGMA foreign_key_list({table})").fetchall()
            return {(r["from"], r["table"]) for r in rows}

        self.assertEqual(targets("User"), set())
        self.assertEqual(targets("Drone"), {("owner_id", "User")})
        self.assertEqual(targets("Session"), {("drone_id", "Drone")})
        self.assertEqual(targets("File"), {
            ("session_id", "Session"), ("drone_id", "Drone"),
            ("owner_id", "User"),
        })
        self.assertEqual(targets("MapRecord"), {
            ("session_id", "Session"), ("file_id", "File"),
        })
        self.assertEqual(targets("ObjectRecord"), {("session_id", "Session")})
        self.assertEqual(targets("ChangeRecord"), {
            ("object_id", "ObjectRecord"),
            ("previous_session_id", "Session"),
            ("current_session_id", "Session"),
        })


class Constraints(unittest.TestCase):
    def setUp(self):
        self.conn = fresh()
        c = self.conn
        c.execute("INSERT INTO User (username, email, password_hash) "
                  "VALUES ('kevin', 'k@example.com', 'x')")
        c.execute("INSERT INTO Drone (serial_number, owner_id, status) "
                  "VALUES ('D1', 1, 'active')")
        c.execute("INSERT INTO Session (drone_id, started_at, ended_at) "
                  "VALUES (1, '2026-10-05 10:00:00', '2026-10-05 10:30:00')")
        c.execute("INSERT INTO Session (drone_id, started_at) "
                  "VALUES (1, '2026-10-12 10:00:00')")

    def rejects(self, sql, params=()):
        with self.assertRaises(sqlite3.IntegrityError):
            self.conn.execute(sql, params)

    def test_username_and_email_are_unique(self):
        self.rejects("INSERT INTO User (username, email, password_hash) "
                     "VALUES ('kevin', 'other@example.com', 'x')")
        self.rejects("INSERT INTO User (username, email, password_hash) "
                     "VALUES ('other', 'k@example.com', 'x')")

    def test_serial_number_is_unique(self):
        self.rejects("INSERT INTO Drone (serial_number) VALUES ('D1')")

    def test_drone_status_is_limited(self):
        self.rejects("INSERT INTO Drone (serial_number, status) "
                     "VALUES ('D2', 'flying')")

    def test_drone_defaults_to_unpaired_with_no_owner(self):
        self.conn.execute("INSERT INTO Drone (serial_number) VALUES ('D2')")
        row = self.conn.execute(
            "SELECT owner_id, status FROM Drone WHERE serial_number = 'D2'"
        ).fetchone()
        self.assertIsNone(row["owner_id"])
        self.assertEqual(row["status"], "unpaired")

    def test_drone_needs_a_real_owner(self):
        self.rejects("INSERT INTO Drone (serial_number, owner_id) "
                     "VALUES ('D2', 999)")

    def test_session_needs_a_real_drone(self):
        self.rejects("INSERT INTO Session (drone_id, started_at) "
                     "VALUES (999, '2026-10-12 10:00:00')")

    def test_session_cannot_end_before_it_starts(self):
        self.rejects("INSERT INTO Session (drone_id, started_at, ended_at) "
                     "VALUES (1, '2026-10-12 10:00:00', '2026-10-12 09:00:00')")

    def test_file_type_is_limited(self):
        self.rejects(
            "INSERT INTO File (session_id, drone_id, file_type, filename, "
            "storage_url) VALUES (1, 1, 'video', 'a.mp4', 'data/a.mp4')")

    def test_file_size_cannot_be_negative(self):
        self.rejects(
            "INSERT INTO File (session_id, drone_id, file_type, filename, "
            "storage_url, size_bytes) "
            "VALUES (1, 1, 'rgb_frame', 'a.png', 'data/a.png', -5)")

    def test_a_file_belongs_to_at_most_one_map(self):
        c = self.conn
        c.execute("INSERT INTO File (session_id, drone_id, file_type, "
                  "filename, storage_url) VALUES "
                  "(1, 1, 'map_export', 'm.npy', 'data/m.npy')")
        c.execute("INSERT INTO MapRecord (session_id, file_id) VALUES (1, 1)")
        self.rejects("INSERT INTO MapRecord (session_id, file_id) VALUES (1, 1)")

    def test_map_needs_a_real_file(self):
        self.rejects("INSERT INTO MapRecord (session_id, file_id) "
                     "VALUES (1, 999)")

    def test_confidence_must_be_between_0_and_1(self):
        for bad in (-0.1, 1.5):
            self.rejects(
                "INSERT INTO ObjectRecord (session_id, label, x, y, z, "
                "confidence, first_seen_at, last_seen_at) VALUES "
                "(1, 'chair', 0, 0, 0, ?, '2026-10-05 10:05:00', "
                "'2026-10-05 10:05:00')", (bad,))

    def test_object_cannot_be_last_seen_before_first_seen(self):
        self.rejects(
            "INSERT INTO ObjectRecord (session_id, label, x, y, z, "
            "first_seen_at, last_seen_at) VALUES "
            "(1, 'chair', 0, 0, 0, '2026-10-05 10:05:00', "
            "'2026-10-05 10:00:00')")

    def test_object_needs_a_label_and_position(self):
        self.rejects(
            "INSERT INTO ObjectRecord (session_id, x, y, z, first_seen_at, "
            "last_seen_at) VALUES (1, 0, 0, 0, '2026-10-05 10:05:00', "
            "'2026-10-05 10:05:00')")

    def test_change_type_is_limited(self):
        self.rejects(
            "INSERT INTO ChangeRecord (previous_session_id, "
            "current_session_id, change_type, detected_at) "
            "VALUES (1, 2, 'vanished', '2026-10-12 10:30:00')")

    def test_change_must_compare_two_different_sessions(self):
        self.rejects(
            "INSERT INTO ChangeRecord (previous_session_id, "
            "current_session_id, change_type, detected_at) "
            "VALUES (1, 1, 'moved', '2026-10-12 10:30:00')")

    def test_change_needs_real_sessions_and_object(self):
        self.rejects(
            "INSERT INTO ChangeRecord (previous_session_id, "
            "current_session_id, change_type, detected_at) "
            "VALUES (1, 999, 'moved', '2026-10-12 10:30:00')")
        self.rejects(
            "INSERT INTO ChangeRecord (object_id, previous_session_id, "
            "current_session_id, change_type, detected_at) "
            "VALUES (999, 1, 2, 'moved', '2026-10-12 10:30:00')")

    def test_cannot_delete_a_row_that_others_point_at(self):
        # No ON DELETE CASCADE on purpose: deleting a user should not
        # silently erase their drones and every scan those drones made.
        self.rejects("DELETE FROM User WHERE user_id = 1")
        self.rejects("DELETE FROM Drone WHERE drone_id = 1")


class SeedData(unittest.TestCase):
    def setUp(self):
        self.conn = fresh()
        seed.seed(self.conn)

    def count(self, table):
        return self.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

    def test_row_counts(self):
        self.assertEqual(self.count("User"), 1)
        self.assertEqual(self.count("Drone"), 2)
        self.assertEqual(self.count("Session"), 2)
        self.assertEqual(self.count("File"), 6)
        self.assertEqual(self.count("MapRecord"), 2)
        self.assertEqual(self.count("ObjectRecord"), 4)
        self.assertEqual(self.count("ChangeRecord"), 4)

    def test_nothing_points_at_a_missing_row(self):
        self.assertEqual(self.conn.execute("PRAGMA foreign_key_check").fetchall(), [])

    def test_one_change_of_each_kind(self):
        rows = self.conn.execute(
            "SELECT change_type, COUNT(*) AS n FROM ChangeRecord "
            "WHERE object_id IS NOT NULL GROUP BY change_type"
        ).fetchall()
        self.assertEqual({r["change_type"]: r["n"] for r in rows},
                         {"moved": 1, "appeared": 1, "disappeared": 1})

    def test_map_screen_query(self):
        """The query the front end's map view needs: latest map for a user."""
        row = self.conn.execute(
            "SELECT f.storage_url, m.resolution FROM MapRecord m "
            "JOIN File f ON f.file_id = m.file_id "
            "JOIN Session s ON s.session_id = m.session_id "
            "JOIN Drone d ON d.drone_id = s.drone_id "
            "WHERE d.owner_id = 1 ORDER BY s.started_at DESC LIMIT 1"
        ).fetchone()
        self.assertIn("2026-10-12", row["storage_url"])
        self.assertEqual(row["resolution"], 0.05)

    def test_what_changed_query(self):
        """The query for 'what changed since last time' with object labels."""
        rows = self.conn.execute(
            "SELECT o.label, c.change_type FROM ChangeRecord c "
            "JOIN ObjectRecord o ON o.object_id = c.object_id "
            "WHERE c.current_session_id = 2 ORDER BY o.label"
        ).fetchall()
        self.assertEqual([(r["label"], r["change_type"]) for r in rows],
                         [("box", "appeared"), ("chair", "moved"),
                          ("lamp", "disappeared")])

    def test_seeding_twice_fails_cleanly_and_changes_nothing(self):
        with self.assertRaises(sqlite3.IntegrityError):
            seed.seed(self.conn)
        self.assertEqual(self.count("User"), 1)
        self.assertEqual(self.count("Session"), 2)


class OnDisk(unittest.TestCase):
    def test_init_refuses_to_overwrite_and_reset_rebuilds(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "scratch.db"

            conn = db.init_db(path)
            conn.execute("INSERT INTO User (username, email, password_hash) "
                         "VALUES ('a', 'a@example.com', 'x')")
            conn.commit()
            conn.close()

            with self.assertRaises(FileExistsError):
                db.init_db(path)

            conn = db.connect(path)  # the refusal left the data alone
            self.assertEqual(
                conn.execute("SELECT COUNT(*) FROM User").fetchone()[0], 1)
            conn.close()

            conn = db.init_db(path, reset=True)
            self.assertEqual(
                conn.execute("SELECT COUNT(*) FROM User").fetchone()[0], 0)
            conn.close()

    def test_wal_mode_is_on_for_files(self):
        with tempfile.TemporaryDirectory() as folder:
            conn = db.init_db(Path(folder) / "scratch.db")
            mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
            conn.close()
            self.assertEqual(mode, "wal")

    def test_env_var_picks_the_file(self):
        with tempfile.TemporaryDirectory() as folder:
            target = str(Path(folder) / "elsewhere.db")
            os.environ[db.DB_ENV_VAR] = target
            try:
                self.assertEqual(db.db_path(), target)
            finally:
                del os.environ[db.DB_ENV_VAR]

    def test_now_matches_the_timestamp_format(self):
        self.assertRegex(db.now(), r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$")


if __name__ == "__main__":
    unittest.main(verbosity=2)

"""Sample data for development, so the front end has something to show.

    python seed.py            # adds the sample data to database/delta.db

The story it tells: one user, one drone, two scans of the same room a week
apart. Between the scans a chair was moved, a lamp was taken away and a box
showed up, so the change-detection screens have one of each kind to display.

Demo login for the login page:  demo@example.com / demo1234
(phone 5555550100 for the email and password recovery pages).
Not for production: the demo password is public.
"""

import db
import passwords

DEMO_PASSWORD = "demo1234"


def seed(conn):
    """Insert the sample rows. Raises if they are already there."""
    with conn:  # one transaction: all of it or none of it
        user_id = conn.execute(
            "INSERT INTO User (username, email, password_hash, first_name, "
            "last_name, phone_number, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("demo", "demo@example.com", passwords.hash_password(DEMO_PASSWORD),
             "Demo", "User", "5555550100", "2026-10-01 09:00:00"),
        ).lastrowid

        drone_id = conn.execute(
            "INSERT INTO Drone (serial_number, owner_id, status) "
            "VALUES (?, ?, ?)",
            ("DELTA-SIM-0001", user_id, "active"),
        ).lastrowid

        # A second drone nobody has paired yet, for the pairing screen.
        conn.execute(
            "INSERT INTO Drone (serial_number, owner_id, status) "
            "VALUES (?, NULL, 'unpaired')",
            ("DELTA-SIM-0002",),
        )

        first = _add_session(
            conn, drone_id, user_id,
            started="2026-10-05 10:00:00", ended="2026-10-05 10:20:00",
        )
        second = _add_session(
            conn, drone_id, user_id,
            started="2026-10-12 10:00:00", ended="2026-10-12 10:25:00",
        )

        # Objects seen in the first scan.
        chair = _add_object(conn, first, "chair", 2.0, 0.0, 1.5, 0.91,
                            "2026-10-05 10:04:10", "2026-10-12 10:06:30")
        lamp = _add_object(conn, first, "lamp", 4.2, 0.0, 0.8, 0.84,
                           "2026-10-05 10:09:45", "2026-10-05 10:09:45")
        _add_object(conn, first, "table", 3.0, 0.0, 3.5, 0.97,
                    "2026-10-05 10:12:00", "2026-10-12 10:12:20")

        # Seen for the first time in the second scan.
        box = _add_object(conn, second, "box", 1.0, 0.0, 4.0, 0.78,
                          "2026-10-12 10:15:00", "2026-10-12 10:15:00")

        # What changed between the two scans.
        _add_change(conn, chair, first, second, "moved", 2.6, 0.0, 2.1,
                    "Moved about 0.8 m toward the window",
                    "2026-10-12 10:25:00")
        _add_change(conn, lamp, first, second, "disappeared", 4.2, 0.0, 0.8,
                    None, "2026-10-12 10:25:00")
        _add_change(conn, box, first, second, "appeared", 1.0, 0.0, 4.0,
                    None, "2026-10-12 10:25:00")

        # A change the detector could not tie to one tracked object.
        _add_change(conn, None, first, second, "appeared", 5.0, 0.0, 1.0,
                    "Unidentified object near the east wall",
                    "2026-10-12 10:25:00")


def _add_session(conn, drone_id, user_id, started, ended):
    """One session with a couple of camera frames and a saved map."""
    session_id = conn.execute(
        "INSERT INTO Session (drone_id, started_at, ended_at) VALUES (?, ?, ?)",
        (drone_id, started, ended),
    ).lastrowid

    day = started[:10]

    for kind, folder in (("rgb_frame", "rgb"), ("depth_frame", "depth")):
        extension = "png" if kind == "rgb_frame" else "npy"
        mime = "image/png" if kind == "rgb_frame" else "application/octet-stream"

        _add_file(
            conn, session_id, drone_id, user_id, kind,
            f"{folder}_000001.{extension}",
            f"data/{day}/{folder}/{folder}_000001.{extension}",
            mime, 250_000, started,
        )

    map_file = _add_file(
        conn, session_id, drone_id, user_id, "map_export",
        "occupancy_grid.npy", f"data/{day}/map/occupancy_grid.npy",
        "application/octet-stream", 80_000, ended,
    )

    conn.execute(
        "INSERT INTO MapRecord (session_id, file_id, resolution, origin_x, "
        "origin_y, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        (session_id, map_file, 0.05, -10.0, -10.0, ended),
    )

    return session_id


def _add_file(conn, session_id, drone_id, owner_id, file_type, filename,
              storage_url, mime_type, size_bytes, captured_at):
    return conn.execute(
        "INSERT INTO File (session_id, drone_id, owner_id, file_type, "
        "filename, storage_url, mime_type, size_bytes, captured_at, "
        "uploaded_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (session_id, drone_id, owner_id, file_type, filename, storage_url,
         mime_type, size_bytes, captured_at, captured_at),
    ).lastrowid


def _add_object(conn, session_id, label, x, y, z, confidence, first, last):
    return conn.execute(
        "INSERT INTO ObjectRecord (session_id, label, x, y, z, confidence, "
        "first_seen_at, last_seen_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (session_id, label, x, y, z, confidence, first, last),
    ).lastrowid


def _add_change(conn, object_id, previous, current, change_type, x, y, z,
                description, detected_at):
    return conn.execute(
        "INSERT INTO ChangeRecord (object_id, previous_session_id, "
        "current_session_id, change_type, x, y, z, description, detected_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (object_id, previous, current, change_type, x, y, z, description,
         detected_at),
    ).lastrowid


if __name__ == "__main__":
    connection = db.connect()
    seed(connection)
    connection.close()
    print(f"Sample data added to {db.db_path()}")

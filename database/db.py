"""Opening and building the DELTA database.

Everything that touches SQLite connection details lives in this one file, so
that moving to PostgreSQL later means editing this file and not the code that
uses it.

    import db

    conn = db.connect()                      # database/delta.db
    conn.execute("SELECT * FROM Session")

Set the DELTA_DB environment variable to use a different file, for example a
scratch copy for testing.
"""

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DATABASE_DIR = Path(__file__).resolve().parent
SCHEMA_PATH = DATABASE_DIR / "schema.sql"
DEFAULT_DB_PATH = DATABASE_DIR / "delta.db"
DB_ENV_VAR = "DELTA_DB"

# In the order they are created, which is also the order a foreign key can
# point: every table only refers to tables above it.
TABLES = [
    "User",
    "Drone",
    "Session",
    "File",
    "MapRecord",
    "ObjectRecord",
    "ChangeRecord",
]

MEMORY = ":memory:"


def db_path():
    """Where the database file is: $DELTA_DB if set, otherwise delta.db here."""
    return os.environ.get(DB_ENV_VAR) or str(DEFAULT_DB_PATH)


def now():
    """The current time in the format every timestamp column uses (UTC)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def connect(path=None):
    """Open the database with the settings DELTA relies on.

    Foreign keys are OFF by default in SQLite, and the setting belongs to the
    connection, not the file. Without this PRAGMA every FOREIGN KEY in
    schema.sql would be silently ignored.
    """
    path = str(path) if path else db_path()

    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")

    if path != MEMORY:
        # Readers do not block the writer, so the app can read while the
        # simulation saves a session.
        conn.execute("PRAGMA journal_mode = WAL")

    return conn


def init_db(path=None, reset=False):
    """Create a new database from schema.sql and return an open connection.

    Raises FileExistsError if the file is already there, unless reset=True,
    which deletes it first. That is deliberate: rebuilding throws away every
    row, so it should never happen by accident.
    """
    path = str(path) if path else db_path()

    if path != MEMORY:
        target = Path(path)

        if target.exists():
            if not reset:
                raise FileExistsError(
                    f"{target} already exists; pass reset=True "
                    "(or --reset on the command line) to rebuild it"
                )

            # WAL mode keeps two side files next to the database.
            for suffix in ("", "-wal", "-shm"):
                Path(path + suffix).unlink(missing_ok=True)

        target.parent.mkdir(parents=True, exist_ok=True)

    conn = connect(path)
    conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))

    return conn

import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

# Lets the app import db.py from the database folder next to it.
sys.path.append(str(Path(__file__).resolve().parents[1] / "database"))

import db

app = FastAPI()

# The folder that holds our web page (index.html)
STATIC = Path(__file__).resolve().parent / "static"


def query(sql, params=()):
    # Opens the database, runs one query, and hands back the rows as plain dicts
    conn = db.connect()
    try:
        return [dict(row) for row in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()


@app.get("/")
def home():
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health():
    return {"message": "DELTA backend is running"}


@app.get("/api/drones")
def list_drones():
    return query("SELECT drone_id, serial_number, owner_id, status FROM Drone")


@app.get("/api/sessions")
def list_sessions():
    # Newest scan first
    return query(
        "SELECT session_id, drone_id, started_at, ended_at "
        "FROM Session ORDER BY started_at DESC"
    )


@app.get("/api/sessions/{session_id}/changes")
def list_changes(session_id: int):
    # LEFT JOIN keeps changes that aren't tied to a specific object
    return query(
        "SELECT c.change_id, c.change_type, c.x, c.y, c.z, c.description, "
        "c.detected_at, o.label "
        "FROM ChangeRecord c "
        "LEFT JOIN ObjectRecord o ON o.object_id = c.object_id "
        "WHERE c.current_session_id = ? "
        "ORDER BY c.change_id",
        (session_id,),
    )
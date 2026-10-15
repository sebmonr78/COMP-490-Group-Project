import sys
from pathlib import Path

from fastapi import FastAPI

# Lets the app import db.py from the database folder next to it.
sys.path.append(str(Path(__file__).resolve().parents[1] / "database"))

import db

app = FastAPI()


@app.get("/")
def home():
    return {"message": "DELTA backend is running"}


@app.get("/api/drones")
def list_drones():
    conn = db.connect()
    try:
        rows = conn.execute(
            "SELECT drone_id, serial_number, owner_id, status FROM Drone"
        ).fetchall()
    finally:
        conn.close()
    return [dict(row) for row in rows]
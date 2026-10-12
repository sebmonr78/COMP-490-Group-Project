# DELTA Database

This folder has the database for DELTA. It's what remembers everything the drone finds: who owns which drone, every scan it did, the maps it made, the objects it spotted, and what changed between scans.

We're using **SQLite** for now (more on why below).

## Quick start

You don't need to install anything. Python already comes with SQLite.

```
cd database
python create_db.py --seed
python test_db.py
```

- `create_db.py --seed` builds a new `delta.db` file and fills it with fake practice data so the app has something to show.
- `test_db.py` runs 31 checks to make sure the database is set up right and rejects bad data. It should end with `OK`.

If `python` doesn't work, try `python3`.

**Want to start over?** Run `python create_db.py --reset --seed`. It deletes your old `delta.db` and rebuilds it. It won't overwrite an existing database unless you say `--reset`, so you can't wipe your data by accident.

`delta.db` is in `.gitignore`, so it doesn't get committed. Everyone builds their own copy with the commands above.

## What's in this folder

| File | What it does |
|---|---|
| `schema.sql` | The blueprint. Defines all 7 tables and the rules for them. |
| `db.py` | The one place the rest of the app should use to open the database. It also turns on foreign keys (see "Gotchas"). |
| `create_db.py` | Builds `delta.db` from `schema.sql`. |
| `seed.py` | Adds the fake practice data. |
| `test_db.py` | The checks. |
| `.gitignore` | Keeps `delta.db` out of git. |

## The tables

| Table | What it holds |
|---|---|
| `User` | People who log in. |
| `Drone` | Each drone, and which user owns it. A drone with no owner is `unpaired`. |
| `Session` | One trip, meaning one time a drone went out and scanned. |
| `File` | The saved camera frames (RGB and depth) and map exports. |
| `MapRecord` | Info about a saved map, like its resolution and origin. Points to its `File`. |
| `ObjectRecord` | Things the drone spotted (chair, lamp, etc.), with their position. |
| `ChangeRecord` | What changed between two sessions: `moved`, `appeared`, or `disappeared`. |

How they connect, in short:
- A user has many drones, and a drone has many sessions.
- A session has many files, maps, objects, and changes.
- A change record points to two sessions (the earlier scan and the later scan) and, usually, to the object that changed.

The full diagrams are in the Phase 2 document.

**Heavy data isn't stored in the database.** The actual RGB images, depth data, and map files live as normal files on disk. The `File` table just stores the path (`storage_url`) and some info about each one. This keeps the database small and fast.

## Using it in code

Always open the database through `db.py`, never with `sqlite3.connect` directly:

```python
import db

conn = db.connect()
rows = conn.execute("SELECT * FROM Session").fetchall()
```

Timestamps are text in the format `2026-10-12 12:38:00` (UTC). `db.now()` gives you the current time in that format.

You can also point at a different database file by setting the `DELTA_DB` environment variable. This is handy for testing without touching your real `delta.db`.

## Why SQLite for now

- **No setup.** There's no server to install or run. The whole database is one file.
- **Fits what we're doing.** Right now it's mostly one thing writing data (the simulation) and the app reading it. SQLite handles that fine.
- **Easy for the whole team.** Everyone can build their own copy in a few seconds.
- **Good enough for this phase.** We're building a simulation first, and a heavier database wouldn't make it any better yet.

Basically, it lets us focus on the project instead of on setting up a database server. If we outgrow it, we move.

## Moving to PostgreSQL or MySQL later

If we end up with the real robot, many users, or lots of things writing at the same time, we'll probably want PostgreSQL. That's doable, but it's some work and not a copy-paste. Here's what I'd need to change:

- **Table names.** `User` is a reserved word in PostgreSQL, so it has to be quoted or renamed (like `Users`).
- **IDs.** `INTEGER PRIMARY KEY` would change to `GENERATED ... AS IDENTITY`.
- **Timestamps.** SQLite stores them as plain text. PostgreSQL has real timestamp types, so the columns and the way we write times would need a pass.
- **`db.py`.** It's where everything SQLite-specific lives (the connection and the `PRAGMA` lines), so most of the swap happens here. That's why we always go through it.
- **The rest of the code.** The SQL uses `?` for values, and PostgreSQL drivers use `%s`. The `conn.execute(...)` shortcut we use is SQLite-only too. `seed.py` and `test_db.py` would need small edits for these.
- **Moving the data.** We'd export from SQLite and import into PostgreSQL. Since we mostly keep seed data right now, this shouldn't be a big deal.

I tried to keep things standard to make the move easier: plain SQL, `CHECK` constraints, one access file, and a schema file in git. But it's manageable, not "no problem."

## Gotchas

- **Foreign keys are off by default in SQLite.** They only work if every connection turns them on with `PRAGMA foreign_keys = ON`. `db.connect()` does it for you, which is another reason to always use it.
- **SQLite doesn't enforce `VARCHAR(50)` lengths.** We use `CHECK` constraints for fields that only allow certain values (`status`, `file_type`, `change_type`).
- **No cascading deletes, on purpose.** You can't delete a user who still has drones, so scan history can't disappear by accident.
- **`delta.db-wal` and `delta.db-shm`** are extra files SQLite makes while the database is open (we use WAL mode so reading and writing don't block each other). They're normal, and they're ignored by git too.
- **The seed data is fake.** The password hash in it is a placeholder, so don't use it for anything real.

## What's left to do

- Hook the front end up to the database (login screen and map view).
- Real password hashing once login is built.
- Figure out where the image and map files get saved, so `storage_url` points somewhere real.

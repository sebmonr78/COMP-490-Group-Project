# DELTA Database

This is the database for our DELTA project. It stores the stuff the drone figures out: which user owns which drone, each scan (we call them sessions), the maps, the objects it sees, and what changed from one scan to the next.

We went with SQLite for now. Reasons are down below.

## How to run it

You don't have to install anything, Python already has SQLite built in. From inside the `database` folder:

```
python3 create_db.py --seed
python3 test_db.py
```

(On Windows it's probably `python` instead of `python3`.)

The first one makes a file called `delta.db` and fills it with fake data so we have something to look at. The second runs 31 tests and should end with `OK`.

If you mess up the database and want a fresh one, run `python3 create_db.py --reset --seed`. It won't overwrite your database unless you add `--reset`, so you can't delete your data by accident.

`delta.db` is in the `.gitignore` so it doesn't get pushed. Everybody just makes their own.

## Files

- `schema.sql` - creates the 7 tables and sets the rules for them
- `db.py` - how the rest of the code should connect to the database
- `create_db.py` - builds `delta.db` from `schema.sql`
- `seed.py` - the fake sample data
- `passwords.py` - hashing and checking login passwords (PBKDF2, built into Python)
- `test_db.py` - the tests

## Tables

- `User` - people who log in (name, email, phone, password hash)
- `Drone` - each drone and who owns it (no owner = unpaired)
- `Session` - one trip where the drone scanned an area
- `File` - saved RGB frames, depth frames, and map exports
- `MapRecord` - info about a saved map (resolution, origin), linked to its file
- `ObjectRecord` - things the drone spotted, like a chair, plus where they were
- `ChangeRecord` - what changed between two sessions: moved, appeared, or disappeared

A user has drones, a drone has sessions, and a session has files, maps, objects, and changes. A change record points at two sessions (the old scan and the new one) and usually at the object that changed. The diagrams are in the Phase 2 doc.

The actual images and map data are NOT stored in the database. They're regular files on disk, and the `File` table just keeps the path and some info about them. That keeps the database small.

## Using it in code

Use `db.py` to connect instead of calling `sqlite3.connect` yourself:

```python
import db

conn = db.connect()
rows = conn.execute("SELECT * FROM Session").fetchall()
```

Times are stored like `2026-10-12 12:38:00` (UTC). `db.now()` gives you the current time in that format. You can also set the `DELTA_DB` environment variable if you want to use a different database file for testing.

## Why SQLite

- nothing to set up, no server, the whole database is one file
- right now it's basically the simulation saving data and the app reading it, and SQLite is fine for that
- everyone on the team can have their own copy in a few seconds
- we're still in the simulation phase, so a bigger database wouldn't really give us anything yet

Mostly we didn't want to spend time on database setup when we could be working on the actual project.

## Switching to PostgreSQL later

If we end up with the real robot, more users, or a bunch of things writing at once, we'd probably switch to PostgreSQL. It's doable but not a copy and paste. Stuff that would need to change:

- `User` is a reserved word in PostgreSQL, so that table needs to be renamed or quoted
- the `INTEGER PRIMARY KEY` ids would change to PostgreSQL's identity columns
- SQLite saves timestamps as plain text, PostgreSQL has real timestamp types, so those columns and how we write them need to be updated
- `db.py` is where all the SQLite-specific connection stuff lives, so most of the change goes there
- SQLite uses `?` for values and PostgreSQL uses `%s`, and `conn.execute()` is SQLite only, so `seed.py` and `test_db.py` need small changes
- we'd have to move the existing data over (export and import)

I tried to keep the SQL standard so this is easier (plain SQL, CHECK constraints, everything going through `db.py`, schema in git). It should be manageable, just not "no problem."

## Things to know

- SQLite has foreign keys turned off by default. `db.connect()` turns them on, which is another reason to use it instead of connecting directly.
- SQLite doesn't actually enforce the VARCHAR lengths, so for things like `status`, `file_type`, and `change_type` we used CHECK constraints.
- There's no cascade delete on purpose. You can't delete a user who still has drones, so scan history can't get wiped by accident.
- You might see `delta.db-wal` and `delta.db-shm` show up. SQLite makes those while the database is open. They're normal and git ignores them.
- The seed data is fake. The demo user logs in with `demo@example.com` / `demo1234`.

## Still to do

- connect the map view to the front end
- decide where the image and map files get saved so `storage_url` points to something real

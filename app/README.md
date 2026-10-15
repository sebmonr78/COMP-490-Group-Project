# DELTA App

This is the web app part of DELTA. The backend (FastAPI) pulls data out of the SQLite database in `../database`, and a simple web page shows the drones and what changed between scans.

## Setting it up

Make sure the database exists first. If you haven't built it yet, `../database/README.md` has the details, but basically:

```
cd database
python3 create_db.py --seed
```

Then in the `app` folder, make a virtual environment and install the packages:

```
cd app
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

On Windows the activate line is `venv\Scripts\activate`. If it worked you'll see `(venv)` at the start of your terminal line.

## Running it

```
uvicorn main:app --reload
```

Go to http://127.0.0.1:8000 and you should see the page. `Ctrl+C` stops it, and `--reload` restarts it on its own whenever you save a file. There's also an auto-made API page at http://127.0.0.1:8000/docs if you want to poke around.

## What's in here

- `main.py` is the backend (the endpoints and the database queries)
- `static/index.html` is the page
- `requirements.txt` is what pip installs

## Endpoints

- `/` the page
- `/api/health` just checks the backend is up
- `/api/drones` all the drones
- `/api/sessions` all the scans, newest first
- `/api/sessions/{id}/changes` what moved, appeared, or disappeared in that scan compared to the one before it

## Heads up

- Open the page through uvicorn (the link above), not Live Server. Live Server doesn't have our API, so you'll just get "Server returned 404."
- No login yet, so it shows every drone and always the newest scan.
- `venv` isn't in git, so everybody makes their own.
- The backend uses `database/db.py` to open the database, so keep the `database` folder next to `app`.

## Still to do

- login screen
- map view
- let you pick which scan to look at instead of always the newest

import re
import secrets
import sqlite3
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel

# lets the app import db.py from the database folder next to it
sys.path.append(str(Path(__file__).resolve().parents[1] / "database"))

import db
import passwords
import simulations

@asynccontextmanager
async def lifespan(app):
    yield
    # uvicorn is shutting down (Ctrl+C, --reload, terminal closed): stop any
    # simulation it started, and its in-browser screen
    simulations.stop_on_shutdown()

app = FastAPI(lifespan=lifespan)

# The Run buttons for the simulation scripts (see simulations.py)
app.include_router(simulations.router)

# folder that holds our web page (index.html)
STATIC = Path(__file__).resolve().parent / "static"


def query(sql, params=()):
    # Opens the database, runs one query, and hands back the rows as plain dicts
    conn = db.connect()
    try:
        return [dict(row) for row in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()

# who is logged in: maps a random session token to (user_id, expires_at)
# it lives in memory, so restarting the server logs everyone out
SESSIONS = {}
SESSION_COOKIE = "delta_session"
SESSION_HOURS = 2  # how long a login lasts before you have to log in again

# reset tokens waiting to be used: token --> (user_id, expires_at)
# in memory, like SESSIONS, so a server restart forgets them
RESET_TOKENS = {}
RESET_MINUTES = 15

def current_user(request: Request):
    # the logged in user's id, or None if there is no valid ticket
    token = request.cookies.get(SESSION_COOKIE)
    entry = SESSIONS.get(token)
    if entry is None:
        return None
    user_id, expires_at = entry
    if expires_at < time.time():
        del SESSIONS[token]        # ticket ran out, so throw it away
        return None
    return user_id

def require_login(request: Request):
    # used on the data endpoints: no valid ticket --> 401, no data
    if current_user(request) is None:
        raise HTTPException(status_code=401, detail="You got to log in!! Your drone is in another castle.")

@app.get("/")
def home():
    # login page start
    return FileResponse(STATIC / "login.html")


@app.get("/dashboard")
def dashboard(request: Request):
    # not logged in --> goes back to the login page
    if current_user(request) is None:
        return RedirectResponse("/")
    # no-store tells the browser not to keep a copy of this page, so the Back
    # button can't show the dashboard again after logging out
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-store"})


@app.get("/api/health")
def health():
    return {"message": "DELTA backend is running"}


@app.get("/api/drones", dependencies=[Depends(require_login)])
def list_drones():
    return query("SELECT drone_id, serial_number, owner_id, status FROM Drone")


@app.get("/api/sessions", dependencies=[Depends(require_login)])
def list_sessions():
    # Newest scan first
    return query(
        "SELECT session_id, drone_id, started_at, ended_at "
        "FROM Session ORDER BY started_at DESC"
    )


@app.get("/api/sessions/{session_id}/changes", dependencies=[Depends(require_login)])
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


# ---------------------------------------------------------------------------
# Login, recovery and registration (Log in State Diagram)
#
# Each endpoint below is one of the "Backend DB" boxes in the diagram.
# A database problem comes back as a 500, which the page shows as the
# Server Error Box.
# ---------------------------------------------------------------------------

EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
MIN_PASSWORD_LENGTH = 8


class LoginInfo(BaseModel):
    email: str = ""
    password: str = ""


class PhoneInfo(BaseModel):
    phone_number: str = ""


class PasswordRecoveryInfo(BaseModel):
    email: str = ""
    phone_number: str = ""


class RegistrationInfo(BaseModel):
    first_name: str = ""
    last_name: str = ""
    email: str = ""
    email_confirm: str = ""
    password: str = ""
    password_confirm: str = ""
    phone_number: str = ""


def clean_email(email):
    return email.strip().lower()


def clean_phone(phone):
    # "(555) 555-0100" and "555.555.0100" both become "5555550100"
    return re.sub(r"\D", "", phone)


def run(sql, params=()):
    # Like query(), but turns a database failure into a 500 for the Server Error Box
    try:
        return query(sql, params)
    except sqlite3.Error:
        raise HTTPException(status_code=500, detail="Server error: could not reach the database")


@app.get("/login")
def login_page():
    return FileResponse(STATIC / "login.html")

@app.post("/api/login")
def login(info: LoginInfo, response: Response):
    # Backend DB: check if email and password are a correct pair
    rows = run(
        "SELECT user_id, first_name, password_hash FROM User WHERE email = ?",
        (clean_email(info.email),),
    )

    # Same message for a wrong email and a wrong password, so the page
    # doesn't tell strangers which emails have accounts.
    if not rows or not passwords.verify_password(info.password, rows[0]["password_hash"]):
        raise HTTPException(status_code=401, detail="Email or password is wrong")

    # correct login: give the browser a session ticket
    token = secrets.token_urlsafe(32)
    SESSIONS[token] = (rows[0]["user_id"], time.time() + SESSION_HOURS * 3600)
    response.set_cookie(SESSION_COOKIE, token, httponly=True, samesite="lax",
                        max_age=SESSION_HOURS * 3600)

    return {"user_id": rows[0]["user_id"], "first_name": rows[0]["first_name"]}

@app.post("/api/logout")
def logout(request: Request, response: Response):
    # Throw away the ticket on the server, then tell the browser to forget it too
    SESSIONS.pop(request.cookies.get(SESSION_COOKIE), None)
    response.delete_cookie(SESSION_COOKIE)
    return {"message": "Logged out"}

@app.post("/api/recover-email")
def recover_email(info: PhoneInfo):
    # Backend DB: is phone number in system
    phone = clean_phone(info.phone_number)
    rows = run("SELECT email FROM User WHERE phone_number = ?", (phone,)) if phone else []

    if not rows:
        raise HTTPException(status_code=404, detail="That phone number is not in our system")

    return {"email": rows[0]["email"]}


@app.post("/api/recover-password")
def recover_password(info: PasswordRecoveryInfo):
    # Backend DB: is email in system (and does the phone number match it)
    email = clean_email(info.email)
    phone = clean_phone(info.phone_number)
    rows = run(
        "SELECT user_id FROM User WHERE email = ? AND phone_number = ?",
        (email, phone),
    ) if email and phone else []

    if not rows:
        raise HTTPException(
            status_code=404,
            detail="No account matches that email and phone number",
        )

    # No email server yet, so the reset link goes to the uvicorn terminal
    # instead of an inbox. Swap this print for a real email send later.
    token = secrets.token_urlsafe(32)
    RESET_TOKENS[token] = (rows[0]["user_id"], time.time() + RESET_MINUTES * 60)
    print(f"[password reset] {email}: http://127.0.0.1:8000/reset-password?token={token}")

    return {"message": f"An email to reset your password was sent to {email}"}


class ResetInfo(BaseModel):
    token: str = ""
    password: str = ""
    password_confirm: str = ""


@app.get("/reset-password")
def reset_page():
    return FileResponse(STATIC / "reset.html")


@app.post("/api/reset-password")
def reset_password(info: ResetInfo):
    entry = RESET_TOKENS.get(info.token)
    if entry is None or entry[1] < time.time():
        RESET_TOKENS.pop(info.token, None)
        raise HTTPException(status_code=400, detail="This reset link is invalid or has expired")
    if len(info.password) < MIN_PASSWORD_LENGTH:
        raise HTTPException(status_code=400, detail=f"Password must be at least {MIN_PASSWORD_LENGTH} characters")
    if info.password != info.password_confirm:
        raise HTTPException(status_code=400, detail="Passwords do not match")

    user_id = entry[0]
    try:
        conn = db.connect()
        try:
            with conn:
                conn.execute("UPDATE User SET password_hash = ? WHERE user_id = ?",
                             (passwords.hash_password(info.password), user_id))
        finally:
            conn.close()
    except sqlite3.Error:
        raise HTTPException(status_code=500, detail="Server error: could not save the new password")

    del RESET_TOKENS[info.token]          # a link only works once
    for key in [k for k, (u, _) in SESSIONS.items() if u == user_id]:
        del SESSIONS[key]                 # log out anyone already signed in as them
    return {"message": "Password changed"}


@app.post("/api/register", status_code=201)
def register(info: RegistrationInfo):
    # Backend DB, first box: check all info is filled out and valid
    fields = {
        "first_name": info.first_name.strip(),
        "last_name": info.last_name.strip(),
        "email": clean_email(info.email),
        "email_confirm": clean_email(info.email_confirm),
        "password": info.password,
        "password_confirm": info.password_confirm,
        "phone_number": clean_phone(info.phone_number),
    }

    missing = [name for name, value in fields.items() if not value]
    if missing:
        raise HTTPException(
            status_code=400,
            detail={"message": "Not all the registration info was filled out", "fields": missing},
        )

    problems = {}
    if not EMAIL_PATTERN.match(fields["email"]):
        problems["email"] = "Enter a valid email address"
    if fields["email"] != fields["email_confirm"]:
        problems["email_confirm"] = "Emails do not match"
    if len(fields["password"]) < MIN_PASSWORD_LENGTH:
        problems["password"] = f"Password must be at least {MIN_PASSWORD_LENGTH} characters"
    if fields["password"] != fields["password_confirm"]:
        problems["password_confirm"] = "Passwords do not match"
    if not 10 <= len(fields["phone_number"]) <= 15:
        problems["phone_number"] = "Enter a 10 to 15 digit phone number"

    if problems:
        raise HTTPException(
            status_code=400,
            detail={"message": "Some of the registration info is not valid", "fields": problems},
        )

    # Backend DB, first box: check the info is not already taken
    taken = {}
    if run("SELECT 1 FROM User WHERE email = ?", (fields["email"],)):
        taken["email"] = "That email already has an account"
    if run("SELECT 1 FROM User WHERE phone_number = ?", (fields["phone_number"],)):
        taken["phone_number"] = "That phone number already has an account"

    if taken:
        raise HTTPException(
            status_code=409,
            detail={"message": "Registration info inputted is already taken", "fields": taken},
        )

    # Backend DB, second box: upload the registration info to the server.
    # The schema needs a username, so the email doubles as one.
    try:
        conn = db.connect()
        try:
            with conn:
                conn.execute(
                    "INSERT INTO User (username, email, password_hash, first_name, "
                    "last_name, phone_number, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        fields["email"],
                        fields["email"],
                        passwords.hash_password(fields["password"]),
                        fields["first_name"],
                        fields["last_name"],
                        fields["phone_number"],
                        db.now(),
                    ),
                )
        finally:
            conn.close()
    except sqlite3.IntegrityError:
        # Someone registered the same email or phone a split second earlier
        raise HTTPException(
            status_code=409,
            detail={"message": "Registration info inputted is already taken", "fields": {}},
        )
    except sqlite3.Error:
        raise HTTPException(status_code=500, detail="Server error: could not save the account")

    return {"message": "Account created"}

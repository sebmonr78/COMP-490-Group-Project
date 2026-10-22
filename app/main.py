import re
import secrets
import sqlite3
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Lets the app import db.py from the database folder next to it.
sys.path.append(str(Path(__file__).resolve().parents[1] / "database"))

import db
import passwords

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
    # login page start
    return FileResponse(STATIC / "login.html")


@app.get("/dashboard")
def dashboard():
    # Drones and what changed (where you land after logging in)
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
def login(info: LoginInfo):
    # Backend DB: check if email and password are a correct pair
    rows = run(
        "SELECT user_id, first_name, password_hash FROM User WHERE email = ?",
        (clean_email(info.email),),
    )

    # Same message for a wrong email and a wrong password, so the page
    # doesn't tell strangers which emails have accounts.
    if not rows or not passwords.verify_password(info.password, rows[0]["password_hash"]):
        raise HTTPException(status_code=401, detail="Email or password is wrong")

    return {"user_id": rows[0]["user_id"], "first_name": rows[0]["first_name"]}


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
    print(f"[password reset] {email}: http://127.0.0.1:8000/reset-password?token={token}")

    return {"message": f"An email to reset your password was sent to {email}"}


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

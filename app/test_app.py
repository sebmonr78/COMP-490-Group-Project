"""Checks for the DELTA app: login, logout, per-user data, the lockout,
password reset and the reset email.

    cd app
    python3 test_app.py

Every test builds its own scratch database in a temp folder, so your
delta.db is never touched, and no real email is ever sent.
"""

import os
import re
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

# Point the app at a scratch database BEFORE it gets imported, and make sure
# a real email setup on this computer can't leak into the tests.
TEMP = tempfile.TemporaryDirectory()
os.environ["DELTA_DB"] = str(Path(TEMP.name) / "test.db")
for name in list(os.environ):
    if name.startswith("DELTA_SMTP") or name == "DELTA_BASE_URL":
        del os.environ[name]

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "database"))

import db
import seed
from fastapi.testclient import TestClient

import main

DEMO = {"email": "demo@example.com", "password": "demo1234"}
DEMO_PHONE = "5555550100"


def login(client, email, password):
    return client.post("/api/login", json={"email": email, "password": password})


def register(client, email, password="secret123", phone="5555550199"):
    return client.post("/api/register", json={
        "first_name": "Test", "last_name": "User",
        "email": email, "email_confirm": email,
        "password": password, "password_confirm": password,
        "phone_number": phone,
    })


def ask_for_reset(client, email=DEMO["email"], phone=DEMO_PHONE):
    """Press "forgot password". Returns the response and what got printed."""
    printed = StringIO()
    with redirect_stdout(printed):
        response = client.post("/api/recover-password",
                               json={"email": email, "phone_number": phone})
    return response, printed.getvalue()


def new_password(client, token, password, confirm=None):
    return client.post("/api/reset-password", json={
        "token": token, "password": password,
        "password_confirm": password if confirm is None else confirm,
    })


class AppTest(unittest.TestCase):
    def setUp(self):
        # a fresh database and a clean slate for everything the app keeps in memory
        db.init_db(reset=True).close()
        conn = db.connect()
        seed.seed(conn)
        conn.commit()
        conn.close()

        main.SESSIONS.clear()
        main.FAILED_LOGINS.clear()
        main.RESET_TOKENS.clear()
        self.client = TestClient(main.app)

    def login_demo(self):
        response = login(self.client, **DEMO)
        self.assertEqual(response.status_code, 200)


class LoginAndLogout(AppTest):
    def test_login_page_is_the_default_page(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Log in", response.text)

    def test_dashboard_sends_you_to_login_without_a_ticket(self):
        response = self.client.get("/dashboard", follow_redirects=False)
        self.assertEqual(response.status_code, 307)
        self.assertEqual(response.headers["location"], "/")

    def test_right_login_opens_the_dashboard(self):
        self.login_demo()
        response = self.client.get("/dashboard")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Log out", response.text)
        self.assertEqual(response.headers["cache-control"], "no-store")

    def test_wrong_password_and_unknown_email_look_the_same(self):
        wrong = login(self.client, DEMO["email"], "nope")
        unknown = login(self.client, "nobody@example.com", "nope")
        self.assertEqual(wrong.status_code, 401)
        self.assertEqual(unknown.status_code, 401)
        self.assertEqual(wrong.json(), unknown.json())

    def test_logout_locks_the_dashboard_and_data_again(self):
        self.login_demo()
        self.assertEqual(self.client.post("/api/logout").status_code, 200)
        self.assertEqual(self.client.get("/dashboard", follow_redirects=False).status_code, 307)
        self.assertEqual(self.client.get("/api/drones").status_code, 401)

    def test_logged_out_data_and_simulation_endpoints_return_401(self):
        for method, url in [
            ("get", "/api/drones"), ("get", "/api/sessions"),
            ("get", "/api/sessions/2/changes"),
            ("get", "/api/simulations"), ("get", "/api/simulations/run"),
            ("post", "/api/simulations/stop"),
        ]:
            with self.subTest(url=url):
                self.assertEqual(getattr(self.client, method)(url).status_code, 401)

    def test_an_expired_ticket_is_refused(self):
        self.login_demo()
        for token, (user_id, _) in list(main.SESSIONS.items()):
            main.SESSIONS[token] = (user_id, time.time() - 1)
        self.assertEqual(self.client.get("/api/drones").status_code, 401)
        self.assertEqual(self.client.get("/dashboard", follow_redirects=False).status_code, 307)
        self.assertEqual(main.SESSIONS, {})  # the dead ticket was thrown away


class EachUserSeesOnlyTheirOwnData(AppTest):
    def test_demo_user_sees_their_drone_and_scans(self):
        self.login_demo()
        drones = self.client.get("/api/drones").json()
        self.assertEqual([d["serial_number"] for d in drones], ["DELTA-SIM-0001"])
        sessions = self.client.get("/api/sessions").json()
        self.assertEqual([s["session_id"] for s in sessions], [2, 1])  # newest first
        changes = self.client.get("/api/sessions/2/changes")
        self.assertEqual(changes.status_code, 200)
        self.assertEqual(len(changes.json()), 4)

    def test_a_new_user_sees_an_empty_dashboard(self):
        self.assertEqual(register(self.client, "new@example.com").status_code, 201)
        self.assertEqual(login(self.client, "new@example.com", "secret123").status_code, 200)
        self.assertEqual(self.client.get("/api/drones").json(), [])
        self.assertEqual(self.client.get("/api/sessions").json(), [])

    def test_somebody_elses_scan_is_not_found(self):
        register(self.client, "new@example.com")
        login(self.client, "new@example.com", "secret123")
        response = self.client.get("/api/sessions/2/changes")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"], "Scan not found")
        self.assertEqual(self.client.get("/api/sessions/999/changes").status_code, 404)


class Lockout(AppTest):
    def wrong_tries(self, email, count):
        return [login(self.client, email, "wrong").status_code for _ in range(count)]

    def test_five_wrong_tries_then_locked_even_for_the_right_password(self):
        self.assertEqual(self.wrong_tries(DEMO["email"], 5), [401] * 5)
        response = login(self.client, **DEMO)
        self.assertEqual(response.status_code, 429)
        self.assertIn("Too many wrong tries", response.json()["detail"])

    def test_unknown_emails_get_locked_too(self):
        self.assertEqual(self.wrong_tries("nobody@example.com", 6), [401] * 5 + [429])

    def test_a_right_login_resets_the_count(self):
        self.wrong_tries(DEMO["email"], 4)
        self.assertEqual(login(self.client, **DEMO).status_code, 200)
        self.assertEqual(self.wrong_tries(DEMO["email"], 4), [401] * 4)  # not locked
        self.assertEqual(login(self.client, **DEMO).status_code, 200)

    def test_the_lock_runs_out(self):
        self.wrong_tries(DEMO["email"], 5)
        main.FAILED_LOGINS[DEMO["email"]]["locked_until"] = time.time() - 1
        self.assertEqual(login(self.client, **DEMO).status_code, 200)

    def test_one_email_being_locked_does_not_lock_another(self):
        register(self.client, "other@example.com")
        self.wrong_tries(DEMO["email"], 5)
        self.assertEqual(login(self.client, "other@example.com", "secret123").status_code, 200)


class PasswordReset(AppTest):
    def test_unknown_email_or_wrong_phone_is_404(self):
        self.assertEqual(ask_for_reset(self.client, "nobody@example.com")[0].status_code, 404)
        self.assertEqual(ask_for_reset(self.client, phone="5555550111")[0].status_code, 404)

    def test_without_email_settings_the_link_is_printed(self):
        response, printed = ask_for_reset(self.client)
        self.assertEqual(response.status_code, 200)
        self.assertIn("/reset-password?token=", printed)
        self.assertEqual(len(main.RESET_TOKENS), 1)

    def test_the_link_changes_the_password_once(self):
        ask_for_reset(self.client)
        token = next(iter(main.RESET_TOKENS))
        self.assertEqual(self.client.get("/reset-password").status_code, 200)
        self.assertEqual(new_password(self.client, token, "brandnew123").status_code, 200)
        self.assertEqual(login(self.client, **DEMO).status_code, 401)  # old one is dead
        self.assertEqual(login(self.client, DEMO["email"], "brandnew123").status_code, 200)
        self.assertEqual(new_password(self.client, token, "another123").status_code, 400)

    def test_bad_new_passwords_are_refused_and_the_link_survives(self):
        ask_for_reset(self.client)
        token = next(iter(main.RESET_TOKENS))
        self.assertEqual(new_password(self.client, token, "short").status_code, 400)
        self.assertEqual(new_password(self.client, token, "longenough1", "different1").status_code, 400)
        self.assertEqual(new_password(self.client, token, "longenough1").status_code, 200)

    def test_an_expired_or_made_up_link_is_refused(self):
        ask_for_reset(self.client)
        token = next(iter(main.RESET_TOKENS))
        user_id, _ = main.RESET_TOKENS[token]
        main.RESET_TOKENS[token] = (user_id, time.time() - 1)
        self.assertEqual(new_password(self.client, token, "brandnew123").status_code, 400)
        self.assertEqual(new_password(self.client, "made-up", "brandnew123").status_code, 400)
        self.assertEqual(login(self.client, **DEMO).status_code, 200)  # nothing changed

    def test_a_reset_logs_out_browsers_that_were_signed_in(self):
        other_browser = TestClient(main.app)
        self.assertEqual(login(other_browser, **DEMO).status_code, 200)
        ask_for_reset(self.client)
        new_password(self.client, next(iter(main.RESET_TOKENS)), "brandnew123")
        self.assertEqual(other_browser.get("/api/drones").status_code, 401)


class FakeSMTP:
    """Stands in for a mail server and remembers what it was asked to do."""
    last = None

    def __init__(self, host, port, timeout=None):
        self.host, self.port, self.tls, self.logged_in, self.message = host, port, False, None, None
        FakeSMTP.last = self

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def starttls(self):
        self.tls = True

    def login(self, user, password):
        self.logged_in = (user, password)

    def send_message(self, message):
        self.message = message


class ResetEmail(AppTest):
    def settings(self, **extra):
        values = {"DELTA_SMTP_HOST": "smtp.example.com", "DELTA_SMTP_USER": "sender@example.com",
                  "DELTA_SMTP_PASSWORD": "app-password", "DELTA_BASE_URL": "https://delta.example.com/"}
        values.update(extra)
        return mock.patch.dict(os.environ, values)

    def test_the_email_has_the_right_parts_and_the_link_works(self):
        with self.settings(), mock.patch("smtplib.SMTP", FakeSMTP):
            response, printed = ask_for_reset(self.client)
        self.assertEqual(response.status_code, 200)
        self.assertNotIn("reset-password?token", printed)  # emailed, not printed

        server = FakeSMTP.last
        self.assertEqual((server.host, server.port), ("smtp.example.com", 587))
        self.assertTrue(server.tls)
        self.assertEqual(server.logged_in, ("sender@example.com", "app-password"))

        message = server.message
        self.assertEqual(message["To"], DEMO["email"])
        self.assertEqual(message["From"], "sender@example.com")
        self.assertEqual(message["Subject"], "Reset your DELTA password")

        body = message.get_content()
        self.assertIn("15 minutes", body)
        found = re.search(r"https://delta\.example\.com/reset-password\?token=([\w-]+)", body)
        self.assertIsNotNone(found)
        self.assertEqual(new_password(self.client, found.group(1), "brandnew123").status_code, 200)

    def test_a_mail_server_that_is_not_there_gives_a_500_and_no_usable_link(self):
        with self.settings(DELTA_SMTP_HOST="127.0.0.1", DELTA_SMTP_PORT="1"):
            response, printed = ask_for_reset(self.client)
        self.assertEqual(response.status_code, 500)
        self.assertIn("could not send the reset email", response.json()["detail"])
        self.assertEqual(main.RESET_TOKENS, {})
        self.assertNotIn("reset-password?token", printed)


if __name__ == "__main__":
    unittest.main(verbosity=2)

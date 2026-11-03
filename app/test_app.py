"""Checks for the DELTA app: login, logout, per-user data, the lockout,
password reset, the reset email, and simulations running side by side.

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
import simulations

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


# Stand-ins for the real simulations: plain Python, so these run anywhere.
WAIT_A_MINUTE = [sys.executable, "-u", "-c",
                 "import sys, time; print('slot', sys.argv[1], 'run', sys.argv[2], 'port', sys.argv[3]); "
                 "time.sleep(60)", "{slot}", "{run}", "{brain_port}"]

FAKE_FLIGHT = ["-c",
               "import json, pathlib, sys; out = pathlib.Path(sys.argv[1]); out.mkdir(parents=True); "
               "(out / 'report.json').write_text(json.dumps({'hover_minutes_on_4s2p': 14.3, "
               "'energy_wh': 5.64, 'real_time_factor': 7.0, 'steps': [{'name': 'take off', "
               "'seconds': 3.0, 'mean_watts': 420.0, 'max_tilt_deg': 3.2}]})); "
               "(out / 'flight.png').write_bytes(b'PNG fake')", "{out}"]

FAKES = {
    "wait": {"kind": "command", "title": "Wait", "command": WAIT_A_MINUTE,
             "description": "", "interactive": False},
    "flight": {"kind": "twin", "title": "Flight", "args": FAKE_FLIGHT, "results": True,
               "description": "", "interactive": False},
}


class SimulationsSideBySide(AppTest):
    """Each login session runs its own simulation, up to 3 on the server."""

    def setUp(self):
        super().setUp()
        self.twin_out = tempfile.TemporaryDirectory()
        patches = [
            mock.patch.dict(simulations.SIMULATIONS, FAKES, clear=True),
            mock.patch.object(simulations, "twin_python_command", return_value=[sys.executable]),
            mock.patch.object(simulations, "TWIN_DIR", Path(self.twin_out.name)),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        self.addCleanup(self.stop_everything)

    def stop_everything(self):
        simulations.stop_on_shutdown()
        deadline = time.time() + 10
        while time.time() < deadline and any(
                run["process"] and run["process"].poll() is None for run in simulations._runs.values()):
            time.sleep(0.05)
        time.sleep(0.2)  # the reader threads finish up
        simulations._runs.clear()
        self.twin_out.cleanup()

    def browser(self, email=DEMO["email"], password=DEMO["password"]):
        """A browser of its own (its own cookie), logged in."""
        client = TestClient(main.app)
        self.assertEqual(login(client, email, password).status_code, 200)
        return client

    def wait_for(self, client, *statuses, seconds=10):
        deadline = time.time() + seconds
        while time.time() < deadline:
            run = client.get("/api/simulations/run").json()["run"]
            if run and run["status"] in statuses and (run["line_count"] or "running" not in statuses):
                return run
            time.sleep(0.05)
        self.fail(f"never reached {statuses}: {run}")

    def test_three_sessions_run_at_once_each_with_its_own_slot(self):
        register(self.client, "second@example.com")
        browsers = [self.browser(), self.browser(), self.browser("second@example.com", "secret123")]

        for client in browsers:
            self.assertEqual(client.post("/api/simulations/wait/start").status_code, 200)

        runs = [self.wait_for(client, "running") for client in browsers]
        self.assertEqual(sorted(run["slot"] for run in runs), [1, 2, 3])
        for run in runs:
            # what the run was told matches its slot: its own map name and brain port
            self.assertEqual(run["lines"], [f"slot {run['slot']} run web-{run['slot']} port {47599 + run['slot']}"])
            self.assertEqual(run["server_running"], 3)

        # A fourth is refused while three run...
        fourth = self.browser()
        response = fourth.post("/api/simulations/wait/start")
        self.assertEqual(response.status_code, 503)
        self.assertIn("already running 3", response.json()["detail"])

        # ...and gets the freed slot once one stops
        self.assertEqual(browsers[1].post("/api/simulations/stop").status_code, 200)
        freed = self.wait_for(browsers[1], "stopped")["slot"]
        self.assertEqual(fourth.post("/api/simulations/wait/start").json()["slot"], freed)

    def test_a_session_sees_and_stops_only_its_own_run(self):
        mine, theirs = self.browser(), self.browser()
        mine.post("/api/simulations/wait/start")
        theirs.post("/api/simulations/wait/start")
        self.wait_for(mine, "running")
        self.wait_for(theirs, "running")

        mine.post("/api/simulations/stop")
        self.wait_for(mine, "stopped")
        self.assertEqual(theirs.get("/api/simulations/run").json()["run"]["status"], "running")

    def test_one_run_at_a_time_per_session(self):
        client = self.browser()
        client.post("/api/simulations/wait/start")
        response = client.post("/api/simulations/wait/start")
        self.assertEqual(response.status_code, 409)

    def test_logging_out_stops_that_sessions_run_and_frees_its_slot(self):
        leaving, staying = self.browser(), self.browser()
        leaving.post("/api/simulations/wait/start")
        staying.post("/api/simulations/wait/start")
        self.wait_for(leaving, "running")
        process = next(run["process"] for run in simulations._runs.values() if run["slot"] == 1)

        leaving.post("/api/logout")
        process.wait(timeout=10)
        deadline = time.time() + 5
        while time.time() < deadline and len(simulations._runs) > 1:
            time.sleep(0.05)
        self.assertEqual(len(simulations._runs), 1)
        self.assertEqual(staying.get("/api/simulations/run").json()["run"]["status"], "running")

    def test_a_password_reset_stops_the_runs_of_the_browsers_it_logs_out(self):
        client = self.browser()
        client.post("/api/simulations/wait/start")
        self.wait_for(client, "running")
        process = next(iter(simulations._runs.values()))["process"]

        _, printed = ask_for_reset(self.client)
        token = re.search(r"token=([\w-]+)", printed).group(1)
        self.assertEqual(new_password(self.client, token, "brandnew123").status_code, 200)
        self.assertIsNotNone(process.wait(timeout=10))

    def test_the_twin_flight_shows_its_numbers_and_plot_to_its_own_session_only(self):
        mine, theirs = self.browser(), self.browser()
        mine.post("/api/simulations/flight/start")
        run = self.wait_for(mine, "finished")

        self.assertIn(["Hover time on the 4S2P battery", "14.3 min"], run["results"]["summary"])
        self.assertEqual(run["results"]["steps"][0]["name"], "take off")
        picture = mine.get(run["results"]["picture"])
        self.assertEqual(picture.status_code, 200)
        self.assertEqual(picture.headers["content-type"], "image/png")
        self.assertEqual(theirs.get(run["results"]["picture"]).status_code, 404)


if __name__ == "__main__":
    unittest.main(verbosity=2)

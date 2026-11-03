"""Sends the password reset email.

Email only goes out when DELTA_SMTP_HOST is set. If it isn't, send_reset_email()
returns False and main.py prints the link in the uvicorn terminal instead.

Settings come from environment variables, never from a file in git:
    DELTA_SMTP_HOST      mail server, e.g. smtp.gmail.com
    DELTA_SMTP_PORT      default 587
    DELTA_SMTP_USER      login for that mail server
    DELTA_SMTP_PASSWORD  its password (Gmail needs an "app password")
    DELTA_SMTP_FROM      who the email says it's from (default: the user)
    DELTA_BASE_URL       site address used in the link (default http://127.0.0.1:8000)
"""

import os
import smtplib
from email.message import EmailMessage


def send_reset_email(to_address, token, minutes):
    """Email the reset link. Returns False if email isn't set up.
    A failed send raises an OSError (smtplib's errors are OSErrors)."""
    host = os.environ.get("DELTA_SMTP_HOST")
    if not host:
        return False

    user = os.environ.get("DELTA_SMTP_USER")
    password = os.environ.get("DELTA_SMTP_PASSWORD")
    base_url = os.environ.get("DELTA_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
    link = f"{base_url}/reset-password?token={token}"

    message = EmailMessage()
    message["Subject"] = "Reset your DELTA password"
    message["From"] = os.environ.get("DELTA_SMTP_FROM") or user
    message["To"] = to_address
    message.set_content(
        f"Someone asked to reset the password for your DELTA account.\n\n"
        f"Open this link to choose a new one (it works once and lasts {minutes} minutes):\n"
        f"{link}\n\n"
        f"If that wasn't you, ignore this email and nothing changes."
    )

    with smtplib.SMTP(host, int(os.environ.get("DELTA_SMTP_PORT", "587")), timeout=15) as server:
        server.starttls()
        if user:
            server.login(user, password)
        server.send_message(message)
    return True

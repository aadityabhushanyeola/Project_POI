from __future__ import annotations

import os
import smtplib
from email.message import EmailMessage
from pathlib import Path

import requests

from src.behavior_analyzer import SuspiciousEvent


class AlertManager:
    def __init__(self, telegram_enabled: bool, email_enabled: bool):
        self.telegram_enabled = telegram_enabled
        self.email_enabled = email_enabled

    def send(self, event: SuspiciousEvent, snapshot_path: str | None = None) -> None:
        message = _format_message(event)
        if self.telegram_enabled:
            self._send_telegram(message, snapshot_path)
        if self.email_enabled:
            self._send_email(message, snapshot_path)

    def _send_telegram(self, message: str, snapshot_path: str | None) -> None:
        token = os.getenv("TELEGRAM_BOT_TOKEN")
        chat_id = os.getenv("TELEGRAM_CHAT_ID")
        if not token or not chat_id:
            print("Telegram alert skipped: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID missing.")
            return

        base_url = f"https://api.telegram.org/bot{token}"
        if snapshot_path and Path(snapshot_path).exists():
            with Path(snapshot_path).open("rb") as image_file:
                response = requests.post(
                    f"{base_url}/sendPhoto",
                    data={"chat_id": chat_id, "caption": message},
                    files={"photo": image_file},
                    timeout=15,
                )
        else:
            response = requests.post(
                f"{base_url}/sendMessage",
                data={"chat_id": chat_id, "text": message},
                timeout=15,
            )

        response.raise_for_status()

    def _send_email(self, message: str, snapshot_path: str | None) -> None:
        host = os.getenv("EMAIL_HOST")
        port = int(os.getenv("EMAIL_PORT", "587"))
        user = os.getenv("EMAIL_USER")
        password = os.getenv("EMAIL_PASSWORD")
        to_address = os.getenv("EMAIL_TO")

        if not host or not user or not password or not to_address:
            print("Email alert skipped: email environment variables missing.")
            return

        email = EmailMessage()
        email["Subject"] = "CCTV Suspicious Activity Alert"
        email["From"] = user
        email["To"] = to_address
        email.set_content(message)

        if snapshot_path and Path(snapshot_path).exists():
            image_bytes = Path(snapshot_path).read_bytes()
            email.add_attachment(image_bytes, maintype="image", subtype="jpeg", filename=Path(snapshot_path).name)

        with smtplib.SMTP(host, port) as smtp:
            smtp.starttls()
            smtp.login(user, password)
            smtp.send_message(email)


def _format_message(event: SuspiciousEvent) -> str:
    return (
        "Suspicious Activity Detected\n"
        f"Camera: {event.camera_name} ({event.camera_id})\n"
        f"Event: {event.event_type}\n"
        f"Track ID: {event.track_id if event.track_id is not None else 'multiple/unknown'}\n"
        f"Confidence: {event.confidence:.2f}\n"
        f"Reason: {event.reason}"
    )

"""Eigene Inhalte über die offizielle Instagram Graph API veröffentlichen.

Das ist der EINZIGE von Instagram erlaubte Automatisierungsweg zum Posten.
Voraussetzung: Business-/Creator-Account + Meta-Developer-App mit den Scopes
`instagram_basic`, `instagram_content_publish`.

Doku: https://developers.facebook.com/docs/instagram-platform/content-publishing
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional

import requests

GRAPH_API_VERSION = "v21.0"
GRAPH_BASE = f"https://graph.facebook.com/{GRAPH_API_VERSION}"


class PublishError(RuntimeError):
    """Wird ausgelöst, wenn die Graph API einen Fehler meldet."""


@dataclass
class IGCredentials:
    ig_user_id: str
    access_token: str

    @classmethod
    def from_env(cls) -> "IGCredentials":
        user_id = os.environ.get("IG_USER_ID")
        token = os.environ.get("IG_ACCESS_TOKEN")
        if not user_id or not token:
            raise PublishError(
                "IG_USER_ID und IG_ACCESS_TOKEN müssen als Umgebungsvariablen gesetzt sein. "
                "Siehe README.md → Setup."
            )
        return cls(ig_user_id=user_id, access_token=token)


class InstagramPublisher:
    """Dünner Wrapper um den zweistufigen Publish-Flow der Graph API.

    Ablauf laut Meta-Doku:
      1. Media-Container erstellen (POST /{ig-user-id}/media).
      2. Bei Video/Reel: warten, bis der Container ``FINISHED`` ist.
      3. Container veröffentlichen (POST /{ig-user-id}/media_publish).
    """

    def __init__(self, creds: IGCredentials, session: Optional[requests.Session] = None):
        self._creds = creds
        self._session = session or requests.Session()

    # -- interne Helfer -------------------------------------------------

    def _post(self, path: str, params: dict) -> dict:
        params = {**params, "access_token": self._creds.access_token}
        resp = self._session.post(f"{GRAPH_BASE}/{path}", data=params, timeout=60)
        return self._handle(resp)

    def _get(self, path: str, params: dict) -> dict:
        params = {**params, "access_token": self._creds.access_token}
        resp = self._session.get(f"{GRAPH_BASE}/{path}", params=params, timeout=60)
        return self._handle(resp)

    @staticmethod
    def _handle(resp: requests.Response) -> dict:
        try:
            data = resp.json()
        except ValueError:
            raise PublishError(f"Ungültige Antwort der Graph API (HTTP {resp.status_code}).")
        if resp.status_code >= 400 or "error" in data:
            err = data.get("error", {})
            msg = err.get("message", f"HTTP {resp.status_code}")
            raise PublishError(f"Graph API Fehler: {msg}")
        return data

    # -- öffentliche Methoden ------------------------------------------

    def create_image_container(self, image_url: str, caption: str) -> str:
        """Container für ein einzelnes Bild erstellen. Gibt die Container-ID zurück."""
        data = self._post(
            f"{self._creds.ig_user_id}/media",
            {"image_url": image_url, "caption": caption},
        )
        return data["id"]

    def create_reel_container(self, video_url: str, caption: str,
                              cover_url: Optional[str] = None,
                              share_to_feed: bool = True) -> str:
        """Container für ein Reel erstellen. Gibt die Container-ID zurück.

        ``video_url`` muss eine öffentlich erreichbare URL zu DEINEM eigenen
        Video sein (Meta lädt es von dort).
        """
        params = {
            "media_type": "REELS",
            "video_url": video_url,
            "caption": caption,
            "share_to_feed": "true" if share_to_feed else "false",
        }
        if cover_url:
            params["cover_url"] = cover_url
        data = self._post(f"{self._creds.ig_user_id}/media", params)
        return data["id"]

    def wait_until_ready(self, container_id: str, timeout_s: float = 300.0,
                         poll_s: float = 5.0) -> None:
        """Pollt den Container-Status, bis er ``FINISHED`` ist (für Videos/Reels).

        Raises PublishError bei ``ERROR`` oder Timeout.
        """
        deadline = time.monotonic() + timeout_s
        while True:
            data = self._get(container_id, {"fields": "status_code,status"})
            status = data.get("status_code")
            if status == "FINISHED":
                return
            if status == "ERROR":
                raise PublishError(f"Media-Container fehlgeschlagen: {data.get('status')}")
            if time.monotonic() > deadline:
                raise PublishError("Timeout beim Warten auf den Media-Container.")
            time.sleep(poll_s)

    def publish(self, container_id: str) -> str:
        """Einen fertigen Container veröffentlichen. Gibt die Media-ID zurück."""
        data = self._post(
            f"{self._creds.ig_user_id}/media_publish",
            {"creation_id": container_id},
        )
        return data["id"]

    # -- High-Level ----------------------------------------------------

    def publish_image(self, image_url: str, caption: str) -> str:
        container = self.create_image_container(image_url, caption)
        return self.publish(container)

    def publish_reel(self, video_url: str, caption: str,
                     cover_url: Optional[str] = None,
                     share_to_feed: bool = True) -> str:
        """Kompletter Reel-Flow: Container → warten → veröffentlichen."""
        container = self.create_reel_container(
            video_url, caption, cover_url=cover_url, share_to_feed=share_to_feed
        )
        self.wait_until_ready(container)
        return self.publish(container)

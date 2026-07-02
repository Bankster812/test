"""Performance-Daten über die Graph API ziehen und Format-Prinzipien gewichten.

So lernt das System datenbasiert, welche *eigenen* Formate funktionieren —
und schlägt für den nächsten Brief die erfolgreichsten Prinzipien vor.

Insights-Doku: https://developers.facebook.com/docs/instagram-platform/insights
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import requests

from .instagram_publisher import IGCredentials, GRAPH_BASE, PublishError
from .content_patterns import ContentPattern


@dataclass
class MediaInsights:
    media_id: str
    reach: int = 0
    likes: int = 0
    comments: int = 0
    saved: int = 0
    shares: int = 0

    def engagement_rate(self) -> float:
        """Saves+Shares+Comments relativ zur Reichweite — der Wert, der Reichweite treibt.

        Instagram bevorzugt Inhalte mit hoher Save-/Share-Rate. Likes allein sind
        ein schwächeres Signal, daher gewichten wir sie niedriger.
        """
        if self.reach <= 0:
            return 0.0
        weighted = self.saved * 1.0 + self.shares * 1.0 + self.comments * 0.5 + self.likes * 0.1
        return weighted / self.reach


def fetch_media_insights(creds: IGCredentials, media_id: str,
                         session: Optional[requests.Session] = None) -> MediaInsights:
    """Insights für ein veröffentlichtes Media-Objekt abrufen."""
    session = session or requests.Session()
    metrics = "reach,likes,comments,saved,shares"
    resp = session.get(
        f"{GRAPH_BASE}/{media_id}/insights",
        params={"metric": metrics, "access_token": creds.access_token},
        timeout=60,
    )
    data = resp.json()
    if resp.status_code >= 400 or "error" in data:
        err = data.get("error", {})
        raise PublishError(f"Insights-Fehler: {err.get('message', resp.status_code)}")

    values = {item["name"]: _first_value(item) for item in data.get("data", [])}
    return MediaInsights(
        media_id=media_id,
        reach=values.get("reach", 0),
        likes=values.get("likes", 0),
        comments=values.get("comments", 0),
        saved=values.get("saved", 0),
        shares=values.get("shares", 0),
    )


def _first_value(item: dict) -> int:
    values = item.get("values", [])
    if values and isinstance(values[0], dict):
        return int(values[0].get("value", 0))
    return 0


def feed_back(pattern: ContentPattern, insights: MediaInsights) -> None:
    """Das Ergebnis eines Posts in die Bewertung seines Format-Prinzips zurückführen."""
    pattern.update_score(insights.engagement_rate())

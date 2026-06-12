"""Orchestriert den Loop: Analyse → Brief → Posten → Messen → Wiederholen.

CLI:
    python -m content_automation.pipeline --demo
    python -m content_automation.pipeline --publish-reel --video-url URL --caption TEXT
"""

from __future__ import annotations

import argparse
import sys

from .content_patterns import PatternLibrary, starter_library
from .brief_generator import generate_brief


def _demo(subject: str, angle: str) -> int:
    """Zeigt Analyse → Brief ohne API-Calls."""
    lib = starter_library()
    print("Verfügbare Format-Prinzipien (Demo, ungewichtet):")
    for p in lib.patterns:
        print(f"  - {p.name}  [{p.topic_space}]  Hook: {p.hook.type.value}")
    print()

    best = lib.best(limit=1)[0]
    brief = generate_brief(best, subject=subject, angle=angle)
    print("=" * 60)
    print(brief.to_text())
    print("=" * 60)
    print("\nNächster Schritt: Eigenes Video nach diesem Brief produzieren,")
    print("dann mit --publish-reel veröffentlichen.")
    return 0


def _publish_reel(video_url: str, caption: str) -> int:
    # Import erst hier, damit die Demo ohne `requests`/Credentials läuft.
    from .instagram_publisher import InstagramPublisher, IGCredentials, PublishError

    try:
        creds = IGCredentials.from_env()
        publisher = InstagramPublisher(creds)
        print("Erstelle Reel-Container und warte auf Verarbeitung ...")
        media_id = publisher.publish_reel(video_url, caption)
        print(f"Veröffentlicht. Media-ID: {media_id}")
        return 0
    except PublishError as exc:
        print(f"Fehler: {exc}", file=sys.stderr)
        return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Legales Content-Automation-Toolkit")
    parser.add_argument("--demo", action="store_true",
                        help="Analyse → Brief ohne API-Calls zeigen")
    parser.add_argument("--subject", default="Espresso zuhause",
                        help="Dein neues Thema/Motiv für den Brief")
    parser.add_argument("--angle", default="",
                        help="Optionaler Blickwinkel, der den Hook konkretisiert")
    parser.add_argument("--publish-reel", action="store_true",
                        help="Ein eigenes Reel über die Graph API veröffentlichen")
    parser.add_argument("--video-url", help="Öffentliche URL zu DEINEM eigenen Video")
    parser.add_argument("--caption", default="", help="Caption-Text")
    args = parser.parse_args(argv)

    if args.publish_reel:
        if not args.video_url:
            parser.error("--publish-reel benötigt --video-url")
        return _publish_reel(args.video_url, args.caption)

    # Standard: Demo.
    return _demo(args.subject, args.angle)


if __name__ == "__main__":
    raise SystemExit(main())

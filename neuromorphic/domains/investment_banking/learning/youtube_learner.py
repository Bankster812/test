"""
YouTubeLearner — Extract and ingest IB content from YouTube
============================================================
Downloads transcripts from IB-focused YouTube channels and feeds them
into the brain as structured FinancialChunks.

No audio processing needed — YouTube auto-captions and manual transcripts
are fetched via yt-dlp (if installed) or the YouTube transcript API fallback.

Top IB channels hardcoded as defaults:
  - Aswath Damodaran (NYU, valuation master)
  - Wall Street Prep
  - Mergers & Inquisitions / Breaking Into Wall Street
  - Goldman Sachs / Morgan Stanley public content
  - Bloomberg Markets
"""

from __future__ import annotations

import re
import time
import json
import logging
from dataclasses import dataclass, field
from urllib.request import urlopen, Request
from urllib.parse import quote

logger = logging.getLogger("ib_brain.youtube_learner")


# ---------------------------------------------------------------------------
# Curated IB YouTube channel IDs
# ---------------------------------------------------------------------------
IB_YOUTUBE_CHANNELS = {
    "damodaran":           "UCLvnJL8htRR1T9cbpqa9SKw",   # Aswath Damodaran
    "wall_st_prep":        "UCYHWKRMwT3P5Z0R4LBE97vg",   # Wall Street Prep
    "biws":                "UCaxGCMhbNLHvnCFJVdX0wjg",   # Breaking Into Wall Street
    "bloomberg_markets":   "UCIALMKvObZNtJ6AmdCLP7Lg",   # Bloomberg
    "financial_education": "UCnMn36GT_H0X-w5_ckLtlgQ",   # Jeremy explains finance
}

@dataclass
class VideoTranscript:
    video_id:   str
    title:      str
    channel:    str
    transcript: str
    duration_s: int = 0


class YouTubeLearner:
    """
    Fetches YouTube video transcripts for IB brain ingestion.

    Strategy (in order of preference):
    1. yt-dlp (if installed): most reliable, gets auto-subs and manual subs
    2. youtube-transcript-api (if installed): clean JSON transcript
    3. Raw YouTube HTML scraping fallback: extracts auto-generated captions

    Parameters
    ----------
    config : ib_config module
    max_videos_per_session : int
    """

    def __init__(self, config, max_videos_per_session: int = 10):
        self.cfg       = config
        self.max_videos = max_videos_per_session
        self._ytdlp    = self._check_ytdlp()
        self._yta_api  = self._check_yta_api()

    # ------------------------------------------------------------------
    # Main interface
    # ------------------------------------------------------------------

    def fetch_channel_transcripts(
        self,
        channel_name: str,
        max_videos: int | None = None,
    ) -> list[VideoTranscript]:
        """
        Fetch transcripts from a named channel (from IB_YOUTUBE_CHANNELS).
        """
        channel_id = IB_YOUTUBE_CHANNELS.get(channel_name.lower())
        if not channel_id:
            logger.warning(f"Unknown channel: {channel_name}")
            return []

        video_ids = self._get_channel_video_ids(channel_id, max_videos or self.max_videos)
        transcripts = []
        for vid_id in video_ids[:self.max_videos]:
            t = self.fetch_transcript(vid_id, channel_name)
            if t:
                transcripts.append(t)
            time.sleep(1.0)
        return transcripts

    def fetch_transcript(self, video_id: str, channel: str = "youtube") -> VideoTranscript | None:
        """Fetch transcript for a single video ID."""
        # Try yt-dlp first
        if self._ytdlp:
            t = self._fetch_ytdlp(video_id, channel)
            if t:
                return t

        # Try youtube-transcript-api
        if self._yta_api:
            t = self._fetch_yta_api(video_id, channel)
            if t:
                return t

        # Fallback: scrape YouTube HTML for auto-captions
        return self._fetch_html_fallback(video_id, channel)

    def fetch_url(self, youtube_url: str) -> VideoTranscript | None:
        """Fetch transcript from a YouTube URL."""
        vid_id = self._extract_video_id(youtube_url)
        if not vid_id:
            return None
        return self.fetch_transcript(vid_id)

    # ------------------------------------------------------------------
    # Fetching strategies
    # ------------------------------------------------------------------

    def _fetch_ytdlp(self, video_id: str, channel: str) -> VideoTranscript | None:
        """Use yt-dlp to download auto-subtitles as text."""
        import yt_dlp
        import tempfile, os
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                opts = {
                    "skip_download":    True,
                    "writeautomaticsub": True,
                    "writesubtitles":   True,
                    "subtitleslangs":   ["en"],
                    "subtitlesformat":  "json3",
                    "outtmpl":          os.path.join(tmpdir, "%(id)s.%(ext)s"),
                    "quiet":            True,
                    "no_warnings":      True,
                }
                url = f"https://www.youtube.com/watch?v={video_id}"
                with yt_dlp.YoutubeDL(opts) as ydl:
                    info   = ydl.extract_info(url, download=True)
                    title  = info.get("title", video_id)
                    duration = info.get("duration", 0)

                # Read the subtitle file
                sub_file = os.path.join(tmpdir, f"{video_id}.en.json3")
                if not os.path.exists(sub_file):
                    # Try .vtt
                    sub_file = os.path.join(tmpdir, f"{video_id}.en.vtt")
                if os.path.exists(sub_file):
                    with open(sub_file, "r", encoding="utf-8") as f:
                        raw = f.read()
                    text = self._parse_subtitle_file(raw, sub_file)
                    if text and len(text) > 50:
                        return VideoTranscript(video_id, title, channel, text, duration)
        except Exception as e:
            logger.debug(f"yt-dlp failed for {video_id}: {e}")
        return None

    def _fetch_yta_api(self, video_id: str, channel: str) -> VideoTranscript | None:
        """Use youtube-transcript-api to get clean JSON transcript."""
        from youtube_transcript_api import YouTubeTranscriptApi
        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
            text = " ".join(entry["text"] for entry in transcript_list)
            return VideoTranscript(video_id, video_id, channel, text)
        except Exception as e:
            logger.debug(f"youtube-transcript-api failed for {video_id}: {e}")
        return None

    def _fetch_html_fallback(self, video_id: str, channel: str) -> VideoTranscript | None:
        """
        Scrape YouTube HTML page and extract any embedded caption data.
        This is a best-effort fallback.
        """
        try:
            url = f"https://www.youtube.com/watch?v={video_id}"
            req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(req, timeout=10) as resp:
                html = resp.read().decode("utf-8", errors="ignore")

            # Extract title
            title_m = re.search(r'"title":"([^"]+)"', html)
            title   = title_m.group(1) if title_m else video_id

            # Extract any embedded transcript text (timedtext)
            text_parts = re.findall(r'"text":"([^"]+)"', html)
            text = " ".join(text_parts[:500])   # limit to first 500 segments

            if len(text) > 100:
                return VideoTranscript(video_id, title, channel,
                                       self._clean_transcript(text))
        except Exception as e:
            logger.debug(f"HTML fallback failed for {video_id}: {e}")
        return None

    # ------------------------------------------------------------------
    # Channel video discovery
    # ------------------------------------------------------------------

    def _get_channel_video_ids(self, channel_id: str, max_n: int) -> list[str]:
        """Get recent video IDs from a channel via RSS feed (no API key)."""
        rss_url = f"https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"
        try:
            req = Request(rss_url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(req, timeout=10) as resp:
                xml = resp.read().decode("utf-8", errors="ignore")
            ids = re.findall(r'<yt:videoId>([^<]+)</yt:videoId>', xml)
            return ids[:max_n]
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _parse_subtitle_file(self, raw: str, path: str) -> str:
        """Parse .json3 or .vtt subtitle file to plain text."""
        if path.endswith(".json3"):
            try:
                data   = json.loads(raw)
                events = data.get("events", [])
                segs   = []
                for ev in events:
                    for seg in ev.get("segs", []):
                        t = seg.get("utf8", "")
                        if t and t.strip() != "\n":
                            segs.append(t.strip())
                return " ".join(segs)
            except Exception:
                pass
        # VTT
        text = re.sub(r"\d{2}:\d{2}:\d{2}\.\d{3} --> [^\n]+\n", "", raw)
        text = re.sub(r"<[^>]+>", "", text)
        text = re.sub(r"\n+", " ", text)
        return text.strip()

    def _clean_transcript(self, text: str) -> str:
        """Clean up extracted transcript text."""
        text = text.replace("\\n", " ").replace("\\u0026", "&")
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @staticmethod
    def _extract_video_id(url: str) -> str | None:
        patterns = [
            r"youtube\.com/watch\?v=([A-Za-z0-9_-]{11})",
            r"youtu\.be/([A-Za-z0-9_-]{11})",
            r"youtube\.com/embed/([A-Za-z0-9_-]{11})",
        ]
        for p in patterns:
            m = re.search(p, url)
            if m:
                return m.group(1)
        return None

    @staticmethod
    def _check_ytdlp() -> bool:
        try:
            import yt_dlp  # noqa: F401
            return True
        except ImportError:
            return False

    @staticmethod
    def _check_yta_api() -> bool:
        try:
            import youtube_transcript_api  # noqa: F401
            return True
        except ImportError:
            return False

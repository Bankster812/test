"""
ContinuousLearningDaemon — 24/7 background learning
=====================================================
Runs in a daemon thread. Every `interval_minutes`:
  1. Fetches new IB content from the web
  2. Optionally fetches YouTube transcripts
  3. Feeds everything into the IBBrain ingestion pipeline
  4. Reports learning stats

The brain's synaptic weights update permanently through STDP.
Stop it any time — progress is preserved in the weights.

Usage
-----
    daemon = ContinuousLearningDaemon(brain, interval_minutes=30)
    daemon.start()           # non-blocking — returns immediately
    daemon.status()          # check what it's learning
    daemon.stop()            # graceful shutdown
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field

logger = logging.getLogger("ib_brain.daemon")


@dataclass
class LearningStats:
    sessions_completed:  int = 0
    articles_ingested:   int = 0
    videos_ingested:     int = 0
    chunks_processed:    int = 0
    last_session_at:     float = 0.0
    last_source:         str = ""
    errors:              list[str] = field(default_factory=list)

    def __str__(self) -> str:
        last = time.strftime("%H:%M:%S", time.localtime(self.last_session_at)) \
               if self.last_session_at else "never"
        return (
            f"Sessions: {self.sessions_completed}  |  "
            f"Articles: {self.articles_ingested}  |  "
            f"Videos: {self.videos_ingested}  |  "
            f"Chunks: {self.chunks_processed}  |  "
            f"Last run: {last}"
        )


class ContinuousLearningDaemon:
    """
    Background daemon that keeps the brain learning around the clock.

    Parameters
    ----------
    brain        : IBBrain instance
    interval_minutes : float — how often to run a learning session
    topics       : list[str] — web search topics
    youtube_channels : list[str] — channel names from IB_YOUTUBE_CHANNELS
    enable_web   : bool — enable web scraping
    enable_youtube : bool — enable YouTube ingestion
    """

    def __init__(
        self,
        brain,
        interval_minutes:  float      = 30.0,
        topics:            list[str]  | None = None,
        youtube_channels:  list[str]  | None = None,
        enable_web:        bool       = True,
        enable_youtube:    bool       = False,   # off by default (requires yt-dlp)
    ):
        self.brain             = brain
        self.interval          = interval_minutes * 60.0
        self.topics            = topics
        self.yt_channels       = youtube_channels or ["damodaran", "wall_st_prep"]
        self.enable_web        = enable_web
        self.enable_youtube    = enable_youtube
        self.stats             = LearningStats()

        self._stop_event       = threading.Event()
        self._thread: threading.Thread | None = None
        self._paused           = False

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------

    def start(self):
        """Start the daemon thread. Non-blocking."""
        if self._thread and self._thread.is_alive():
            logger.info("Daemon already running")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop,
            daemon=True,        # thread dies with main process
            name="IBLearningDaemon",
        )
        self._thread.start()
        logger.info(
            f"Continuous learning daemon started "
            f"(interval: {self.interval/60:.0f} min, "
            f"web: {self.enable_web}, youtube: {self.enable_youtube})"
        )

    def stop(self):
        """Gracefully stop the daemon."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10.0)
        logger.info("Daemon stopped")

    def pause(self):
        """Pause learning without stopping the thread."""
        self._paused = True

    def resume(self):
        """Resume learning."""
        self._paused = False

    def run_once(self):
        """Run a single learning session immediately (blocking)."""
        self._run_session()

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def status(self) -> str:
        state = "running" if self.is_running else "stopped"
        if self._paused:
            state = "paused"
        return f"[{state}] {self.stats}"

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _loop(self):
        """Main daemon loop — runs forever until stop_event set."""
        # Run first session immediately on start
        self._run_session()

        while not self._stop_event.is_set():
            # Sleep in small increments so stop() is responsive
            sleep_remaining = self.interval
            while sleep_remaining > 0 and not self._stop_event.is_set():
                chunk = min(5.0, sleep_remaining)
                time.sleep(chunk)
                sleep_remaining -= chunk

            if self._stop_event.is_set():
                break
            if not self._paused:
                self._run_session()

    def _run_session(self):
        """One complete learning session."""
        session_start = time.time()
        logger.info("Learning session starting...")

        # Import here to avoid circular dependency at module level
        from neuromorphic.domains.investment_banking.learning.web_learner import WebLearner
        from neuromorphic.domains.investment_banking.learning.youtube_learner import YouTubeLearner
        from neuromorphic.domains.investment_banking.encoders.document_ingestion import DocumentIngestion

        ingestion = DocumentIngestion(self.brain.cfg)
        total_chunks = 0

        # ---- Web learning ----
        if self.enable_web:
            try:
                web_learner = WebLearner(
                    self.brain.cfg,
                    max_articles_per_session=15,
                    request_delay=2.0,
                )
                articles = web_learner.fetch_session(topics=self.topics)
                for article in articles:
                    chunks = ingestion.extract_from_string(
                        article.text,
                        source=article.url,
                        chunk_type="web",
                    )
                    if chunks:
                        self.brain._ingest_chunks(chunks, reward=0.6)
                        total_chunks += len(chunks)
                self.stats.articles_ingested += len(articles)
                self.stats.last_source = f"web ({len(articles)} articles)"
                logger.info(f"  Web: ingested {len(articles)} articles → {total_chunks} chunks")
            except Exception as e:
                err = f"Web learning error: {e}"
                logger.warning(err)
                self.stats.errors.append(err)

        # ---- YouTube learning ----
        if self.enable_youtube:
            try:
                yt = YouTubeLearner(self.brain.cfg, max_videos_per_session=3)
                vid_count = 0
                for channel in self.yt_channels[:2]:  # max 2 channels per session
                    transcripts = yt.fetch_channel_transcripts(channel, max_videos=2)
                    for t in transcripts:
                        chunks = ingestion.extract_from_string(
                            t.transcript,
                            source=f"youtube/{t.video_id}",
                            chunk_type="transcript",
                        )
                        if chunks:
                            self.brain._ingest_chunks(chunks, reward=0.65)
                            total_chunks += len(chunks)
                            vid_count += 1
                self.stats.videos_ingested += vid_count
                logger.info(f"  YouTube: ingested {vid_count} videos")
            except Exception as e:
                err = f"YouTube learning error: {e}"
                logger.warning(err)
                self.stats.errors.append(err)

        self.stats.sessions_completed  += 1
        self.stats.chunks_processed    += total_chunks
        self.stats.last_session_at      = time.time()

        elapsed = time.time() - session_start
        logger.info(
            f"Session complete: {total_chunks} chunks in {elapsed:.1f}s  |  {self.stats}"
        )

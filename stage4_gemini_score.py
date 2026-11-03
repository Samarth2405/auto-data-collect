"""
Stage 4 — AI Crowd Scoring (Gemini 3.5 Flash)
Scores each candidate frame from Stage 3 on a 1-10 crowd density scale.
Only frames that survived the local YOLO pre-filter reach Gemini.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel

from stage3_prefilter import FrameScore, PrefilteredVideo

load_dotenv()

# ── Constants ────────────────────────────────────────────────────────────────
GEMINI_MODEL = "gemini-3.5-flash"
CROWD_THRESHOLD = 7                 # score >= this counts as high-density
FREE_TIER_RPM = 15                  # Gemini free-tier rate limit
DEFAULT_WORKERS = 1                 # sequential; raise with --workers on a paid tier
TEMPERATURE = 0.2                  # low variance for repeatable scoring
MAX_RETRIES = 5


# ── Terminal vs. transient failures ─────────────────────────────────────────
class QuotaExhausted(RuntimeError):
    """
    The API is unusable for reasons no retry will fix — spend cap, daily quota,
    bad credentials. Account-wide rather than per-video, so the batch layer
    stops the run instead of reproducing it on every remaining frame.
    """


# Matched against the exception text, lowercased with underscores folded to
# spaces so "PERMISSION_DENIED" and "permission denied" both hit. A plain
# per-minute 429 is absent on purpose: it is transient and should back off.
_TERMINAL_ERROR_MARKERS = (
    "spending cap",         # AI Studio project spend cap
    "spend cap",
    "billing",              # billing disabled or account problem
    "per day",              # daily quota, as opposed to per-minute
    "daily limit",
    "api key not valid",
    "api key invalid",
    "permission denied",
    "unauthenticated",
    "consumer suspended",
)


def _is_terminal(exc: Exception) -> bool:
    """True when retrying cannot help, so the caller should stop rather than back off."""
    text = str(exc).lower().replace("_", " ")
    return any(marker in text for marker in _TERMINAL_ERROR_MARKERS)


# ── Structured response schema ───────────────────────────────────────────────
class CameraVantage(str, Enum):
    """Height the shot was taken from. The dataset wants anything but ground."""

    GROUND = "ground"       # camera at human height, in or beside the crowd
    ELEVATED = "elevated"   # above the crowd: balcony, terrace, upper floor, bridge
    AERIAL = "aerial"       # drone or helicopter, above roofline
    UNKNOWN = "unknown"     # genuinely cannot tell


ELEVATED_VANTAGES = (CameraVantage.ELEVATED, CameraVantage.AERIAL)


class CrowdScoreResponse(BaseModel):
    score: int                      # expected 1-10
    vantage: CameraVantage          # camera height, asked for in the same call
    reasoning: str                  # one-sentence explanation


# ── Data contract ────────────────────────────────────────────────────────────
@dataclass
class GeminiScore:
    frame: FrameScore
    crowd_score: int
    reasoning: str
    vantage: str = CameraVantage.UNKNOWN.value

    @property
    def is_high_density(self) -> bool:
        return self.crowd_score >= CROWD_THRESHOLD

    @property
    def is_elevated(self) -> bool:
        """True for shots taken above the crowd — the dataset's height criterion."""
        return self.vantage in (CameraVantage.ELEVATED.value, CameraVantage.AERIAL.value)


@dataclass
class ScoredVideo:
    prefiltered: PrefilteredVideo
    gemini_scores: list[GeminiScore] = field(default_factory=list)

    @property
    def high_density_frames(self) -> list[GeminiScore]:
        return [s for s in self.gemini_scores if s.is_high_density]

    @property
    def elevated_frames(self) -> list[GeminiScore]:
        """High-density AND shot from above — what the dataset actually keeps."""
        return [s for s in self.gemini_scores if s.is_high_density and s.is_elevated]


# ── Prompt ───────────────────────────────────────────────────────────────────
PROMPT = """Rate this image on two independent axes.

1. CROWD DENSITY, 1-10:
- 1-3: Empty or sparse — few individuals, ample personal space
- 4-6: Moderate — visible groups, movement remains easy
- 7-8: High density — packed but individuals still distinguishable
- 9-10: Extreme — crowd surge, no personal space

2. CAMERA VANTAGE — the height the camera was at, not how far away the
   subject is. A zoomed-in shot from a rooftop is still elevated; a wide
   shot from the pavement is still ground.
- "ground": camera at roughly human height. You see faces and torsos from
  the front, people overlap and occlude each other, the horizon and sky sit
  in the upper part of the frame.
- "elevated": camera clearly above the crowd but below roofline — balcony,
  terrace, upper floor, staircase, bridge, flyover. You look down on heads
  and shoulders, people foreshorten, the ground plane fills much of the
  frame, little or no sky.
- "aerial": drone or helicopter, above the rooftops. Buildings seen from
  above, people are small marks on the ground.
- "unknown": genuinely cannot tell.

Judge vantage from geometry — viewing angle onto the ground plane, how much
sky is visible, whether you see tops of heads — never from image quality.
Handle low-light scenes and wide-angle distortion. When unsure on density,
err lower. Return the score, the vantage, and a one-sentence reason."""


# ── Lazy-init client ─────────────────────────────────────────────────────────
_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY environment variable not set")
        _client = genai.Client(api_key=api_key)
    return _client


# ── Rate limiter (simple spacing) ────────────────────────────────────────────
_last_call_at: float = 0.0
_rate_lock = threading.Lock()


def _rate_limit(rpm: int) -> None:
    # Held while sleeping on purpose: workers queue for the next slot, so the
    # spacing is global rather than per-thread.
    global _last_call_at
    interval = 60.0 / rpm
    with _rate_lock:
        elapsed = time.time() - _last_call_at
        if elapsed < interval:
            time.sleep(interval - elapsed)
        _last_call_at = time.time()


# ── Core call ────────────────────────────────────────────────────────────────
def _score_frame(frame_path: Path, rpm: int) -> CrowdScoreResponse:
    client = _get_client()
    image_bytes = frame_path.read_bytes()

    for attempt in range(MAX_RETRIES):
        _rate_limit(rpm)
        try:
            resp = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                    PROMPT,
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=CrowdScoreResponse,
                    temperature=TEMPERATURE,
                ),
            )
            if resp.parsed is None:
                # Safety block or malformed schema output — not retryable.
                # Emit a floor-score sentinel so the frame drops out of the
                # high-density set without crashing the pipeline.
                finish = None
                if resp.candidates:
                    finish = getattr(resp.candidates[0], "finish_reason", None)
                return CrowdScoreResponse(
                    score=1,
                    vantage=CameraVantage.UNKNOWN,
                    reasoning=f"no parsed response (finish={finish})",
                )
            return resp.parsed
        except Exception as e:
            if _is_terminal(e):
                # No point spending MAX_RETRIES x every remaining frame on this.
                raise QuotaExhausted(str(e)) from e
            if attempt == MAX_RETRIES - 1:
                raise
            # per-minute 429s and transient 5xx — exponential backoff
            wait = 2 ** attempt
            print(f"  [stage4] retry {attempt + 1}/{MAX_RETRIES} in {wait}s: {e}")
            time.sleep(wait)
    raise RuntimeError("unreachable")


# ── Cache ────────────────────────────────────────────────────────────────────
def _load_cache(cache_path: Path, model: str) -> dict[str, dict]:
    """Reuse cached scores only if the same model produced them.

    Invalidates on `model` change, the way Stage 3 invalidates on YOLO model and
    confidence: two models rate the same frame on subtly different scales, so
    blending them would make one dataset out of two. Thresholds stay a
    post-processing decision and are still re-derived at load time.
    """
    if not cache_path.exists():
        return {}
    cached = json.loads(cache_path.read_text())
    if cached.get("model") != model:
        print(f"[stage4] {cache_path.parent.name}: discarding "
              f"{len(cached.get('scores', []))} scores from "
              f"{cached.get('model', 'unknown')} — now scoring with {model}")
        return {}
    return {s["frame_path"]: s for s in cached.get("scores", [])}


def _save_cache(cache_path: Path, video_id: str, scores: list[GeminiScore]) -> None:
    cache_path.write_text(json.dumps({
        "video_id": video_id,
        "model": GEMINI_MODEL,
        "scores": [
            {
                "frame_path": str(s.frame.path),
                "timestamp": s.frame.timestamp,
                "person_count": s.frame.person_count,
                "crowd_score": s.crowd_score,
                "vantage": s.vantage,
                "reasoning": s.reasoning,
            }
            for s in scores
        ],
    }, indent=2))


def _checkpoint(cache_path: Path, video_id: str, scores: list[GeminiScore]) -> None:
    """Persist scores in timestamp order; used by the normal and failure exits alike."""
    _save_cache(cache_path, video_id, sorted(scores, key=lambda s: s.frame.timestamp))


# ── Public API ───────────────────────────────────────────────────────────────
def score_video(
    prefiltered: PrefilteredVideo,
    rpm: int = FREE_TIER_RPM,
    save_every: int = 10,
    workers: int = DEFAULT_WORKERS,
) -> ScoredVideo | None:
    """
    Score every candidate frame; reuse cached scores for frames already seen.

    Returns None if this video cannot be scored, so a single bad video drops out
    instead of taking the batch with it. Raises QuotaExhausted when the API
    itself is spent — that is not this video's fault and the caller must stop.
    Frames scored before either exit stay in the cache for the next run.
    """
    video = prefiltered.video
    cache_path = video.frames_dir.parent / "gemini_scores.json"
    candidates = prefiltered.candidate_frames

    cached = _load_cache(cache_path, GEMINI_MODEL)
    scores: list[GeminiScore] = []
    to_score: list[FrameScore] = []

    for frame in candidates:
        key = str(frame.path)
        if key in cached:
            c = cached[key]
            scores.append(GeminiScore(
                frame=frame,
                crowd_score=c["crowd_score"],
                reasoning=c["reasoning"],
                # pre-vantage caches predate this field
                vantage=c.get("vantage", CameraVantage.UNKNOWN.value),
            ))
        else:
            to_score.append(frame)

    if cached and scores:
        print(f"[stage4] {video.video_id}: {len(scores)} cached, {len(to_score)} new")

    if to_score:
        pool_note = f", {workers} workers" if workers > 1 else ""
        print(f"[stage4] {video.video_id}: scoring {len(to_score)} frames "
              f"at {rpm} RPM{pool_note}")
        _get_client()   # build the client once, before workers race to
        quota: QuotaExhausted | None = None
        failure: Exception | None = None
        done = 0
        pool = ThreadPoolExecutor(max_workers=workers)
        try:
            futures = {pool.submit(_score_frame, f.path, rpm): f for f in to_score}
            for fut in as_completed(futures):
                if fut.cancelled():
                    continue
                try:
                    parsed = fut.result()
                except QuotaExhausted as exc:
                    quota = quota or exc
                except Exception as exc:
                    failure = failure or exc
                else:
                    done += 1
                    scores.append(GeminiScore(
                        frame=futures[fut],
                        # Clamp: Gemini occasionally ignores the 1-10 guidance
                        crowd_score=max(1, min(10, parsed.score)),
                        reasoning=parsed.reasoning,
                        vantage=parsed.vantage.value,
                    ))
                    # Incremental checkpoint — survives Ctrl-C
                    if done % save_every == 0 or done == len(to_score):
                        _checkpoint(cache_path, video.video_id, scores)
                        print(f"  [stage4] {done}/{len(to_score)} scored")
                if quota or failure:
                    # Stop queueing; frames already in flight still bank.
                    for pending in futures:
                        pending.cancel()
        finally:
            pool.shutdown(wait=True, cancel_futures=True)

        if quota is not None:
            _checkpoint(cache_path, video.video_id, scores)
            print(f"[stage4] {video.video_id}: quota gone after {done}/{len(to_score)} "
                  f"new frames — {len(scores)} banked to gemini_scores.json")
            raise quota
        if failure is not None:
            _checkpoint(cache_path, video.video_id, scores)
            print(f"[stage4] {video.video_id}: giving up after {done}/{len(to_score)} "
                  f"new frames — {failure}")
            return None

    # Downstream segment builder needs monotonic timestamps
    scores.sort(key=lambda s: s.frame.timestamp)
    _save_cache(cache_path, video.video_id, scores)

    result = ScoredVideo(prefiltered=prefiltered, gemini_scores=scores)
    n_high = len(result.high_density_frames)
    n_elev = len(result.elevated_frames)
    vantages = Counter(s.vantage for s in scores)
    print(f"[stage4] {video.video_id}: {n_high}/{len(scores)} high-density "
          f"(score >= {CROWD_THRESHOLD}), {n_elev} of those elevated")
    print(f"[stage4] {video.video_id}: vantage mix — "
          + ", ".join(f"{v}={n}" for v, n in vantages.most_common()))
    return result


def run_stage4(
    prefiltered_videos: list[PrefilteredVideo],
    rpm: int = FREE_TIER_RPM,
    workers: int = DEFAULT_WORKERS,
) -> list[ScoredVideo]:
    """
    Score all candidate frames across a batch of videos.

    One video failing drops that video only. Quota exhaustion is account-wide,
    so it ends the batch and returns what finished; cached partial scores make
    the next run resume where this one stopped.
    """
    results: list[ScoredVideo] = []
    for idx, pv in enumerate(prefiltered_videos):
        try:
            scored = score_video(pv, rpm, workers=workers)
        except QuotaExhausted as exc:
            print(f"[stage4] stopping batch — {exc}")
            print(f"[stage4] {len(prefiltered_videos) - idx} of {len(prefiltered_videos)} "
                  f"video(s) unscored; rerun to resume from cache")
            break
        if scored is None:
            print(f"[stage4] {pv.video.video_id}: dropped from batch")
            continue
        results.append(scored)

    total = sum(len(sv.gemini_scores) for sv in results)
    high = sum(len(sv.high_density_frames) for sv in results)
    elev = sum(len(sv.elevated_frames) for sv in results)
    print(f"[stage4] batch done — {high}/{total} frames high-density, "
          f"{elev} high-density AND elevated, "
          f"across {len(results)}/{len(prefiltered_videos)} videos")
    return results


# ── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    from stage2_download_prepare import run_stage2
    from stage3_prefilter import run_stage3

    parser = argparse.ArgumentParser(description="Stage 4: Gemini crowd scoring")
    parser.add_argument("urls", nargs="+", help="YouTube video URLs")
    parser.add_argument("--out", default="./crowd_output")
    parser.add_argument("--min-persons", type=int, default=5)
    parser.add_argument("--rpm", type=int, default=FREE_TIER_RPM,
                        help="Gemini requests per minute (free tier = 15)")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help="Concurrent Gemini calls; RPM still caps the total (paid tier)")
    args = parser.parse_args()

    videos = run_stage2(args.urls, output_dir=args.out)
    prefiltered = run_stage3(videos, person_count_min=args.min_persons)
    scored = run_stage4(prefiltered, rpm=args.rpm, workers=args.workers)
    for sv in scored:
        print(f"  {sv.prefiltered.video.video_id}: "
              f"{len(sv.high_density_frames)} high-density frames")

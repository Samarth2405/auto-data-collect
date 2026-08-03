"""
Stage 4 — AI Crowd Scoring (Gemini 2.5 Flash)
Scores each candidate frame from Stage 3 on a 1-10 crowd density scale.
Only frames that survived the local YOLO pre-filter reach Gemini.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel

from stage3_prefilter import FrameScore, PrefilteredVideo

load_dotenv()

# ── Constants ────────────────────────────────────────────────────────────────
GEMINI_MODEL = "gemini-flash-latest"
CROWD_THRESHOLD = 7                 # score >= this counts as high-density
FREE_TIER_RPM = 15                  # Gemini free-tier rate limit
TEMPERATURE = 0.2                   # low variance for repeatable scoring
MAX_RETRIES = 5


# ── Structured response schema ───────────────────────────────────────────────
class CrowdScoreResponse(BaseModel):
    score: int                      # expected 1-10
    reasoning: str                  # one-sentence explanation


# ── Data contract ────────────────────────────────────────────────────────────
@dataclass
class GeminiScore:
    frame: FrameScore
    crowd_score: int
    reasoning: str

    @property
    def is_high_density(self) -> bool:
        return self.crowd_score >= CROWD_THRESHOLD


@dataclass
class ScoredVideo:
    prefiltered: PrefilteredVideo
    gemini_scores: list[GeminiScore] = field(default_factory=list)

    @property
    def high_density_frames(self) -> list[GeminiScore]:
        return [s for s in self.gemini_scores if s.is_high_density]


# ── Prompt ───────────────────────────────────────────────────────────────────
PROMPT = """Rate the crowd density in this image on a 1-10 scale.

- 1-3: Empty or sparse — few individuals, ample personal space
- 4-6: Moderate — visible groups, movement remains easy
- 7-8: High density — packed but individuals still distinguishable
- 9-10: Extreme — crowd surge, no personal space

Handle aerial shots, low-light scenes, and wide-angle distortion.
When unsure, err lower. Return the score and a one-sentence reason."""


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


def _rate_limit(rpm: int) -> None:
    global _last_call_at
    interval = 60.0 / rpm
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
                    reasoning=f"no parsed response (finish={finish})",
                )
            return resp.parsed
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                raise
            # 429s and transient 5xx — exponential backoff
            wait = 2 ** attempt
            print(f"  [stage4] retry {attempt + 1}/{MAX_RETRIES} in {wait}s: {e}")
            time.sleep(wait)
    raise RuntimeError("unreachable")


# ── Cache ────────────────────────────────────────────────────────────────────
def _load_cache(cache_path: Path) -> dict[str, dict]:
    if not cache_path.exists():
        return {}
    cached = json.loads(cache_path.read_text())
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
                "reasoning": s.reasoning,
            }
            for s in scores
        ],
    }, indent=2))


# ── Public API ───────────────────────────────────────────────────────────────
def score_video(
    prefiltered: PrefilteredVideo,
    rpm: int = FREE_TIER_RPM,
    save_every: int = 10,
) -> ScoredVideo:
    """Score every candidate frame; reuse cached scores for frames already seen."""
    video = prefiltered.video
    cache_path = video.frames_dir.parent / "gemini_scores.json"
    candidates = prefiltered.candidate_frames

    cached = _load_cache(cache_path)
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
            ))
        else:
            to_score.append(frame)

    if cached and scores:
        print(f"[stage4] {video.video_id}: {len(scores)} cached, {len(to_score)} new")

    if to_score:
        print(f"[stage4] {video.video_id}: scoring {len(to_score)} frames at {rpm} RPM")
        for i, frame in enumerate(to_score, 1):
            parsed = _score_frame(frame.path, rpm)
            scores.append(GeminiScore(
                frame=frame,
                # Clamp: Gemini occasionally ignores the 1-10 guidance
                crowd_score=max(1, min(10, parsed.score)),
                reasoning=parsed.reasoning,
            ))
            # Incremental checkpoint — survives Ctrl-C
            if i % save_every == 0 or i == len(to_score):
                _save_cache(cache_path, video.video_id, scores)
                print(f"  [stage4] {i}/{len(to_score)} scored")

    # Downstream segment builder needs monotonic timestamps
    scores.sort(key=lambda s: s.frame.timestamp)
    _save_cache(cache_path, video.video_id, scores)

    result = ScoredVideo(prefiltered=prefiltered, gemini_scores=scores)
    n_high = len(result.high_density_frames)
    print(f"[stage4] {video.video_id}: {n_high}/{len(scores)} high-density (score >= {CROWD_THRESHOLD})")
    return result


def run_stage4(
    prefiltered_videos: list[PrefilteredVideo],
    rpm: int = FREE_TIER_RPM,
) -> list[ScoredVideo]:
    """Score all candidate frames across a batch of videos."""
    results = [score_video(pv, rpm) for pv in prefiltered_videos]
    total = sum(len(sv.gemini_scores) for sv in results)
    high = sum(len(sv.high_density_frames) for sv in results)
    print(f"[stage4] batch done — {high}/{total} frames marked high-density")
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
    args = parser.parse_args()

    videos = run_stage2(args.urls, output_dir=args.out)
    prefiltered = run_stage3(videos, person_count_min=args.min_persons)
    scored = run_stage4(prefiltered, rpm=args.rpm)
    for sv in scored:
        print(f"  {sv.prefiltered.video.video_id}: "
              f"{len(sv.high_density_frames)} high-density frames")

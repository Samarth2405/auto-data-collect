"""
Stage 6 — Segment Builder
Merges Stage 4's high-density frames into contiguous time segments.
Two consecutive hits stay in the same segment while their gap is
<= MIN_SEGMENT_SECS. Aggregates score, person-count, and reasoning
samples per segment; emits YouTube deep-links.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

from stage2_download_prepare import VideoData
from stage4_gemini_score import CROWD_THRESHOLD, GeminiScore, ScoredVideo

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# ── Constants ────────────────────────────────────────────────────────────────
MIN_SEGMENT_SECS = 3.0          # gap > this starts a new segment
MAX_REASONING_SAMPLES = 3       # per-segment reasoning strings retained


# ── Data contract ────────────────────────────────────────────────────────────
@dataclass
class Segment:
    start: float                # seconds into video
    end: float
    start_fmt: str              # "0:45"
    end_fmt: str                # "1:52"
    frame_count: int
    avg_crowd_score: float
    peak_crowd_score: int
    person_count_avg: float
    person_count_peak: int
    yt_link: str
    reasoning_samples: list[str] = field(default_factory=list)


@dataclass
class VideoSegments:
    scored: ScoredVideo
    segments: list[Segment] = field(default_factory=list)

    @property
    def video(self) -> VideoData:
        return self.scored.prefiltered.video


# ── Internal ─────────────────────────────────────────────────────────────────
def _fmt_time(secs: float) -> str:
    total = int(secs)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _yt_link(url: str, start_secs: float) -> str:
    if not url:
        return ""
    base = url.split("#", 1)[0]
    sep = "&" if "?" in base else "?"
    return f"{base}{sep}t={int(start_secs)}s"


def _group_by_gap(hits: list[GeminiScore], max_gap: float) -> list[list[GeminiScore]]:
    """Split a time-ordered list of hits into runs where each gap <= max_gap."""
    if not hits:
        return []
    groups: list[list[GeminiScore]] = [[hits[0]]]
    for h in hits[1:]:
        prev_ts = groups[-1][-1].frame.timestamp
        if h.frame.timestamp - prev_ts <= max_gap:
            groups[-1].append(h)
        else:
            groups.append([h])
    return groups


def _aggregate(group: list[GeminiScore], url: str, fps: float) -> Segment:
    scores = [g.crowd_score for g in group]
    counts = [g.frame.person_count for g in group]
    start = group[0].frame.timestamp
    # Each frame samples a 1/fps window — extend end to cover the last frame
    end = group[-1].frame.timestamp + 1.0 / fps
    return Segment(
        start=round(start, 2),
        end=round(end, 2),
        start_fmt=_fmt_time(start),
        end_fmt=_fmt_time(end),
        frame_count=len(group),
        avg_crowd_score=round(sum(scores) / len(scores), 2),
        peak_crowd_score=max(scores),
        person_count_avg=round(sum(counts) / len(counts), 1),
        person_count_peak=max(counts),
        yt_link=_yt_link(url, start),
        reasoning_samples=[g.reasoning for g in group[:MAX_REASONING_SAMPLES]],
    )


def _save(video_dir: Path, video_id: str, segments: list[Segment]) -> None:
    (video_dir / "segments.json").write_text(json.dumps({
        "video_id": video_id,
        "segment_count": len(segments),
        "segments": [asdict(s) for s in segments],
    }, indent=2))


# ── Public API ───────────────────────────────────────────────────────────────
def build_segments(
    scored: ScoredVideo,
    min_segment_secs: float = MIN_SEGMENT_SECS,
    crowd_threshold: int = CROWD_THRESHOLD,
) -> VideoSegments:
    """Merge Stage 4's high-density frames into time segments.

    crowd_threshold overrides CROWD_THRESHOLD so tune.py can re-segment
    from cache without changing any module constant.
    """
    video = scored.prefiltered.video
    # Filter inline so callers can pass a custom threshold
    hits = [s for s in scored.gemini_scores if s.crowd_score >= crowd_threshold]
    fps = float(video.metadata.get("fps_extracted", 1.0))

    groups = _group_by_gap(hits, min_segment_secs)
    segments = [_aggregate(g, video.url, fps) for g in groups]

    _save(video.frames_dir.parent, video.video_id, segments)

    if segments:
        peak = max(s.peak_crowd_score for s in segments)
        total = sum(s.end - s.start for s in segments)
        print(f"[stage6] {video.video_id}: {len(segments)} segments, "
              f"peak score {peak}, {total:.0f}s of crowd "
              f"(threshold={crowd_threshold})")
    else:
        print(f"[stage6] {video.video_id}: no segments at threshold={crowd_threshold}")

    return VideoSegments(scored=scored, segments=segments)


def run_stage6(
    scored_videos: list[ScoredVideo],
    min_segment_secs: float = MIN_SEGMENT_SECS,
    crowd_threshold: int = CROWD_THRESHOLD,
) -> list[VideoSegments]:
    """Build segments for a batch of scored videos."""
    results = [build_segments(sv, min_segment_secs, crowd_threshold) for sv in scored_videos]
    total = sum(len(vs.segments) for vs in results)
    print(f"[stage6] batch done — {total} segments across {len(results)} videos")
    return results


# ── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    from stage2_download_prepare import run_stage2
    from stage3_prefilter import run_stage3
    from stage4_gemini_score import run_stage4

    parser = argparse.ArgumentParser(description="Stage 6: Build time segments")
    parser.add_argument("urls", nargs="+", help="YouTube video URLs")
    parser.add_argument("--out", default="./crowd_output")
    parser.add_argument("--min-persons", type=int, default=5)
    parser.add_argument("--rpm", type=int, default=15)
    parser.add_argument("--gap-secs", type=float, default=MIN_SEGMENT_SECS,
                        help="Max seconds between hits within a segment")
    args = parser.parse_args()

    videos = run_stage2(args.urls, output_dir=args.out)
    prefiltered = run_stage3(videos, person_count_min=args.min_persons)
    scored = run_stage4(prefiltered, rpm=args.rpm)
    segmented = run_stage6(scored, min_segment_secs=args.gap_secs)

    for vs in segmented:
        print(f"\n{vs.video.title}")
        for s in vs.segments:
            print(f"  [{s.start_fmt}–{s.end_fmt}]  "
                  f"avg={s.avg_crowd_score}  peak={s.peak_crowd_score}  "
                  f"persons≈{s.person_count_avg}  → {s.yt_link}")

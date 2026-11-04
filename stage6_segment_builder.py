"""
Stage 6 — Segment Builder
Merges Stage 4's high-density frames into contiguous time segments.
Two consecutive hits stay in the same segment while their gap is
<= MIN_SEGMENT_SECS and no scene cut falls between them. Segments are
trimmed to the shot they sit in, and ones shorter than MIN_CLIP_SECS are
dropped. Aggregates score, person-count, and reasoning samples per
segment; emits YouTube deep-links.
"""

from __future__ import annotations

import bisect
import json
import math
import re
import statistics
import subprocess
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

from stage2_download_prepare import VideoData
from stage4_gemini_score import CROWD_THRESHOLD, GeminiScore, ScoredVideo

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# ── Constants ────────────────────────────────────────────────────────────────
MIN_SEGMENT_SECS = 3.0          # gap > this starts a new segment
MIN_CLIP_SECS = 1.5             # shorter segments are dropped: one sampled frame is
                                # not enough evidence (title cards, archival stills)
MAX_REASONING_SAMPLES = 3       # per-segment reasoning strings retained

# Shot boundaries. Edited uploads cut every 1-3 s, and a 1 fps sample says
# nothing about the shot next to it, so segments must not cross a cut.
# A cut is a one-frame spike in FFmpeg's scene score; a fast pan scores high
# for many frames in a row, so the spike must also dwarf its neighbours'
# median (the idea behind PySceneDetect's AdaptiveDetector). Fades to black
# are boundaries too. Dissolves are not detected.
SCENE_MIN_SCORE = 0.2           # post-processing: re-derived from shots.json
# 2.5 over 3.0: a false cut merely splits a good clip in two, a missed one
# leaves another shot inside it. Measured on 17 local videos plus the Puri clips.
SCENE_SPIKE_RATIO = 2.5         # post-processing: re-derived from shots.json
# Detection parameters — changing any of these re-runs FFmpeg.
SHOT_DETECT_PARAMS = {
    "analysis_width": 320,      # scene score is computed on a downscaled copy
    "window": 2,                # neighbours each side a spike is compared with
    "candidate_floor": 0.1,     # frames scoring below this are never cuts
    "dup_floor": 0.01,          # below this a frame is a repeat, not a neighbour
    "black_pix_th": 0.10,
    "black_min_secs": 0.1,
}


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
    vantage: str = "unknown"    # dominant camera height across the group
    reasoning_samples: list[str] = field(default_factory=list)
    peak_timestamp: float | None = None  # best-scoring hit frame; Stage 8's poster


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


def _scene_candidates(series: list[tuple[float, float]], window: int, floor: float,
                      dup_floor: float) -> list[tuple[float, float, float]]:
    """(time, score, neighbour median) for every frame scoring >= floor.

    The neighbour level is what tells a cut from motion, and it needs the full
    series, so it is computed here once and cached; the thresholds applied to
    it later stay free to change.

    Repeated frames (60 fps uploads of 24/25 fps footage) score ~0 and would
    drag a pan's neighbour level down until every moving frame looked like a
    spike, so neighbours are taken from frames that actually changed.
    """
    moving = [(t, s) for t, s in series if s >= dup_floor]
    out = []
    for i, (t, s) in enumerate(moving):
        if s < floor:
            continue
        nb = [moving[j][1] for j in range(max(0, i - window), min(len(moving), i + window + 1))
              if j != i]
        # Median, not mean: a neighbouring cut (either side of a still title
        # card, say) must not hide this one.
        out.append((round(t, 3), round(s, 4), round(statistics.median(nb), 4) if nb else 0.0))
    return out


def _pick_cuts(candidates: list, min_score: float = SCENE_MIN_SCORE,
               ratio: float = SCENE_SPIKE_RATIO) -> list[float]:
    """Cut times: frames that score high AND stand well clear of their neighbours."""
    return [t for t, s, nb in candidates if s >= min_score and s >= ratio * nb]


def _shot_boundaries(shots: dict, min_score: float = SCENE_MIN_SCORE,
                     ratio: float = SCENE_SPIKE_RATIO) -> list[float]:
    """Sorted cut times plus both edges of every black interval."""
    cuts = _pick_cuts(shots.get("candidates", []), min_score, ratio)
    black = [edge for interval in shots.get("black", []) for edge in interval]
    return sorted(set(cuts + black))


_PTS_RE = re.compile(r"pts_time:\s*(-?[\d.]+)")
_SCENE_RE = re.compile(r"lavfi\.scene_score=([\d.]+)")
_BLACK_RE = re.compile(r"black_start:\s*(-?[\d.]+)\s+black_end:\s*(-?[\d.]+)")


def _detect_shots(video_path: Path) -> dict:
    """One FFmpeg pass: per-frame scene score and black intervals. ~2 s per minute of 720p."""
    p = SHOT_DETECT_PARAMS
    vf = (f"scale={p['analysis_width']}:-2,"
          f"blackdetect=d={p['black_min_secs']}:pix_th={p['black_pix_th']},"
          "select='gte(scene,0)',metadata=print:key=lavfi.scene_score")
    cmd = ["ffmpeg", "-hide_banner", "-nostats", "-i", str(video_path),
           "-an", "-sn", "-dn", "-vf", vf, "-f", "null", "-"]
    proc = subprocess.Popen(cmd, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL,
                            text=True, encoding="utf-8", errors="replace")
    series: list[tuple[float, float]] = []
    black: list[list[float]] = []
    t = None
    for line in proc.stderr:
        if m := _PTS_RE.search(line):
            t = float(m.group(1))
        elif (m := _SCENE_RE.search(line)) and t is not None:
            series.append((t, float(m.group(1))))
        elif m := _BLACK_RE.search(line):
            black.append([float(m.group(1)), float(m.group(2))])
    if proc.wait() != 0 or not series:
        raise RuntimeError(f"ffmpeg shot detection failed for {video_path.name}")
    # Frame timestamps elsewhere count from the first frame, not the container's start_time
    t0 = series[0][0]
    series = [(t - t0, s) for t, s in series]
    black = [[round(s - t0, 3), round(e - t0, 3)] for s, e in black]
    return {
        "params": p,
        "candidates": _scene_candidates(series, p["window"], p["candidate_floor"],
                                        p["dup_floor"]),
        "black": black,
    }


def _load_or_detect_shots(video: VideoData) -> dict | None:
    """shots.json beside metadata.json; re-detects only when detection params change."""
    cache = video.frames_dir.parent / "shots.json"
    if cache.exists():
        try:
            data = json.loads(cache.read_text())
            if data.get("params") == SHOT_DETECT_PARAMS:
                return data
        except json.JSONDecodeError:
            pass
    if not video.video_path.exists():
        print(f"[stage6] {video.video_id}: no video file, segments may cross scene cuts")
        return None
    try:
        data = _detect_shots(video.video_path)
    except (RuntimeError, OSError) as exc:
        print(f"[stage6] {video.video_id}: {exc}; segments may cross scene cuts")
        return None
    cache.write_text(json.dumps(data))
    return data


def _group_by_gap(hits: list[GeminiScore], max_gap: float,
                  boundaries: list[float] | None = None,
                  fps: float = 1.0) -> list[list[GeminiScore]]:
    """Split time-ordered hits into runs where each gap <= max_gap and no
    shot boundary falls between neighbours.

    A frame stamped t is FFmpeg's sample from the middle of [t, t + 1/fps),
    so its shot is the one containing t + 0.5/fps.
    """
    boundaries = boundaries or []
    groups: list[list[GeminiScore]] = []
    prev_shot = None
    for h in hits:
        shot = bisect.bisect_right(boundaries, h.frame.timestamp + 0.5 / fps)
        if (groups and shot == prev_shot
                and h.frame.timestamp - groups[-1][-1].frame.timestamp <= max_gap):
            groups[-1].append(h)
        else:
            groups.append([h])
        prev_shot = shot
    return groups


def _shot_span(t: float, boundaries: list[float], fps: float) -> tuple[float, float]:
    """(start, end) of the shot holding the frame stamped t."""
    i = bisect.bisect_right(boundaries, t + 0.5 / fps)
    return (boundaries[i - 1] if i > 0 else 0.0,
            boundaries[i] if i < len(boundaries) else math.inf)


def _aggregate(group: list[GeminiScore], url: str, fps: float,
               boundaries: list[float] | None = None) -> Segment:
    scores = [g.crowd_score for g in group]
    counts = [g.frame.person_count for g in group]
    shot_start, shot_end = _shot_span(group[0].frame.timestamp, boundaries or [], fps)
    # Each frame samples a 1/fps window — extend end to cover the last frame,
    # but never past the cut that ends this shot
    start = max(group[0].frame.timestamp, shot_start)
    end = min(group[-1].frame.timestamp + 1.0 / fps, shot_end)
    peak = max(group, key=lambda g: g.crowd_score)
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
        vantage=Counter(g.vantage for g in group).most_common(1)[0][0],
        reasoning_samples=[g.reasoning for g in group[:MAX_REASONING_SAMPLES]],
        peak_timestamp=peak.frame.timestamp,
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
    require_elevated: bool = True,
    min_clip_secs: float = MIN_CLIP_SECS,
    shot_aware: bool = True,
) -> VideoSegments:
    """Merge Stage 4's high-density frames into time segments.

    crowd_threshold overrides CROWD_THRESHOLD so tune.py can re-segment
    from cache without changing any module constant.

    require_elevated also demands the frame was shot from above the crowd.
    Applied here rather than in Stage 4 so the vantage stays cached and a
    rerun can widen or narrow the gate for free, same as the threshold.

    shot_aware splits segments at scene cuts and fades to black and trims
    them to their shot, so a clip never runs into the next shot of an
    edited video. Cuts are detected once per video and cached in shots.json.
    """
    video = scored.prefiltered.video
    # Filter inline so callers can pass a custom threshold
    hits = [s for s in scored.gemini_scores if s.crowd_score >= crowd_threshold]
    if require_elevated:
        before = len(hits)
        hits = [s for s in hits if s.is_elevated]
        if before != len(hits):
            print(f"[stage6] {video.video_id}: vantage gate kept "
                  f"{len(hits)}/{before} high-density frames (elevated only)")
    fps = float(video.metadata.get("fps_extracted", 1.0))

    boundaries: list[float] = []
    if shot_aware and hits:
        shots = _load_or_detect_shots(video)
        if shots is not None:
            boundaries = _shot_boundaries(shots)

    groups = _group_by_gap(hits, min_segment_secs, boundaries, fps)
    segments = [_aggregate(g, video.url, fps, boundaries) for g in groups]
    short = [s for s in segments if s.end - s.start < min_clip_secs - 1e-6]
    segments = [s for s in segments if s.end - s.start >= min_clip_secs - 1e-6]
    if short:
        print(f"[stage6] {video.video_id}: dropped {len(short)} segment(s) "
              f"shorter than {min_clip_secs:g}s")

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
    require_elevated: bool = True,
    min_clip_secs: float = MIN_CLIP_SECS,
    shot_aware: bool = True,
) -> list[VideoSegments]:
    """Build segments for a batch of scored videos."""
    results = [build_segments(sv, min_segment_secs, crowd_threshold, require_elevated,
                              min_clip_secs=min_clip_secs, shot_aware=shot_aware)
               for sv in scored_videos]
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
    parser.add_argument("--workers", type=int, default=1,
                        help="Concurrent Gemini calls (paid tier); --rpm still caps the total")
    parser.add_argument("--any-vantage", action="store_true",
                        help="Keep ground-level segments too (default: elevated only)")
    parser.add_argument("--gap-secs", type=float, default=MIN_SEGMENT_SECS,
                        help="Max seconds between hits within a segment")
    parser.add_argument("--min-clip-secs", type=float, default=MIN_CLIP_SECS,
                        help="Drop segments shorter than this (0 keeps all)")
    parser.add_argument("--ignore-cuts", action="store_true",
                        help="Let segments run across scene cuts (old behaviour)")
    args = parser.parse_args()

    videos = run_stage2(args.urls, output_dir=args.out)
    prefiltered = run_stage3(videos, person_count_min=args.min_persons)
    scored = run_stage4(prefiltered, rpm=args.rpm, workers=args.workers)
    segmented = run_stage6(scored, min_segment_secs=args.gap_secs,
                           require_elevated=not args.any_vantage,
                           min_clip_secs=args.min_clip_secs,
                           shot_aware=not args.ignore_cuts)

    for vs in segmented:
        print(f"\n{vs.video.title}")
        for s in vs.segments:
            print(f"  [{s.start_fmt}–{s.end_fmt}]  "
                  f"avg={s.avg_crowd_score}  peak={s.peak_crowd_score}  "
                  f"persons≈{s.person_count_avg}  → {s.yt_link}")

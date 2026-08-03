"""
tune.py — Re-segment and re-export using cached Stage 4 scores.
No Gemini API calls. Reads every crowd_output/*/gemini_scores.json,
rebuilds the object graph, then runs Stage 6 + Stage 8 with new params.

Usage:
    python tune.py [--out DIR] [--threshold T] [--gap-secs G]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage2_download_prepare import VideoData
from stage3_prefilter import FrameScore, PrefilteredVideo
from stage4_gemini_score import CROWD_THRESHOLD, GeminiScore, ScoredVideo
from stage6_segment_builder import MIN_SEGMENT_SECS, run_stage6
from stage8_output import run_stage8


def _load_scored_video(video_dir: Path) -> ScoredVideo | None:
    score_path = video_dir / "gemini_scores.json"
    meta_path = video_dir / "metadata.json"
    if not score_path.exists() or not meta_path.exists():
        return None

    try:
        metadata = json.loads(meta_path.read_text())
        scores_data = json.loads(score_path.read_text())
    except (json.JSONDecodeError, KeyError) as exc:
        print(f"[tune] skipping {video_dir.name} — corrupted cache: {exc}")
        return None

    frames_dir = video_dir / "frames"

    video = VideoData(
        video_id=video_dir.name,
        url=metadata.get("url", ""),
        title=metadata.get("title", ""),
        duration=float(metadata.get("duration") or 0),
        video_path=next(video_dir.glob("video.*"), video_dir / "video.mp4"),
        frames_dir=frames_dir,
        frame_paths=sorted(frames_dir.glob("*.jpg")),
        metadata=metadata,
    )

    gemini_scores: list[GeminiScore] = []
    for s in scores_data.get("scores", []):
        frame = FrameScore(
            path=Path(s["frame_path"]),
            timestamp=s["timestamp"],
            person_count=s.get("person_count", 0),
            passes_filter=True,
        )
        gemini_scores.append(GeminiScore(
            frame=frame,
            crowd_score=s["crowd_score"],
            reasoning=s["reasoning"],
        ))

    gemini_scores.sort(key=lambda g: g.frame.timestamp)
    prefiltered = PrefilteredVideo(video=video)
    return ScoredVideo(prefiltered=prefiltered, gemini_scores=gemini_scores)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-segment from cached Gemini scores with new parameters. Free — no API calls."
    )
    parser.add_argument("--out", default="./crowd_output", help="Output directory to scan")
    parser.add_argument("--threshold", type=int, default=CROWD_THRESHOLD,
                        help=f"Minimum Gemini score for high-density (default: {CROWD_THRESHOLD})")
    parser.add_argument("--gap-secs", type=float, default=MIN_SEGMENT_SECS,
                        help=f"Max gap within a segment in seconds (default: {MIN_SEGMENT_SECS})")
    args = parser.parse_args()

    out = Path(args.out)
    if not out.is_dir():
        print(f"[tune] Directory not found: {out}")
        return

    video_dirs = sorted(
        d for d in out.iterdir()
        if d.is_dir() and (d / "gemini_scores.json").exists()
    )

    if not video_dirs:
        print(f"[tune] No cached gemini_scores.json found under {out}/")
        return

    print(f"[tune] {len(video_dirs)} cached video(s)  "
          f"threshold={args.threshold}  gap={args.gap_secs}s")

    scored = [sv for d in video_dirs if (sv := _load_scored_video(d))]
    segmented = run_stage6(scored, min_segment_secs=args.gap_secs, crowd_threshold=args.threshold)
    run_stage8(segmented, output_dir=args.out)


if __name__ == "__main__":
    main()

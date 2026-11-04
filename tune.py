"""
tune.py — Re-segment and re-export using cached Stage 4 scores.
No Gemini API calls. Reads every crowd_output/*/gemini_scores.json,
rebuilds the object graph, then runs Stage 6 + Stage 8 with new params.

Usage:
    python tune.py [--out DIR] [--threshold T] [--gap-secs G] [--min-clip-secs S]
    python tune.py --out crowd_output_india/<slug> --dataset dataset --location <slug>

The second form re-exports a location's dataset clips with the current Stage 6
rules (scene-cut splitting etc.), replacing that location's earlier assets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage2_download_prepare import VideoData, find_video_file
from stage3_prefilter import FrameScore, PrefilteredVideo
from stage4_gemini_score import CROWD_THRESHOLD, GeminiScore, ScoredVideo
from stage6_segment_builder import MIN_CLIP_SECS, MIN_SEGMENT_SECS, run_stage6
from stage7_crowd_count import run_stage7
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
        video_path=find_video_file(video_dir),
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
            # Without this every frame reloads as "unknown" and the elevated-only
            # gate in Stage 6 rejects all of them.
            vantage=s.get("vantage", "unknown"),
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
    parser.add_argument("--min-clip-secs", type=float, default=MIN_CLIP_SECS,
                        help=f"Drop segments shorter than this (default: {MIN_CLIP_SECS})")
    parser.add_argument("--ignore-cuts", action="store_true",
                        help="Let segments run across scene cuts (old behaviour)")
    parser.add_argument("--any-vantage", action="store_true",
                        help="Keep ground-level segments too (default: elevated only)")
    parser.add_argument("--dataset", default="",
                        help="Also re-export the dataset tree into this directory")
    parser.add_argument("--location", default="",
                        help="Location slug for --dataset, e.g. puri-jagannath-rath-yatra")
    parser.add_argument("--export-clips", action="store_true",
                        help="Re-extract per-segment MP4s under OUT/clips/")
    parser.add_argument("--no-count", action="store_true",
                        help="Skip Stage 7 crowd counting (local model, cached per frame)")
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
    segmented = run_stage6(scored, min_segment_secs=args.gap_secs, crowd_threshold=args.threshold,
                           require_elevated=not args.any_vantage,
                           min_clip_secs=args.min_clip_secs, shot_aware=not args.ignore_cuts)
    counted = run_stage7(segmented, enabled=not args.no_count)
    run_stage8(counted, output_dir=args.out, export_clips=args.export_clips,
               dataset_dir=args.dataset or None, location=args.location or None)


if __name__ == "__main__":
    main()

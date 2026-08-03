"""
Stage 8 — Output & Analytics
Aggregates VideoSegments across a batch into results.json + segments.csv.
Optionally extracts a per-segment MP4 clip for each hit via FFmpeg.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from stage6_segment_builder import VideoSegments

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# Signal sources present in our lean stack. Grows if Stage 5 (audio) or a
# custom fusion model is added.
SIGNAL_SOURCES = ["gemini", "yolo"]


# ── JSON writer ──────────────────────────────────────────────────────────────
def _video_dict(vs: VideoSegments) -> dict:
    video = vs.video
    return {
        "video_id": video.video_id,
        "title": video.title,
        "url": video.url,
        "duration": video.duration,
        # populated by Stage 7 (enrichment) if it runs first; else placeholders
        "location": video.metadata.get("location"),
        "entities": video.metadata.get("entities", []),
        "high_density_segments": [
            {**asdict(s), "signal_sources": SIGNAL_SOURCES}
            for s in vs.segments
        ],
    }


def _write_json(out_path: Path, videos: list[VideoSegments]) -> None:
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "video_count": len(videos),
        "total_segments": sum(len(v.segments) for v in videos),
        "videos": [_video_dict(v) for v in videos],
    }
    out_path.write_text(json.dumps(payload, indent=2))


# ── CSV writer ───────────────────────────────────────────────────────────────
CSV_FIELDS = [
    "video_id", "title", "url",
    "segment_index",
    "start", "end", "duration_secs",
    "start_fmt", "end_fmt",
    "frame_count",
    "avg_crowd_score", "peak_crowd_score",
    "person_count_avg", "person_count_peak",
    "yt_link",
]


def _write_csv(out_path: Path, videos: list[VideoSegments]) -> None:
    # newline="" avoids blank rows on Windows; utf-8 for non-ASCII titles
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for vs in videos:
            video = vs.video
            for i, s in enumerate(vs.segments):
                writer.writerow({
                    "video_id": video.video_id,
                    "title": video.title,
                    "url": video.url,
                    "segment_index": i,
                    "start": s.start,
                    "end": s.end,
                    "duration_secs": round(s.end - s.start, 2),
                    "start_fmt": s.start_fmt,
                    "end_fmt": s.end_fmt,
                    "frame_count": s.frame_count,
                    "avg_crowd_score": s.avg_crowd_score,
                    "peak_crowd_score": s.peak_crowd_score,
                    "person_count_avg": s.person_count_avg,
                    "person_count_peak": s.person_count_peak,
                    "yt_link": s.yt_link,
                })


# ── Clip extraction ──────────────────────────────────────────────────────────
def _extract_clip(video_path: Path, start: float, end: float, out_path: Path) -> bool:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    duration = end - start
    # Input-seek + stream copy: fast; may snap to nearest keyframe (~few tenths early).
    # Drop "-c copy" if you need frame-accurate cuts and can afford re-encoding.
    cmd = [
        "ffmpeg",
        "-ss", f"{start:.2f}",
        "-i", str(video_path),
        "-t", f"{duration:.2f}",
        "-c", "copy",
        "-y", "-loglevel", "error",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[stage8] clip failed for {out_path.name}: {result.stderr.strip()}")
        return False
    return True


def _extract_clips_for_video(vs: VideoSegments, clips_dir: Path) -> int:
    video = vs.video
    if not video.video_path.exists():
        print(f"[stage8] {video.video_id}: source mp4 missing, skipping clips")
        return 0
    count = 0
    for s in vs.segments:
        # Zero-padded times keep the filename lexicographically sortable
        name = f"{video.video_id}_{int(s.start):04d}-{int(s.end):04d}.mp4"
        if _extract_clip(video.video_path, s.start, s.end, clips_dir / name):
            count += 1
    return count


# ── Public API ───────────────────────────────────────────────────────────────
def run_stage8(
    videos: list[VideoSegments],
    output_dir: str = "./crowd_output",
    export_clips: bool = False,
) -> dict[str, Path]:
    """
    Write results.json + segments.csv at the top level of output_dir.
    If export_clips is set, also emit per-segment MP4s under output_dir/clips/.
    Returns a mapping of artifact-name → path for the CLI to report.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    json_path = out / "results.json"
    csv_path = out / "segments.csv"
    _write_json(json_path, videos)
    _write_csv(csv_path, videos)

    total_segs = sum(len(v.segments) for v in videos)
    print(f"[stage8] wrote {json_path.name} + {csv_path.name} — "
          f"{total_segs} segments across {len(videos)} videos")

    outputs: dict[str, Path] = {"json": json_path, "csv": csv_path}

    if export_clips:
        clips_dir = out / "clips"
        total_clips = sum(_extract_clips_for_video(vs, clips_dir) for vs in videos)
        print(f"[stage8] extracted {total_clips} clips → {clips_dir}")
        outputs["clips_dir"] = clips_dir

    return outputs


# ── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    from stage2_download_prepare import run_stage2
    from stage3_prefilter import run_stage3
    from stage4_gemini_score import run_stage4
    from stage6_segment_builder import run_stage6

    parser = argparse.ArgumentParser(description="Stage 8: Output & analytics")
    parser.add_argument("urls", nargs="+", help="YouTube video URLs")
    parser.add_argument("--out", default="./crowd_output")
    parser.add_argument("--min-persons", type=int, default=5)
    parser.add_argument("--rpm", type=int, default=15)
    parser.add_argument("--gap-secs", type=float, default=3.0)
    parser.add_argument("--export-clips", action="store_true",
                        help="Extract per-segment MP4 clips via FFmpeg")
    args = parser.parse_args()

    videos = run_stage2(args.urls, output_dir=args.out)
    prefiltered = run_stage3(videos, person_count_min=args.min_persons)
    scored = run_stage4(prefiltered, rpm=args.rpm)
    segmented = run_stage6(scored, min_segment_secs=args.gap_secs)
    outputs = run_stage8(segmented, output_dir=args.out, export_clips=args.export_clips)

    print("\n[stage8] artifacts:")
    for k, v in outputs.items():
        print(f"  {k}: {v}")

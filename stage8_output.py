"""
Stage 8 — Output & Analytics
Aggregates VideoSegments across a batch into results.json + segments.csv.
Optionally extracts a per-segment MP4 clip for each hit via FFmpeg.
"""

from __future__ import annotations

import csv
import functools
import json
import re
import shutil
import subprocess
import sys
import unicodedata
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from stage4_gemini_score import CROWD_THRESHOLD, GEMINI_MODEL
from stage6_segment_builder import VideoSegments
from stage7_crowd_count import COUNT_METHOD, SegmentCount

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# Signal sources present in our lean stack. Grows if Stage 5 (audio) or a
# custom fusion model is added.
SIGNAL_SOURCES = ["gemini", "yolo"]
YOLO_COUNT_NOTE = "YOLOv8n detections; misses most people seen from height"


def _segment_counts(vs) -> list[SegmentCount | None]:
    """Stage 7 counts aligned with vs.segments; all None for a plain VideoSegments."""
    counts = getattr(vs, "segment_counts", None) or []
    return counts if len(counts) == len(vs.segments) else [None] * len(vs.segments)


def _count_fields(c: SegmentCount | None) -> dict:
    return {"crowd_count_avg": c.avg if c else "", "crowd_count_peak": c.peak if c else ""}


# ── JSON writer ──────────────────────────────────────────────────────────────
def _video_dict(vs: VideoSegments) -> dict:
    video = vs.video
    return {
        "video_id": video.video_id,
        "title": video.title,
        "url": video.url,
        "duration": video.duration,
        # placeholders for location/entity enrichment, which is not built yet
        "location": video.metadata.get("location"),
        "entities": video.metadata.get("entities", []),
        "high_density_segments": [
            {**asdict(s), **_count_fields(c), "signal_sources": SIGNAL_SOURCES}
            for s, c in zip(vs.segments, _segment_counts(vs))
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
    "crowd_count_avg", "crowd_count_peak",
    "yt_link",
]


def _write_csv(out_path: Path, videos: list[VideoSegments]) -> None:
    # newline="" avoids blank rows on Windows; utf-8 for non-ASCII titles
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for vs in videos:
            video = vs.video
            for i, (s, c) in enumerate(zip(vs.segments, _segment_counts(vs))):
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
                    **_count_fields(c),
                    "yt_link": s.yt_link,
                })


# ── Clip extraction ──────────────────────────────────────────────────────────
@functools.lru_cache(maxsize=64)
def _source_video_kbps(video_path: str) -> int | None:
    """Video-stream bitrate of the download, falling back to the container's."""
    for entries in (["-select_streams", "v:0", "-show_entries", "stream=bit_rate"],
                    ["-show_entries", "format=bit_rate"]):
        r = subprocess.run(["ffprobe", "-v", "error", *entries, "-of", "csv=p=0", video_path],
                           capture_output=True, text=True)
        value = r.stdout.strip().splitlines()[0] if r.stdout.strip() else ""
        if value.isdigit() and int(value) > 0:
            return int(value) // 1000
    return None


def _extract_clip(video_path: Path, start: float, end: float, out_path: Path) -> bool:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    duration = end - start
    # Re-encode rather than stream-copy. "-c copy" has to start at the previous
    # keyframe, and in an edited upload that is often the previous shot: the MP4
    # edit list hides it in most players, but the frames are still in the file
    # and loaders that ignore edit lists show them. Segments now end exactly on
    # scene cuts, so the cut must be frame-accurate. Costs ~1 s per clip.
    #
    # Bitrate capped at 2x the source: plain CRF spends 3-6x YouTube's bitrate
    # preserving its compression noise, with no visible gain. Downloads are
    # mostly AV1/VP9, which H.264 needs ~2x the bits to match; in exchange the
    # clips play everywhere (AV1-in-MP4 does not on stock Windows).
    rate = []
    if kbps := _source_video_kbps(str(video_path)):
        cap = max(2 * kbps, 1000)
        rate = ["-maxrate", f"{cap}k", "-bufsize", f"{2 * cap}k"]
    cmd = [
        "ffmpeg",
        "-ss", f"{start:.3f}",
        "-i", str(video_path),
        "-t", f"{duration:.3f}",
        "-map", "0:v:0", "-map", "0:a:0?",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20", *rate, "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
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


# ── Getty-style dataset export ───────────────────────────────────────────────
# Layout: dataset/<location>/<vantage>/<ASSET_ID>/{clip.mp4,poster.jpg,metadata.json}
# The pipeline's working directory stays separate — this tree is the deliverable,
# rebuildable from cache and safe to delete.
MANIFEST_FIELDS = [
    "asset_id", "location", "city", "vantage",
    "duration_secs", "start_fmt", "end_fmt",
    "avg_crowd_score", "peak_crowd_score",
    "person_count_avg", "person_count_peak",
    "crowd_count_avg", "crowd_count_peak",
    "frame_count", "keywords",
    "source_video_id", "source_title", "source_channel",
    "source_url", "yt_link", "clip_path",
]


def _slugify(text: str) -> str:
    """Lowercase, hyphen-separated, ASCII-only — safe as a directory name."""
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"-{2,}", "-", re.sub(r"[^a-z0-9]+", "-", ascii_text.lower())).strip("-")


# Place-type words carry no identity: "market" would code half the dataset MAR.
_GENERIC_PLACE_WORDS = frozenset({
    "market", "bazaar", "bazar", "road", "street", "bridge", "chowk",
    "nagar", "square", "station", "temple", "fort", "gate", "circle",
    "junction", "mall", "complex", "area", "lane", "ghat", "crossing",
})


def _location_code(location_slug: str) -> str:
    """Three-letter asset-ID code: mumbai-crawford-market -> CRA, delhi-chandni-chowk -> CHA.

    Skips the city (first word) and generic place types, so the code names the
    thing that actually distinguishes one location from another.
    """
    words = [w for w in location_slug.split("-") if w]
    if not words:
        return "GEN"
    candidates = [w for w in words[1:] if w not in _GENERIC_PLACE_WORDS and len(w) >= 3]
    if not candidates:
        candidates = [w for w in words[1:] if len(w) >= 3] or words
    return candidates[0][:3].upper()


def _city_of(location_slug: str) -> str:
    return location_slug.split("-")[0] if location_slug else "unknown"


def _peak_frame_path(vs: VideoSegments, seg):
    """The segment's best-scoring hit frame — becomes the poster image.

    Uses the hit Stage 6 recorded rather than the best frame in the time range,
    which could be a ground-level frame the vantage gate rejected.
    """
    if seg.peak_timestamp is not None:
        in_range = [g for g in vs.scored.gemini_scores
                    if abs(g.frame.timestamp - seg.peak_timestamp) < 1e-6]
    else:  # segments.json written before peak_timestamp existed
        in_range = [g for g in vs.scored.gemini_scores
                    if seg.start <= g.frame.timestamp <= seg.end]
    if not in_range:
        return None
    best = max(in_range, key=lambda g: g.crowd_score)
    path = Path(best.frame.path)
    return path if path.exists() else None


def _asset_metadata(asset_id, location, vantage, vs, seg, keywords, count=None) -> dict:
    video = vs.video
    return {
        "asset_id": asset_id,
        "caption": (f"{vantage.capitalize()} view of a crowd at "
                    f"{location.replace('-', ' ')}, India."),
        "location": location,
        "city": _city_of(location),
        "country": "India",
        "vantage": vantage,
        "keywords": keywords,
        "duration_secs": round(seg.end - seg.start, 2),
        "source": {
            "video_id": video.video_id,
            "title": video.title,
            "channel": video.metadata.get("channel") or video.metadata.get("uploader", ""),
            "url": video.url,
            "yt_link": seg.yt_link,
            "segment_start": seg.start,
            "segment_end": seg.end,
        },
        "measurements": {
            "avg_crowd_score": seg.avg_crowd_score,
            "peak_crowd_score": seg.peak_crowd_score,
            # Estimated people (Stage 7). Blank if counting was skipped.
            **_count_fields(count),
            "crowd_count_method": COUNT_METHOD if count else "",
            # Raw YOLO numbers, kept for provenance: near 0 for most aerial frames.
            "person_count_avg": seg.person_count_avg,
            "person_count_peak": seg.person_count_peak,
            "person_count_method": YOLO_COUNT_NOTE,
            "frame_count": seg.frame_count,
        },
        "reasoning_samples": seg.reasoning_samples,
        # Provenance, not a licence. These are third-party YouTube uploads;
        # clearing rights for any downstream use is the dataset owner's job.
        "rights": {
            "source_platform": "YouTube",
            "rights_cleared": False,
            "note": "Derived from a third-party upload. Not licensed for redistribution.",
        },
        "pipeline": {
            "scored_by": GEMINI_MODEL,
            "crowd_threshold": CROWD_THRESHOLD,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
    }


def export_dataset(
    videos: list[VideoSegments],
    dataset_dir: str,
    location: str | None = None,
) -> Path:
    """
    Write the curated tree plus manifest.csv. Returns the dataset root.

    `location` labels every asset in this batch; without it each video falls
    back to a slug of its own title, which is noisier but keeps the tree valid.

    Re-exporting a video replaces that video's earlier assets at the same
    location (after re-segmenting, say), and new IDs number on from the highest
    ID already in the manifest. So rerunning a query, adding a second query for
    a location, or two locations sharing a code never reuse an ID or overwrite
    another video's folder.
    """
    root = Path(dataset_dir)
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / "manifest.csv"
    rows: list[dict] = []

    def loc_of(vs: VideoSegments) -> str:
        return _slugify(location) if location else (_slugify(vs.video.title)[:40] or "unspecified")

    replacing = {(loc_of(vs), vs.video.video_id) for vs in videos}
    kept, stale = [], []
    for r in _read_manifest(manifest):
        (stale if (r.get("location"), r.get("source_video_id")) in replacing else kept).append(r)
    for r in stale:
        _remove_asset_dir(root, r)

    counters: dict[str, int] = {}
    for r in kept:
        if m := _ASSET_ID_RE.fullmatch(r["asset_id"]):
            counters[m["code"]] = max(counters.get(m["code"], 0), int(m["n"]))

    for vs in videos:
        video = vs.video
        loc = loc_of(vs)
        code = _location_code(loc)

        for seg, count in zip(vs.segments, _segment_counts(vs)):
            vantage = seg.vantage or "unknown"
            counters[code] = counters.get(code, 0) + 1
            asset_id = f"IND_{code}_{counters[code]:04d}"
            asset_dir = root / loc / vantage / asset_id
            asset_dir.mkdir(parents=True, exist_ok=True)

            if video.video_path.exists():
                _extract_clip(video.video_path, seg.start, seg.end, asset_dir / "clip.mp4")
            else:
                print(f"[stage8] {asset_id}: source mp4 missing, metadata only")

            poster = _peak_frame_path(vs, seg)
            if poster:
                shutil.copy2(poster, asset_dir / "poster.jpg")

            keywords = sorted({
                "crowd", "india", "crowd density", vantage,
                *[w for w in loc.split("-") if len(w) > 2],
            })
            (asset_dir / "metadata.json").write_text(
                json.dumps(_asset_metadata(asset_id, loc, vantage, vs, seg, keywords, count),
                           indent=2, ensure_ascii=False),
                encoding="utf-8")

            rows.append({
                "asset_id": asset_id,
                "location": loc,
                "city": _city_of(loc),
                "vantage": vantage,
                "duration_secs": round(seg.end - seg.start, 2),
                "start_fmt": seg.start_fmt,
                "end_fmt": seg.end_fmt,
                "avg_crowd_score": seg.avg_crowd_score,
                "peak_crowd_score": seg.peak_crowd_score,
                "person_count_avg": seg.person_count_avg,
                "person_count_peak": seg.person_count_peak,
                **_count_fields(count),
                "frame_count": seg.frame_count,
                "keywords": "; ".join(keywords),
                "source_video_id": video.video_id,
                "source_title": video.title,
                "source_channel": video.metadata.get("channel")
                                  or video.metadata.get("uploader", ""),
                "source_url": video.url,
                "yt_link": seg.yt_link,
                "clip_path": (Path(loc) / vantage / asset_id / "clip.mp4").as_posix(),
            })

    _write_manifest(manifest, kept + rows)

    by_vantage: dict[str, int] = {}
    for r in rows:
        by_vantage[r["vantage"]] = by_vantage.get(r["vantage"], 0) + 1
    mix = ", ".join(f"{v}={n}" for v, n in sorted(by_vantage.items())) or "none"
    replaced = f", replaced {len(stale)} earlier" if stale else ""
    print(f"[stage8] dataset — {len(rows)} assets ({mix}){replaced} -> {root}/")
    return root


_ASSET_ID_RE = re.compile(r"IND_(?P<code>[A-Z0-9]+)_(?P<n>\d+)")


def _read_manifest(manifest: Path) -> list[dict]:
    if not manifest.exists():
        return []
    with open(manifest, newline="", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r.get("asset_id")]


def _remove_asset_dir(root: Path, row: dict) -> None:
    """Delete one superseded asset folder, and only if it sits inside the dataset."""
    rel = Path(row["clip_path"]).parent if row.get("clip_path") else \
        Path(row.get("location", "")) / row.get("vantage", "") / row["asset_id"]
    asset_dir = (root / rel).resolve()
    if asset_dir.name == row["asset_id"] and asset_dir.is_relative_to(root.resolve()) \
            and asset_dir.is_dir():
        shutil.rmtree(asset_dir)


def _write_manifest(manifest: Path, rows: list[dict]) -> None:
    with open(manifest, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in MANIFEST_FIELDS})


# ── Public API ───────────────────────────────────────────────────────────────
def run_stage8(
    videos: list[VideoSegments],  # or Stage 7 CountedVideo, which adds crowd counts
    output_dir: str = "./crowd_output",
    export_clips: bool = False,
    dataset_dir: str | None = None,
    location: str | None = None,
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

    if dataset_dir:
        outputs["dataset"] = export_dataset(videos, dataset_dir, location)

    return outputs


# ── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    from stage2_download_prepare import run_stage2
    from stage3_prefilter import run_stage3
    from stage4_gemini_score import run_stage4
    from stage6_segment_builder import run_stage6
    from stage7_crowd_count import run_stage7

    parser = argparse.ArgumentParser(description="Stage 8: Output & analytics")
    parser.add_argument("urls", nargs="+", help="YouTube video URLs")
    parser.add_argument("--out", default="./crowd_output")
    parser.add_argument("--min-persons", type=int, default=5)
    parser.add_argument("--rpm", type=int, default=15)
    parser.add_argument("--workers", type=int, default=1,
                        help="Concurrent Gemini calls (paid tier); --rpm still caps the total")
    parser.add_argument("--gap-secs", type=float, default=3.0)
    parser.add_argument("--min-clip-secs", type=float, default=1.5,
                        help="Drop segments shorter than this (0 keeps all)")
    parser.add_argument("--any-vantage", action="store_true",
                        help="Keep ground-level segments too (default: elevated only)")
    parser.add_argument("--dataset", default="",
                        help="Export a Getty-style dataset tree into this directory")
    parser.add_argument("--location", default="",
                        help="Location slug for the dataset tree, e.g. mumbai-crawford-market")
    parser.add_argument("--export-clips", action="store_true",
                        help="Extract per-segment MP4 clips via FFmpeg")
    parser.add_argument("--no-count", action="store_true",
                        help="Skip Stage 7 crowd counting (faster on CPU; counts left blank)")
    args = parser.parse_args()

    videos = run_stage2(args.urls, output_dir=args.out)
    prefiltered = run_stage3(videos, person_count_min=args.min_persons)
    scored = run_stage4(prefiltered, rpm=args.rpm, workers=args.workers)
    segmented = run_stage6(scored, min_segment_secs=args.gap_secs,
                           require_elevated=not args.any_vantage,
                           min_clip_secs=args.min_clip_secs)
    counted = run_stage7(segmented, enabled=not args.no_count)
    outputs = run_stage8(counted, output_dir=args.out, export_clips=args.export_clips,
                         dataset_dir=args.dataset or None, location=args.location or None)

    print("\n[stage8] artifacts:")
    for k, v in outputs.items():
        print(f"  {k}: {v}")

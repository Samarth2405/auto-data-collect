"""
Stage 2 — Download & Prepare
Downloads YouTube videos via yt-dlp and extracts JPEG frames via FFmpeg.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yt_dlp

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# ── Constants (override via run_stage2 kwargs) ───────────────────────────────
FRAME_RATE = 1.0       # frames per second to extract
RESIZE_WIDTH = 1280    # pixel width; height scales to maintain aspect ratio
MAX_HEIGHT = 720       # cap download resolution; frames are downsampled anyway
OUTPUT_DIR = "./crowd_output"


# ── Data contract passed to downstream stages ────────────────────────────────
@dataclass
class VideoData:
    video_id: str
    url: str
    title: str
    duration: float          # seconds
    video_path: Path
    frames_dir: Path
    frame_paths: list[Path] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


# ── Internal helpers ─────────────────────────────────────────────────────────

_YOUTUBE_ID_RE = re.compile(
    r"(?:youtu\.be/|youtube\.com/(?:watch\?v=|embed/|shorts/|v/))"
    r"(?P<id>[A-Za-z0-9_-]{11})"
)


def _extract_video_id(url: str) -> str | None:
    """Parse the 11-char video ID from a YouTube URL without hitting the network."""
    m = _YOUTUBE_ID_RE.search(url)
    return m.group("id") if m else None


def find_video_file(video_dir: Path) -> Path:
    """The downloaded video, skipping yt-dlp's partial and per-format leftovers.

    A bare glob("video.*") can pick video.f398.mp4.part from an interrupted
    download. Falls back to video.mp4 (possibly absent) so callers can test .exists().
    """
    for ext in ("mp4", "mkv", "webm"):
        if (video_dir / f"video.{ext}").exists():
            return video_dir / f"video.{ext}"
    for p in sorted(video_dir.glob("video.*")):
        if p.suffix.lower() in (".mp4", ".mkv", ".webm"):
            return p
    return video_dir / "video.mp4"


def _load_from_cache(video_dir: Path, url: str) -> VideoData | None:
    """Return VideoData if a *complete* prior extraction exists on disk."""
    done_marker = video_dir / ".stage2_done"
    frames_dir = video_dir / "frames"
    if not done_marker.exists():
        return None
    frame_paths = sorted(frames_dir.glob("*.jpg"))
    if not frame_paths:
        return None
    meta_path = video_dir / "metadata.json"
    metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    video_path = find_video_file(video_dir)
    return VideoData(
        video_id=video_dir.name,
        url=url,
        title=metadata.get("title", "unknown"),
        duration=float(metadata.get("duration") or 0),
        video_path=video_path,
        frames_dir=frames_dir,
        frame_paths=frame_paths,
        metadata=metadata,
    )


def _check_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise EnvironmentError(
            "ffmpeg not found. Install it and ensure it is on PATH.\n"
            "  Windows: winget install ffmpeg\n"
            "  macOS:   brew install ffmpeg\n"
            "  Linux:   sudo apt install ffmpeg"
        )


def _download_video(
    url: str,
    video_dir: Path,
    max_height: int = MAX_HEIGHT,
) -> tuple[Path, dict]:
    """
    Download a single video; return (mp4_path, yt-dlp info dict).

    Capped at `max_height` because frames are downsampled to RESIZE_WIDTH before
    anything looks at them — pulling 4K costs ~10x the bytes for identical scores.
    """
    video_dir.mkdir(parents=True, exist_ok=True)
    out_template = str(video_dir / "video.%(ext)s")

    # Four alternatives, each capped, because a bare uncapped tail would let any
    # upload lacking an m4a audio stream fall through to 4K. format_sort keeps the
    # last-resort "best" at or below the cap too.
    ydl_opts = {
        # prefer mp4+m4a at the cap, then any codec at the cap, then pre-muxed
        "format": (
            f"bestvideo[ext=mp4][height<={max_height}]+bestaudio[ext=m4a]/"
            f"bestvideo[height<={max_height}]+bestaudio/"
            f"best[height<={max_height}]/best"
        ),
        "format_sort": [f"res:{max_height}"],
        "outtmpl": out_template,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "merge_output_format": "mp4",
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)

    # yt-dlp may produce .mp4 or .mkv depending on merge; find what landed
    for ext in ("mp4", "mkv", "webm"):
        candidate = video_dir / f"video.{ext}"
        if candidate.exists():
            return candidate, info

    raise FileNotFoundError(f"Downloaded file not found in {video_dir}")


def _extract_frames(
    video_path: Path,
    frames_dir: Path,
    fps: float,
    resize_width: int,
) -> list[Path]:
    """Run FFmpeg to extract JPEG frames; return sorted list of frame paths."""
    frames_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-i", str(video_path),
        "-vf", f"fps={fps},scale={resize_width}:-1",
        "-q:v", "2",                          # JPEG quality 2 = high quality
        str(frames_dir / "%06d.jpg"),
        "-y",                                 # overwrite if rerunning
        "-loglevel", "error",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed:\n{result.stderr}")

    return sorted(frames_dir.glob("*.jpg"))


def _save_metadata(video_dir: Path, data: dict) -> None:
    with open(video_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


# ── Public API ───────────────────────────────────────────────────────────────

def prepare_video(
    url: str,
    output_dir: Path,
    fps: float = FRAME_RATE,
    resize_width: int = RESIZE_WIDTH,
    max_height: int = MAX_HEIGHT,
) -> VideoData | None:
    """
    Download one video and extract its frames.
    Returns VideoData on success, None if download fails (private/geo-blocked).
    Idempotent: reuses cached extractions marked by `.stage2_done`.
    """
    _check_ffmpeg()

    # Fast path: derive video_id from URL locally and reuse cache without network
    local_id = _extract_video_id(url)
    if local_id:
        cached = _load_from_cache(output_dir / local_id, url)
        if cached:
            print(f"[stage2] {local_id}: cached (offline) — {len(cached.frame_paths)} frames")
            return cached

    # Cache miss — probe yt-dlp for the canonical id and metadata
    print(f"[stage2] probing {url}")
    try:
        with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True}) as ydl:
            probe = ydl.extract_info(url, download=False)
        video_id = probe["id"]
        title = probe.get("title", "untitled")
        duration = float(probe.get("duration") or 0)
    except yt_dlp.utils.DownloadError as exc:
        print(f"[stage2] skipping {url} — {exc}")
        return None

    video_dir = output_dir / video_id

    # Second cache check — for shortened URLs where the local regex missed
    cached = _load_from_cache(video_dir, url)
    if cached:
        print(f"[stage2] {video_id}: cached — {len(cached.frame_paths)} frames")
        return cached

    # ── Download ──────────────────────────────────────────────────────────────
    print(f"[stage2] downloading {video_id}")
    try:
        video_path, info = _download_video(url, video_dir, max_height)
    except yt_dlp.utils.DownloadError as exc:
        print(f"[stage2] skipping {url} — {exc}")
        return None

    # ── Extract frames ────────────────────────────────────────────────────────
    frames_dir = video_dir / "frames"
    print(f"[stage2] {video_id}: extracting frames at {fps} fps → {frames_dir}")
    frame_paths = _extract_frames(video_path, frames_dir, fps, resize_width)
    print(f"[stage2] {video_id}: {len(frame_paths)} frames extracted")

    # Keep only the subset of yt-dlp info that is useful downstream
    metadata = {
        "video_id": video_id,
        "title": title,
        "duration": duration,
        "url": url,
        "uploader": info.get("uploader"),
        "upload_date": info.get("upload_date"),
        "view_count": info.get("view_count"),
        "like_count": info.get("like_count"),
        "description": (info.get("description") or "")[:500],
        "tags": info.get("tags", [])[:20],
        "fps_extracted": fps,
        "resize_width": resize_width,
    }
    _save_metadata(video_dir, metadata)
    # Sentinel: extraction is complete. _load_from_cache uses this to reject
    # directories left half-populated by an interrupted FFmpeg run.
    (video_dir / ".stage2_done").touch()

    return VideoData(
        video_id=video_id,
        url=url,
        title=title,
        duration=duration,
        video_path=video_path,
        frames_dir=frames_dir,
        frame_paths=frame_paths,
        metadata=metadata,
    )


def run_stage2(
    urls: list[str],
    output_dir: str = OUTPUT_DIR,
    fps: float = FRAME_RATE,
    resize_width: int = RESIZE_WIDTH,
    max_height: int = MAX_HEIGHT,
) -> list[VideoData]:
    """Process a list of video URLs. Returns only successfully prepared videos."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    results = []
    for url in urls:
        video_data = prepare_video(url, out, fps, resize_width, max_height)
        if video_data is not None:
            results.append(video_data)

    print(f"[stage2] done — {len(results)}/{len(urls)} videos prepared")
    return results


# ── CLI for standalone testing ───────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Stage 2: Download & Prepare")
    parser.add_argument("urls", nargs="+", help="YouTube video URLs")
    parser.add_argument("--out", default=OUTPUT_DIR, help="Output directory")
    parser.add_argument("--fps", type=float, default=FRAME_RATE, help="Frames per second")
    parser.add_argument("--width", type=int, default=RESIZE_WIDTH, help="Resize width in pixels")
    parser.add_argument("--max-height", type=int, default=MAX_HEIGHT,
                        help="Cap download resolution (default 720)")
    args = parser.parse_args()

    videos = run_stage2(args.urls, output_dir=args.out, fps=args.fps,
                        resize_width=args.width, max_height=args.max_height)
    for v in videos:
        print(f"  {v.video_id}: {len(v.frame_paths)} frames — {v.title}")

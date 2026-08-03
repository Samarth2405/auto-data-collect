"""
Stage 3 — Local Person Pre-Filter
Replaces Google Cloud Video Intelligence + Cloud Vision with a local
YOLOv8-nano detector. Counts persons per frame and marks frames as
candidates for Gemini scoring only if they exceed PERSON_COUNT_MIN.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from stage2_download_prepare import VideoData

# Lazy import: ultralytics pulls torch (~500 MB). Kept out of module scope
# so downstream code can import FrameScore / PrefilteredVideo without it.
if TYPE_CHECKING:
    from ultralytics import YOLO

# ── Constants ────────────────────────────────────────────────────────────────
PERSON_COUNT_MIN = 5            # threshold to survive the pre-filter
YOLO_MODEL = "yolov8n.pt"       # nano — ~6 MB, CPU-friendly
YOLO_CONFIDENCE = 0.3           # per-detection confidence threshold
PERSON_CLASS_ID = 0             # COCO class 0 = person
BATCH_SIZE = 32                 # frames per YOLO forward pass


# ── Data contract ────────────────────────────────────────────────────────────
@dataclass
class FrameScore:
    path: Path
    timestamp: float            # seconds into the video
    person_count: int
    passes_filter: bool


@dataclass
class PrefilteredVideo:
    video: VideoData
    frame_scores: list[FrameScore] = field(default_factory=list)

    @property
    def candidate_frames(self) -> list[FrameScore]:
        return [f for f in self.frame_scores if f.passes_filter]


# ── Internal ─────────────────────────────────────────────────────────────────
_MODEL_CACHE: "YOLO | None" = None


def _get_model() -> "YOLO":
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        from ultralytics import YOLO  # lazy — see module docstring note
        print(f"[stage3] loading YOLO model: {YOLO_MODEL}")
        _MODEL_CACHE = YOLO(YOLO_MODEL)
    return _MODEL_CACHE


def _batched(items, n):
    for i in range(0, len(items), n):
        yield items[i:i + n]


def _frame_timestamp(frame_path: Path, fps: float) -> float:
    # FFmpeg frame numbering starts at 1
    return (int(frame_path.stem) - 1) / fps


def _load_cache(
    cache_path: Path,
    confidence: float,
    threshold: int,
    model: str,
) -> list[FrameScore] | None:
    """Reuse cached YOLO counts if inference params match; re-derive threshold.

    Invalidates on `confidence` or `model` change — both affect the raw count.
    Threshold is a post-processing decision, so it's re-derived at load time.
    """
    if not cache_path.exists():
        return None
    cached = json.loads(cache_path.read_text())
    if cached.get("confidence") != confidence:
        return None
    if cached.get("model") != model:
        return None
    return [
        FrameScore(
            path=Path(s["path"]),
            timestamp=s["timestamp"],
            person_count=s["person_count"],
            passes_filter=s["person_count"] >= threshold,
        )
        for s in cached["frame_scores"]
    ]


def _save_cache(
    cache_path: Path,
    video_id: str,
    confidence: float,
    threshold: int,
    model: str,
    scores: list[FrameScore],
) -> None:
    cache_path.write_text(json.dumps({
        "video_id": video_id,
        "model": model,
        "confidence": confidence,
        "person_count_min": threshold,
        "frame_scores": [
            {
                "path": str(s.path),
                "timestamp": s.timestamp,
                "person_count": s.person_count,
                "passes_filter": s.passes_filter,
            }
            for s in scores
        ],
    }, indent=2))


# ── Public API ───────────────────────────────────────────────────────────────

def prefilter_video(
    video: VideoData,
    person_count_min: int = PERSON_COUNT_MIN,
    confidence: float = YOLO_CONFIDENCE,
    batch_size: int = BATCH_SIZE,
) -> PrefilteredVideo:
    """Run YOLO on every extracted frame; mark which ones survive the filter."""
    cache_path = video.frames_dir.parent / "prefilter.json"

    cached = _load_cache(cache_path, confidence, person_count_min, YOLO_MODEL)
    if cached is not None:
        print(f"[stage3] {video.video_id}: cached prefilter reused (threshold={person_count_min})")
        result = PrefilteredVideo(video=video, frame_scores=cached)
        _log_reduction(video.video_id, result)
        return result

    model = _get_model()
    fps = float(video.metadata.get("fps_extracted", 1.0))
    scores: list[FrameScore] = []

    print(f"[stage3] {video.video_id}: scoring {len(video.frame_paths)} frames")
    for batch in _batched(video.frame_paths, batch_size):
        results = model.predict(
            source=[str(p) for p in batch],
            conf=confidence,
            classes=[PERSON_CLASS_ID],
            verbose=False,
        )
        for path, r in zip(batch, results):
            # classes=[0] was passed to predict, so every returned box is a person
            count = 0 if r.boxes is None else len(r.boxes)
            scores.append(FrameScore(
                path=path,
                timestamp=_frame_timestamp(path, fps),
                person_count=count,
                passes_filter=count >= person_count_min,
            ))

    _save_cache(cache_path, video.video_id, confidence, person_count_min, YOLO_MODEL, scores)
    result = PrefilteredVideo(video=video, frame_scores=scores)
    _log_reduction(video.video_id, result)
    return result


def _log_reduction(video_id: str, result: PrefilteredVideo) -> None:
    total = len(result.frame_scores)
    kept = len(result.candidate_frames)
    if total == 0:
        return
    reduction = 100 * (1 - kept / total)
    print(f"[stage3] {video_id}: {kept}/{total} candidates ({reduction:.0f}% reduction)")


def run_stage3(
    videos: list[VideoData],
    person_count_min: int = PERSON_COUNT_MIN,
    confidence: float = YOLO_CONFIDENCE,
    batch_size: int = BATCH_SIZE,
) -> list[PrefilteredVideo]:
    """Batch-prefilter a list of prepared videos."""
    results = [
        prefilter_video(v, person_count_min, confidence, batch_size)
        for v in videos
    ]
    total = sum(len(pv.frame_scores) for pv in results)
    kept = sum(len(pv.candidate_frames) for pv in results)
    if total:
        reduction = 100 * (1 - kept / total)
        print(f"[stage3] batch done — {kept}/{total} frames pass ({reduction:.0f}% reduction)")
    return results


# ── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    from stage2_download_prepare import run_stage2

    parser = argparse.ArgumentParser(description="Stage 3: Local person pre-filter")
    parser.add_argument("urls", nargs="+", help="YouTube video URLs (runs Stage 2 first)")
    parser.add_argument("--out", default="./crowd_output", help="Output directory")
    parser.add_argument("--min-persons", type=int, default=PERSON_COUNT_MIN)
    parser.add_argument("--confidence", type=float, default=YOLO_CONFIDENCE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    videos = run_stage2(args.urls, output_dir=args.out)
    prefiltered = run_stage3(
        videos,
        person_count_min=args.min_persons,
        confidence=args.confidence,
        batch_size=args.batch_size,
    )
    for pv in prefiltered:
        print(f"  {pv.video.video_id}: {len(pv.candidate_frames)} candidates")

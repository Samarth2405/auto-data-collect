"""
Stage 7 — Crowd Count
Estimates how many people each segment shows, using DM-Count: a density-map
crowd counter (VGG19 backbone) trained on UCF-QNRF, a dataset of very dense
crowds. YOLO-style detectors find almost nobody from height: they need a
whole visible body, and an aerial shot shows thousands of 5-pixel heads.
Density models are built for exactly that view.

Runs locally: about 0.1 s per frame on a GPU, 2 s on CPU. It only counts frames
inside segments, and caches each frame's count in crowd_counts.json.

The reference doc's Stage 7 (location/entity enrichment) is still deferred;
this slot counts people instead.

Model: Wang et al., "Distribution Matching for Crowd Counting", NeurIPS 2020
(github.com/cvlab-stonybrook/DM-Count, MIT). Weights: converted by the LWCC
project (github.com/tersekmatija/lwcc, MIT).
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from stage2_download_prepare import VideoData
from stage4_gemini_score import ScoredVideo
from stage6_segment_builder import Segment, VideoSegments

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# ── Constants ────────────────────────────────────────────────────────────────
# QNRF over ShanghaiTech-A, chosen on this project's aerial frames: QNRF's counts
# mostly held steady when the input was upscaled 1.5x and 2x while SHA's drifted,
# and a hand-counted patch (~20-30 people) read 23 with QNRF where SHA reads
# ~40% lower. Changing either value invalidates the cache.
COUNT_MODEL = "DM-Count_QNRF"
INPUT_SCALE = 1.0                   # frames are already 1280 px wide
WEIGHTS_URL = "https://github.com/tersekmatija/lwcc_weights/releases/download/v0.1/{}.pth"
WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"
SAVE_EVERY = 25                     # checkpoint cadence — CPU runs are slow
COUNT_METHOD = (f"{COUNT_MODEL} density map (Wang et al. 2020), "
                "every 1 fps frame in the clip; an estimate, not a headcount")


# ── Data contract ────────────────────────────────────────────────────────────
@dataclass
class SegmentCount:
    avg: int                        # mean estimated people over the segment's frames
    peak: int                       # highest single-frame estimate
    frames: int                     # frames counted


@dataclass
class CountedVideo:
    segmented: VideoSegments
    segment_counts: list[SegmentCount | None] = field(default_factory=list)  # aligned with .segments

    # Stage 8 reads these on either a VideoSegments or a CountedVideo
    @property
    def video(self) -> VideoData:
        return self.segmented.video

    @property
    def segments(self) -> list[Segment]:
        return self.segmented.segments

    @property
    def scored(self) -> ScoredVideo:
        return self.segmented.scored


# ── Model ────────────────────────────────────────────────────────────────────
_VGG19 = [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M",
          512, 512, 512, 512, "M", 512, 512, 512, 512]
_model = None
_device = None


def _build_model():
    import torch.nn as nn

    class DMCount(nn.Module):
        """DM-Count: VGG19 features (no last pool) + 3-layer density head."""

        def __init__(self):
            super().__init__()
            layers, c = [], 3
            for v in _VGG19:
                if v == "M":
                    layers.append(nn.MaxPool2d(2, 2))
                else:
                    layers += [nn.Conv2d(c, v, 3, padding=1), nn.ReLU(inplace=True)]
                    c = v
            self.features = nn.Sequential(*layers)
            self.reg_layer = nn.Sequential(
                nn.Conv2d(512, 256, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(inplace=True))
            self.density_layer = nn.Sequential(nn.Conv2d(128, 1, 1), nn.ReLU())

        def forward(self, x):
            x = self.features(x)
            x = nn.functional.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
            return self.density_layer(self.reg_layer(x))

    return DMCount()


def _weights_path() -> Path:
    path = WEIGHTS_DIR / f"{COUNT_MODEL}.pth"
    if not path.exists():
        import torch
        WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        print(f"[stage7] downloading {COUNT_MODEL} weights (86 MB, once) -> {path}")
        tmp = path.with_suffix(".part")
        torch.hub.download_url_to_file(WEIGHTS_URL.format(COUNT_MODEL), str(tmp), progress=True)
        tmp.replace(path)
    return path


def _get_model():
    global _model, _device
    if _model is None:
        import torch
        _device = ("cuda" if torch.cuda.is_available()
                   else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
                   else "cpu")
        model = _build_model()
        # weights_only: the file comes from the internet; refuse pickled code
        state = torch.load(_weights_path(), map_location="cpu", weights_only=True)["model"]
        model.load_state_dict(state)
        _model = model.eval().to(_device)
        print(f"[stage7] {COUNT_MODEL} on {_device}")
    return _model


def count_frame(path: Path) -> float:
    """Estimated number of people in one image (sum of the density map)."""
    import numpy as np
    import torch
    from PIL import Image

    model = _get_model()
    im = Image.open(path).convert("RGB")
    if INPUT_SCALE != 1.0:
        im = im.resize((round(im.width * INPUT_SCALE), round(im.height * INPUT_SCALE)),
                       Image.BILINEAR)
    a = np.asarray(im, dtype=np.float32) / 255.0
    a = (a - np.array([0.485, 0.456, 0.406], dtype=np.float32)) \
        / np.array([0.229, 0.224, 0.225], dtype=np.float32)
    x = torch.from_numpy(a.transpose(2, 0, 1)).unsqueeze(0).to(_device)
    with torch.no_grad():
        return float(model(x).sum())


# ── Internal ─────────────────────────────────────────────────────────────────
def _frame_time(path: Path, fps: float) -> float:
    return (int(path.stem) - 1) / fps          # FFmpeg numbers frames from 1


def _frames_by_segment(video: VideoData, segments: list[Segment]) -> list[list[Path]]:
    """Every extracted frame whose sample moment (t + 0.5/fps) lies inside each segment."""
    fps = float(video.metadata.get("fps_extracted", 1.0))
    timed = [(_frame_time(p, fps) + 0.5 / fps, p) for p in video.frame_paths]
    return [[p for m, p in timed if s.start <= m < s.end] for s in segments]


def _aggregate(frames: list[Path], counts: dict[str, float]) -> SegmentCount | None:
    values = [counts[p.name] for p in frames if p.name in counts]
    if not values:
        return None
    return SegmentCount(avg=round(sum(values) / len(values)), peak=round(max(values)),
                        frames=len(values))


def _load_cache(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}
    if data.get("model") != COUNT_MODEL or data.get("input_scale") != INPUT_SCALE:
        return {}                               # different model: never blend scales
    return data.get("counts", {})


def _save_cache(path: Path, counts: dict[str, float]) -> None:
    path.write_text(json.dumps({
        "model": COUNT_MODEL,
        "input_scale": INPUT_SCALE,
        "counts": {k: round(v, 2) for k, v in sorted(counts.items())},
    }, indent=1))


# ── Public API ───────────────────────────────────────────────────────────────
def count_video(vs: VideoSegments) -> CountedVideo:
    """Count the frames inside each segment; cached per frame in crowd_counts.json."""
    video = vs.video
    per_segment = _frames_by_segment(video, vs.segments)
    cache_path = video.frames_dir.parent / "crowd_counts.json"
    counts = _load_cache(cache_path)

    todo = sorted({p for frames in per_segment for p in frames if p.name not in counts})
    if todo:
        print(f"[stage7] {video.video_id}: counting {len(todo)} frames "
              f"({sum(map(len, per_segment)) - len(todo)} cached)")
    for i, p in enumerate(todo, 1):
        counts[p.name] = count_frame(p)
        if i % SAVE_EVERY == 0:
            _save_cache(cache_path, counts)
    if todo:
        _save_cache(cache_path, counts)

    seg_counts = [_aggregate(frames, counts) for frames in per_segment]
    if seg_counts:
        peaks = [c.peak for c in seg_counts if c]
        print(f"[stage7] {video.video_id}: {len(seg_counts)} segments, "
              f"peak ~{max(peaks) if peaks else 0} people")
    return CountedVideo(segmented=vs, segment_counts=seg_counts)


def run_stage7(videos: list[VideoSegments], enabled: bool = True) -> list[CountedVideo]:
    """Count a batch. One video failing loses only its counts; a model that cannot
    load (no network for the first weights download, say) skips counting for all."""
    if not enabled:
        return [CountedVideo(segmented=vs, segment_counts=[None] * len(vs.segments))
                for vs in videos]
    if any(vs.segments for vs in videos):
        try:
            _get_model()
        except Exception as exc:  # noqa: BLE001 — import, download or load; all mean "no counts"
            print(f"[stage7] crowd counting unavailable ({exc}); continuing without counts")
            return run_stage7(videos, enabled=False)

    results = []
    for vs in videos:
        try:
            results.append(count_video(vs))
        except Exception as exc:  # noqa: BLE001 — one bad frame must not sink the batch
            print(f"[stage7] {vs.video.video_id}: counting failed ({exc}); no counts")
            results.append(CountedVideo(segmented=vs, segment_counts=[None] * len(vs.segments)))
    return results


# ── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    from stage2_download_prepare import run_stage2
    from stage3_prefilter import run_stage3
    from stage4_gemini_score import run_stage4
    from stage6_segment_builder import run_stage6

    parser = argparse.ArgumentParser(description="Stage 7: Estimate people per segment")
    parser.add_argument("urls", nargs="+", help="YouTube video URLs")
    parser.add_argument("--out", default="./crowd_output")
    parser.add_argument("--min-persons", type=int, default=5)
    parser.add_argument("--rpm", type=int, default=15)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--any-vantage", action="store_true")
    args = parser.parse_args()

    videos = run_stage2(args.urls, output_dir=args.out)
    prefiltered = run_stage3(videos, person_count_min=args.min_persons)
    scored = run_stage4(prefiltered, rpm=args.rpm, workers=args.workers)
    segmented = run_stage6(scored, require_elevated=not args.any_vantage)
    for cv in run_stage7(segmented):
        print(f"\n{cv.video.title}")
        for s, c in zip(cv.segments, cv.segment_counts):
            est = f"~{c.avg} people (peak {c.peak})" if c else "no count"
            print(f"  [{s.start_fmt}–{s.end_fmt}]  {est}  yolo≈{s.person_count_avg}")

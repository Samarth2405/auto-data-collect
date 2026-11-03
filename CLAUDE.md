# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Python pipeline that searches YouTube for videos, scores per-frame crowd density with Gemini 3.5 Flash, and produces timestamped high-density segments. Designed to run on the Gemini free tier by replacing paid Google Cloud vision APIs with local YOLO inference.

The reference architecture lives in `crowd_pipeline_v2_documentation.docx`. The implementation intentionally deviates from that doc — see "Deviations" below.

## User-facing docs

- **`README.md`** — installation, configuration, and all CLI usage examples. Point users here first.
- **`GETTING_STARTED.md`** — Windows-first, no-Claude-Code walkthrough for collaborators: install, keys, prompt → `dataset/`, location-slug rules. Keep its command lines in sync with the Stage 1 CLI.
- **`ARCHITECTURE.md`** — deep explanation of each stage (what/why/how), data contract chain, caching strategy, and deviations from the reference doc, with Mermaid + ASCII diagrams.
- `CLAUDE.md` (this file) — quick-reference for Claude Code sessions.

When making non-trivial code changes, verify these docs still match reality — especially the on-disk layout, environment variables, and stage list.

## Repository Layout

```
stage1_search.py             YouTube Data API v3 discovery → DiscoveredVideo
stage2_download_prepare.py   yt-dlp + FFmpeg → VideoData
stage3_prefilter.py          local YOLOv8n person counter → PrefilteredVideo
stage4_gemini_score.py       Gemini 3.5 Flash scorer (density + camera vantage) → ScoredVideo
stage6_segment_builder.py    gap-merge high-density frames → VideoSegments
stage8_output.py             batch JSON/CSV + optional FFmpeg clips + Getty-style dataset export
discover_india.py            India-wide candidate discovery + tiering → candidates/india_candidates.json
run_india_batch.py           runs the best tier-A candidates through stages 2..8, one location at a time
tune.py                      re-segment from cache with new threshold/gap
test_smoke.py                pure-function tests (no network, no API keys)
requirements.txt             pinned dependencies
.claude/commands/            /test-stage, /cost-estimate, /tune-threshold
crowd_pipeline_v2_documentation.docx   reference architecture (not code)
```

Stage 7 (enrichment) is deferred — see the "Deviations" section. Each stage is a standalone module with its own CLI (`python stageN_*.py --help`). Running `stageN` in the CLI implicitly runs stages 2..N-1 as prerequisites. `stage1_search.py --full-pipeline` runs everything from a search query in one command; `stage8_output.py` is the URL-based end-to-end entry point.

## Data Contract

Each stage produces a dataclass that composes the previous stage's output:

```
DiscoveredVideo   (stage 1)  →  VideoData         (stage 2)  →  PrefilteredVideo (stage 3)  →  ScoredVideo (stage 4)  →  VideoSegments (stage 6)
  video_id                       video_id                       .video: VideoData             .prefiltered: PrefilteredVideo   .scored: ScoredVideo
  url                            url                            .frame_scores                 .gemini_scores                   .segments
  duration_secs                  frame_paths                    .candidate_frames             .high_density_frames             .video (shortcut)
  view_count                     metadata                         (filtered subset)             (score >= CROWD_THRESHOLD)
```

Stage 1 → Stage 2 handoff is just URL extraction: `[v.url for v in discovered]`. All other stages compose the previous output.

Downstream stages should reach through by attribute, not by re-parsing metadata.

## On-Disk Layout

Every video gets its own directory under `--out` (default `./crowd_output`):

```
crowd_output/
  discovery.json         Stage 1 search results (per query, overwrites each run)
  results.json           Stage 8 batch aggregate (all videos)
  segments.csv           Stage 8 flat segment table (one row per segment)
  clips/                 Stage 8 per-segment MP4s (only with --export-clips)
  {video_id}/
    video.mp4            yt-dlp download
    frames/000001.jpg …  FFmpeg 1-indexed frame dump
    metadata.json        curated yt-dlp info + extraction params
    .stage2_done         sentinel: extraction completed cleanly
    prefilter.json       Stage 3 YOLO counts + threshold (invalidates on model/confidence change)
    gemini_scores.json   Stage 4 per-frame crowd score + reasoning (invalidates on model change)
    segments.json        Stage 6 merged time segments + yt_link deep-links
```

**All caches are keyed by frame path.** Stage 3 re-derives `passes_filter` from the current threshold without re-running YOLO. Stage 4 skips any frame already scored *by the same model* — changing `GEMINI_MODEL` discards the cache rather than blending two models' scales. This means tuning `PERSON_COUNT_MIN` or `CROWD_THRESHOLD` is nearly free on reruns.

## Running

```bash
pip install -r requirements.txt
# Ensure FFmpeg is on PATH: winget install ffmpeg  (or brew/apt)
export YOUTUBE_API_KEY="AIza..."
export GEMINI_API_KEY="AI..."

# Full pipeline from a search query (stages 1..8)
python stage1_search.py "Times Square New Year crowd" --max-results 5 --full-pipeline

# Full pipeline from explicit URLs (stages 2..8)
python stage8_output.py "https://www.youtube.com/watch?v=XXXX" --export-clips

# Discovery only — writes discovery.json without downloading anything
python stage1_search.py "Times Square crowd" --max-results 10

# Iterate on threshold without re-running Gemini (free)
python tune.py --threshold 8 --gap-secs 5

# Verify pure-function correctness across all stages (no network)
python test_smoke.py
```

## Key Constants

| Constant | Default | Location | Effect |
|----------|---------|----------|--------|
| `MAX_RESULTS` | `20` | stage1 | YouTube search page cap (max 50) |
| `MIN/MAX_DURATION_SECS` | `30`/`3600` | stage1 | Duration filter — excludes shorts and multi-hour uploads |
| `FRAME_RATE` | `1.0` | stage2 | FPS to extract |
| `RESIZE_WIDTH` | `1280` | stage2 | JPEG width |
| `MAX_HEIGHT` | `720` | stage2 | Download resolution cap — frames are downsampled anyway, so 4K is ~9x the bytes for identical scores |
| `PERSON_COUNT_MIN` | `5` | stage3 | YOLO count to pass pre-filter |
| `YOLO_CONFIDENCE` | `0.3` | stage3 | Per-detection confidence — changing this invalidates stage3 cache |
| `YOLO_MODEL` | `yolov8n.pt` | stage3 | Model weights — also invalidates cache |
| `CROWD_THRESHOLD` | `7` | stage4 | Gemini score to count as high-density |
| `FREE_TIER_RPM` | `15` | stage4 | Gemini requests per minute |
| `DEFAULT_WORKERS` | `1` | stage4 | Concurrent Gemini calls (`--workers`); `--rpm` still caps the total. One sequential worker tops out near 30 RPM (~2 s/call) |
| `MIN_SEGMENT_SECS` | `3.0` | stage6 | Max gap between hits before splitting a segment |

## Elevated-vantage collection

The India dataset keeps only crowds filmed from height. Stage 1 biases by metadata (`_vantage_hint`); Stage 4 has Gemini label each frame's `vantage` (ground/elevated/aerial/unknown); Stage 6 keeps only elevated/aerial high-density frames (`--any-vantage` disables). The Stage 6 gate is the authoritative one — metadata hints are priors and have produced videos with zero dense frames.

**Use `--min-persons 0` for this work.** YOLOv8n cannot detect people from above, so the default pre-filter drops exactly the frames wanted. Stage 8 `--dataset DIR --location SLUG` writes `DIR/<location>/<vantage>/IND_<CODE>_NNNN/{clip.mp4,poster.jpg,metadata.json}` plus `DIR/manifest.csv`; `rights_cleared` is always `false`.

## Deviations from `crowd_pipeline_v2_documentation.docx`

The doc describes a cloud-native pipeline. This implementation trims it for zero-cost, local-first operation:

- **Stage 3 replaced.** Video Intelligence API + Cloud Vision API → local YOLOv8n. Free-tier Cloud Vision caps at 1,000 units/month, which is exceeded within a few videos. YOLO runs unlimited on CPU.
- **Stage 5 dropped.** Speech-to-Text is 60 min/month free — insufficient at any scale. Audio cues from YouTube captions (available via yt-dlp) can be substituted if needed for Stage 6.
- **Stage 8 kept local.** JSON/CSV output only; BigQuery integration deferred.
- **Gemini stays cloud.** Stage 4 is the sole paid dependency (and stays free under 15 RPM).

Do not add Google Cloud dependencies back without a clear cost/rate justification.

## Conventions

- Dataclasses use composition, not inheritance. Never mutate the wrapped stage output.
- Cache files sit next to `metadata.json`, not in a separate cache dir. This keeps all state for one video in one folder.
- Caches invalidate on *inference-affecting* parameters (model, confidence) but re-derive *post-processing* decisions (thresholds) at load time. Follow the same pattern when adding new caches.
- Failure of a single video should never abort a batch. Return `None` from single-video functions; filter at the batch layer.
- Account-wide failures are the one exception: Stage 4 raises `QuotaExhausted` (spend cap, daily quota, bad key) so `run_stage4` stops the batch and returns what finished, instead of reproducing the same error on every remaining video. Classify with `_is_terminal` — transient errors still back off and retry. Checkpoint partial work before either exit.

# Crowd Density Extraction Pipeline

Search YouTube for videos, score every frame for crowd density with Gemini 3.5 Flash, and export timestamped high-density segments as JSON, CSV, and MP4 clips.

Designed to run on **free-tier APIs only**. Local YOLOv8-nano replaces Google Cloud Vision (which would exceed its 1K-units/month free tier within ~5 videos). Gemini stays within its 15 RPM free quota. Zero cloud spend by default.

For the *why* behind each design choice and diagrams of the internals, see [ARCHITECTURE.md](ARCHITECTURE.md).

**New collaborator collecting India elevated-crowd data?** Start with [GETTING_STARTED.md](GETTING_STARTED.md). It's a step-by-step setup and prompt-to-dataset guide that doesn't need Claude Code.

---

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Quick Start](#quick-start)
- [Detailed Usage](#detailed-usage)
- [Slash Commands](#slash-commands)
- [Output Format](#output-format)
- [Troubleshooting](#troubleshooting)
- [Cost & Performance](#cost--performance)

---

## Requirements

- **Python 3.10+** (uses `X | None` union syntax)
- **FFmpeg** on `PATH`
- Two free API keys:
  - **YouTube Data API v3** — get one at [console.cloud.google.com](https://console.cloud.google.com) → enable "YouTube Data API v3" → Credentials → API Key
  - **Gemini API** — get one at [aistudio.google.com](https://aistudio.google.com/apikey)

---

## Installation

```bash
git clone <repo-url>
cd crowd_dataset_algo
pip install -r requirements.txt
```

Install FFmpeg:

| OS | Command |
|----|---------|
| Windows | `winget install ffmpeg` |
| macOS | `brew install ffmpeg` |
| Linux (Debian/Ubuntu) | `sudo apt install ffmpeg` |

Verify it's on `PATH`:
```bash
ffmpeg -version
```

**Verify the install works** — the smoke test runs 41 pure-function assertions across all stages without any network access:

```bash
python test_smoke.py
```

Expected output:
```
Running 7 smoke tests...
  stage1  iso8601 duration : 9 cases OK
  stage2  video_id regex   : 9 cases OK
  stage6  time formatting  : 9 cases OK
  ...
All smoke tests passed.
```

---

## Configuration

Set two environment variables.

**bash / zsh (macOS/Linux):**
```bash
export YOUTUBE_API_KEY="AIza..."
export GEMINI_API_KEY="AI..."
```

**PowerShell (Windows):**
```powershell
$env:YOUTUBE_API_KEY = "AIza..."
$env:GEMINI_API_KEY = "AI..."
```

**cmd.exe:**
```cmd
set YOUTUBE_API_KEY=AIza...
set GEMINI_API_KEY=AI...
```

To persist across sessions on Windows, use "System Properties → Environment Variables" or `[Environment]::SetEnvironmentVariable("YOUTUBE_API_KEY", "AIza...", "User")` in PowerShell.

---

## Quick Start

Run the full pipeline from a search query:

```bash
python stage1_search.py "Times Square New Year crowd" --max-results 3 --full-pipeline
```

This executes stages 1 → 8:
1. Search YouTube for 3 matching videos
2. Download each and extract 1 frame per second
3. Local YOLO filters frames without at least 5 people
4. Gemini scores each surviving frame 1–10 for crowd density
5. Merge consecutive high-scoring frames into time segments
6. Write `results.json` and `segments.csv` to `./crowd_output/`

The final output looks like:
```
crowd_output/
├─ discovery.json    # Stage 1 search results
├─ results.json      # aggregated segments across all videos
├─ segments.csv      # flat table for spreadsheet analysis
└─ {video_id}/       # per-video artifacts
```

---

## Detailed Usage

### Search only (no download)

```bash
python stage1_search.py "marathon finish line crowd" --max-results 10
```

Writes `crowd_output/discovery.json` with ten ranked videos. Costs ~101 YouTube quota units.

### Full pipeline with MP4 clips per segment

```bash
python stage1_search.py "concert moshpit" --max-results 5 --full-pipeline --export-clips
```

Adds per-segment clips under `crowd_output/clips/`, named `{video_id}_{start}-{end}.mp4`.

### From explicit URLs (skip Stage 1)

```bash
python stage8_output.py \
  "https://www.youtube.com/watch?v=abc123XXXXX" \
  "https://www.youtube.com/watch?v=xyz456YYYYY" \
  --export-clips
```

### Run one stage at a time

Each stage's CLI runs all its prerequisites first. Useful for debugging:

```bash
python stage2_download_prepare.py <url>    # download + frame extraction
python stage3_prefilter.py <url>            # + YOLO person counting
python stage4_gemini_score.py <url>         # + Gemini scoring
python stage6_segment_builder.py <url>      # + segment merging
python stage8_output.py <url>               # full pipeline (2..8)
```

### Tune thresholds without re-running Gemini (free)

Stage 4 caches every score. Iterating on segmentation thresholds only re-runs Stages 6 and 8:

```bash
python tune.py --threshold 8 --gap-secs 5
```

No API calls made. Rewrites `results.json` from cached data.

### Change frame sampling rate

```bash
python stage2_download_prepare.py <url> --fps 2.0    # 2 frames per second
```

Doubles the number of frames extracted (and roughly doubles Gemini calls). Default is `--fps 1.0`.

### Change download resolution

```bash
python stage2_download_prepare.py <url> --max-height 480    # smaller download
```

Downloads are capped at 720p by default. Frames are rescaled to `RESIZE_WIDTH` (1280 px) before
anything reads them and YOLO sees 640 px, so pulling 4K costs roughly 9x the bytes and disk for
identical scores — a 30-minute upload is ~5.7 GB at 2160p versus ~0.7 GB at 720p. Raise it only if
you plan to export clips at full quality with `--export-clips`.

### Change ranking order

Stage 1 supports YouTube's built-in ranking:

```bash
python stage1_search.py "protest crowd" --order viewCount --max-results 5
python stage1_search.py "flash mob 2026" --order date --max-results 5
```

Options: `relevance` (default), `viewCount`, `date`, `rating`.

### Skip the YOLO pre-filter (send every frame to Gemini)

Set `--min-persons 0` — this counts as passing everything through:

```bash
python stage4_gemini_score.py <url> --min-persons 0
```

Warning: at 1 fps a 3-min video makes ~180 Gemini calls (~12 min at 15 RPM). Use it for validation
and for aerial/elevated footage (next section), where YOLO cannot find people.

### Elevated crowds and the Getty-style dataset

To collect crowds filmed from height (drone, rooftop, overbridge, balcony) rather than at street level:

```bash
python stage1_search.py "Kumbh Mela crowd aerial drone" --max-results 6 --drop-ground \
    --min-persons 0 --full-pipeline --out crowd_output_india/prayagraj-kumbh-mela \
    --dataset dataset --location prayagraj-kumbh-mela
```

Vantage is filtered twice:

- **Stage 1** reads title/description/tags for height markers (`_vantage_hint`): `--require-elevated`
  keeps only `likely_elevated` videos, `--drop-ground` drops `likely_ground` ones. This is a cheap bias, not a gate.
- **Stage 4** has Gemini label each frame `ground` / `elevated` / `aerial` / `unknown` by geometry
  (viewing angle, sky, tops of heads). **Stage 6 keeps only elevated/aerial high-density frames** — this is the
  real gate. It is cached, so changing it is free. `--any-vantage` (stages 6, 8) keeps ground-level segments too.

Always pass `--min-persons 0` here: from directly above, YOLOv8n finds few or no people in genuinely dense crowds,
so the pre-filter would discard exactly the frames you want.

`--dataset DIR --location SLUG` makes Stage 8 export a Getty-style tree, and merges into `DIR/manifest.csv` across runs:

```
dataset/<location>/<vantage>/IND_<CODE>_NNNN/{clip.mp4, poster.jpg, metadata.json}
```

Every `metadata.json` carries `"rights_cleared": false`. These are YouTube uploads; confirm licensing before redistributing.

**Searching all of India.** `discover_india.py` runs ~60 queries (festivals, protests, rallies, transit, markets…) with
`regionCode=IN`, tiers the results (A = metadata signals both height and a crowd) and writes
`candidates/india_candidates.json|csv`. Protest and rally queries are limited to the last 365 days. It costs
~6,000 of the 10,000 daily YouTube quota units; `--rank-only` re-ranks cached results for free and `--dry-run`
prints the plan. `run_india_batch.py` then runs the best tier-A videos location by location (`--dry-run` to preview).
Tiers are metadata priors — Gemini's per-frame vantage and density remain the real filter.

---

## Slash Commands

If using [Claude Code](https://claude.com/claude-code) in this project, three slash commands are available:

| Command | What it does |
|---------|-------------|
| `/test-stage <N> <url>` | Runs stage N against one video and validates the output schema |
| `/cost-estimate <urls...>` | Probes durations, estimates Gemini calls + processing time |
| `/tune-threshold [--threshold T] [--gap-secs G]` | Runs `tune.py` and reports before/after segment counts |

Skill definitions live in `.claude/commands/*.md`.

---

## Output Format

### On-disk layout

```
crowd_output/
├─ discovery.json                  Stage 1 — search results (per query)
├─ results.json                    Stage 8 — batch aggregate across all videos
├─ segments.csv                    Stage 8 — flat segment table (one row per segment)
├─ clips/                          Stage 8 — per-segment MP4s (with --export-clips)
│  └─ {video_id}_0045-0112.mp4
└─ {video_id}/
   ├─ video.mp4                    Stage 2 — yt-dlp download
   ├─ frames/000001.jpg …          Stage 2 — extracted frames
   ├─ metadata.json                Stage 2 — video metadata + extraction params
   ├─ .stage2_done                 Stage 2 — sentinel: extraction succeeded
   ├─ prefilter.json               Stage 3 — YOLO person counts per frame
   ├─ gemini_scores.json           Stage 4 — Gemini score + reasoning per frame
   └─ segments.json                Stage 6 — merged time segments
```

### `results.json` shape

```json
{
  "generated_at": "2026-07-22T12:00:00+00:00",
  "video_count": 3,
  "total_segments": 8,
  "videos": [
    {
      "video_id": "dQw4w9WgXcQ",
      "title": "Times Square NYE 2026",
      "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
      "duration": 892.0,
      "location": null,
      "entities": [],
      "high_density_segments": [
        {
          "start": 45.0,
          "end": 112.0,
          "start_fmt": "0:45",
          "end_fmt": "1:52",
          "frame_count": 68,
          "avg_crowd_score": 8.6,
          "peak_crowd_score": 9,
          "person_count_avg": 87.3,
          "person_count_peak": 105,
          "signal_sources": ["gemini", "yolo"],
          "yt_link": "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=45s",
          "reasoning_samples": [
            "Dense crowd fills the frame with visible individual faces...",
            "Extreme density — no personal space visible..."
          ]
        }
      ]
    }
  ]
}
```

### `segments.csv` columns

```
video_id, title, url, segment_index, start, end, duration_secs,
start_fmt, end_fmt, frame_count, avg_crowd_score, peak_crowd_score,
person_count_avg, person_count_peak, yt_link
```

One row per detected segment. Import directly into pandas, Excel, or Google Sheets.

---

## Troubleshooting

### `ffmpeg not found. Install it and ensure it is on PATH.`

FFmpeg isn't on `PATH`. Install with the command from the [Installation](#installation) section, then restart your terminal so `PATH` is refreshed.

### `YOUTUBE_API_KEY environment variable not set`

The env var isn't exported. See [Configuration](#configuration).

### YouTube API returns HTTP 403 quotaExceeded

You've used all 10,000 daily units. The quota resets at midnight Pacific Time. Wait, or request a higher quota in the Cloud Console.

### YouTube API returns HTTP 403 accessNotConfigured

You haven't enabled the YouTube Data API v3 on your Google Cloud project. Go to Cloud Console → APIs & Services → enable "YouTube Data API v3".

### Gemini scoring is slow

At the free tier (15 RPM), 100 candidate frames take ~7 minutes. Options:
- `--rpm 300 --workers 8` (requires paid tier). Calls take ~2 s each, so one worker tops out near 30 RPM no matter how high `--rpm` is; `--workers` runs calls concurrently while `--rpm` still caps the total
- Raise `--min-persons` in Stage 3 to reduce candidates
- Reduce `--fps` in Stage 2 (1 fps → 0.5 fps halves the load)

### `ultralytics` install is huge (~500 MB)

That's PyTorch, which YOLO depends on. Necessary — no workaround unless you swap YOLO for a lighter detector.

### Videos won't download

YouTube changes internal formats frequently. Update:
```bash
pip install -U yt-dlp
```

### Gemini returns "no parsed response" for many frames

Safety filter is blocking. Rare, but happens with e.g. violent crowds. The pipeline treats these as score-1 sentinels and continues. Check `gemini_scores.json` for how many frames were affected.

### Re-running the pipeline re-does everything

It shouldn't — every stage caches. If it's re-downloading, check that `.stage2_done` exists in the video dir. If it's re-scoring with Gemini, check that `gemini_scores.json` exists and isn't corrupted.

---

## Cost & Performance

### Per 3-minute video at 1 fps

| Stage | Free tier | Paid tier |
|-------|-----------|-----------|
| Search + metadata | 101 YouTube quota units | 101 units |
| Download + frames | Free (local) | Free |
| YOLO filter | Free (local) | Free |
| Gemini scoring (~40 candidates) | 3 min @ 15 RPM | 40 sec @ 60 RPM |
| Segment + output | Free (local) | Free |
| **Total time** | **~3 min** | **~1 min** |
| **Total cost** | **$0** | **~$0.04** |

### 30-video batch (free tier)

Stays comfortably within free quotas:
- YouTube: ~3,030 units (30% of the 10K/day cap)
- Gemini: ~1,200 candidate frames = 80 min at 15 RPM
- Everything else: local, unlimited

### Estimating for your batch

Use the `/cost-estimate` slash command:
```
/cost-estimate <url1> <url2> ...
```

Or manually: for each video, `gemini_calls ≈ ceil(duration_secs × 0.25)`. Time at 15 RPM = `total_calls / 15` minutes.

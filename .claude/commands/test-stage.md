Test a single pipeline stage against one YouTube video and validate its output.

Arguments: $ARGUMENTS
Expected format: <stage_number> <youtube_url> [--out DIR]
Example: `3 https://www.youtube.com/watch?v=XXXX`

If $ARGUMENTS is empty or missing a stage number or URL, ask the user to provide them before proceeding.

## Steps

1. Parse the stage number (2, 3, 4, 6, or 8) and URL from the arguments.
   Use `--out ./test_output` if no output dir was given.

2. **Resolve video_id** from the URL before running anything — this is needed to
   find the output directory in later steps. Use this regex on the URL:
   `(?:youtu\.be/|youtube\.com/(?:watch\?v=|embed/|shorts/|v/))([A-Za-z0-9_-]{11})`
   If the URL doesn't match, extract video_id from the CLI output in step 3
   (look for a line matching `[stage2] <video_id>:`).

3. Run the stage CLI. Note: running Stage N triggers Stages 2..N-1 as prerequisites,
   which may download the video on first run.
   - Stage 2: `python stage2_download_prepare.py <url> --out <out>`
   - Stage 3: `python stage3_prefilter.py <url> --out <out>`
   - Stage 4: `python stage4_gemini_score.py <url> --out <out>`
   - Stage 6: `python stage6_segment_builder.py <url> --out <out>`
   - Stage 8: `python stage8_output.py <url> --out <out>`

4. Validate stage-specific output. Read the relevant file from `<out>/<video_id>/`:

   **Stage 2** — check `frames/` contains `.jpg` files and `.stage2_done` exists.
   - PASS if both conditions met.
   - Report: frame count, video duration from `metadata.json`.

   **Stage 3** — read `prefilter.json`. Compute:
   - `total_frames` = len(frame_scores)
   - `candidates` = count where passes_filter == true
   - `reduction_pct` = (1 - candidates / total_frames) × 100
   - PASS if `candidates < total_frames` (YOLO filtered at least one frame).
   - WARN (not FAIL) if `reduction_pct < 20%`: threshold `PERSON_COUNT_MIN` may be too low.
   - WARN (not FAIL) if `reduction_pct > 95%`: threshold too high, almost nothing reaches Gemini.
   - Report: total_frames, candidates, reduction_pct.

   **Stage 4** — read `gemini_scores.json`. For every entry check:
   - `crowd_score` is an integer in [1, 10].
   - `reasoning` does not start with `"no parsed response"` (sentinel for blocked responses).
   - PASS if all scores in range. WARN (not FAIL) if sentinel count > 0.
   - Report: score min / mean / max, sentinel count.

   **Stage 6** — read `segments.json`. For each segment check `start < end` and `start >= 0`.
   - PASS if all segments valid.
   - Report: segment count, min/max duration, total crowd seconds.

   **Stage 8** — check `<out>/results.json` is valid JSON and `<out>/segments.csv` has
   at least a header row.
   - PASS if both files are readable.
   - Report: total_segments from results.json.

5. Print a PASS / WARN / FAIL line per check with the specific numbers.

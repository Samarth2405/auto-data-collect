Estimate Gemini API calls and processing time for a list of YouTube URLs before running the pipeline.

Arguments: $ARGUMENTS — one or more space-separated YouTube URLs.
Example: `https://youtube.com/watch?v=A https://youtube.com/watch?v=B`

If $ARGUMENTS is empty, ask the user to provide at least one YouTube URL before proceeding.

## Steps

1. **Resolve video_id and get duration for each URL.**

   First, parse video_id from the URL without any network call using this regex:
   `(?:youtu\.be/|youtube\.com/(?:watch\?v=|embed/|shorts/|v/))([A-Za-z0-9_-]{11})`

   Then check for a cached duration (avoids a network call):
   - If `crowd_output/<video_id>/metadata.json` exists, read `duration` from it.
   - If the regex didn't match (e.g. short URL) or no cache exists, probe with yt-dlp:
     ```
     python -c "import yt_dlp, sys; ydl=yt_dlp.YoutubeDL({'quiet':True,'no_warnings':True}); info=ydl.extract_info(sys.argv[1],download=False); print(info.get('title','?'), info.get('duration',0))" "<url>"
     ```

2. **Per-video estimates.** Constants (from the codebase):
   - `FRAME_RATE = 1.0` fps (stage2)
   - YOLO pass rate: if `crowd_output/<video_id>/prefilter.json` exists, compute the
     actual observed rate: `passes_true / total_frames`. Otherwise assume **25%**
     (midpoint of the 20–40% typical range). Note the source in the table.

   Compute for each video:
   - `total_frames = ceil(duration × FRAME_RATE)`
   - `yolo_candidates = ceil(total_frames × pass_rate)`
   - `gemini_calls = yolo_candidates`

3. **Batch totals.**
   - `total_calls = sum(gemini_calls)`
   - `time_at_15rpm = total_calls / 15` minutes  ← free-tier
   - `time_at_60rpm = total_calls / 60` minutes  ← paid-tier
   - Cost (paid tier only, approximate): `total_calls × $0.0009`
     [~$0.36 per 400 frames based on Gemini 2.5 Flash pricing — stale: re-check rates for Gemini 3.5 Flash]

4. **Print a table:**

   | # | Title (truncated 40 chars) | Duration | Frames | Pass rate | Gemini calls |
   |---|----------------------------|----------|--------|-----------|--------------|
   | 1 | ...                        | 3:12     | 192    | 25% (est) | 48           |
   | **Total** |                   |          |        |           | **NNN**      |

   Then print a summary block:
   ```
   Processing time  @ 15 RPM (free):  X min
   Processing time  @ 60 RPM (paid):  X min
   Estimated cost   @ paid tier:      $X.XX  (free tier = $0)
   ```

5. If free-tier time > 60 minutes, add:
   "Consider running overnight or using --rpm 60 (paid tier, ~$X.XX)."

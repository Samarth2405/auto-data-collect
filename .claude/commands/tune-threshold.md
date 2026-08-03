Re-run Stage 6 and Stage 8 with a different CROWD_THRESHOLD or MIN_SEGMENT_SECS
using already-cached Gemini scores. No API calls are made — only re-segmentation
and re-export. Useful for iterating on "what counts as high density."

Arguments: $ARGUMENTS
Format: [--threshold T] [--gap-secs G] [--out DIR]
Defaults: --threshold 7  --gap-secs 3.0  --out ./crowd_output

If $ARGUMENTS is empty, ask the user which parameter(s) they want to change before proceeding.

## Steps

1. Parse `--threshold`, `--gap-secs`, and `--out` from the arguments.
   Use defaults for any flag not provided.

2. **Before** running, read `<out>/results.json` if it exists and record:
   - `before_total` = `total_segments` field (0 if file doesn't exist yet)
   - Per-video: video_id, title, segment count from each entry in `videos[].high_density_segments`

3. Run the tune helper:
   ```
   python tune.py --threshold <T> --gap-secs <G> --out <out>
   ```

4. **After** completion, read the new `<out>/results.json` and compute:
   - `after_total` = new `total_segments`
   - Delta per video: `before → after`

5. Print a comparison table:

   | Video ID | Title           | Before | After | Δ  |
   |----------|-----------------|--------|-------|----|
   | abc123   | Times Square... | 4      | 6     | +2 |

   Then a summary line:
   `Total segments: <before_total> → <after_total> (Δ<signed_delta>)`

6. Sanity checks — print a warning if any of these apply:
   - Threshold was **lowered** and segment count **more than doubled**:
     "Threshold may be too permissive — many low-confidence frames are now included."
   - Threshold was **raised** and `after_total == 0`:
     "No segments remain. Consider lowering threshold or reducing --gap-secs."
   - `gap-secs` was set **above 30**:
     "Large gap values can merge distant events into one segment — verify results manually."

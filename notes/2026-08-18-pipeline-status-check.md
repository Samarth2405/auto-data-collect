# Pipeline status check — 2026-08-18

## Question
Are any frames/videos stuck in the pipeline and not yet in `crowd_output/`?

## Method
Checked every video directory under `crowd_output/` for the completion
markers used by each stage (`.stage2_done`, `prefilter.json`,
`gemini_scores.json`, `segments.json`), then cross-checked against the
batch outputs `results.json` and `segments.csv`.

## Result: nothing stuck

| video_id | .stage2_done | prefilter.json | gemini_scores.json | segments.json | in results.json | in segments.csv |
|---|---|---|---|---|---|---|
| 2raeKviCZF4 | yes | yes | yes | yes | yes | yes (19 segments) |
| 4h7tXm5b5KM | yes | yes | yes | yes | yes | yes (2 segments) |
| a0Ud85Xdxn4 | yes | yes | yes | yes | yes | yes (1 segment) |
| aMPoOSaAjuc | yes | yes | yes | yes | yes | yes (24 segments) |
| p-PmleOtRYY | yes | yes | yes | yes | yes | yes (20 segments) |

`stage4_resume2.log` confirms the last Gemini scoring run finished cleanly:
"batch done — 333/459 frames marked high-density". All 5 videos are fully
processed end-to-end and present in the final batch outputs (67 total
segment rows in `segments.csv`).

## Side notes

- `discovery.json` only listed 3 videos for the latest search query
  ("indian crowd") at the time of this check. That's expected — per
  CLAUDE.md, `discovery.json` is a per-query Stage 1 result file that gets
  overwritten on each run. It is not an indicator of missing/stuck work;
  don't use it to judge pipeline completeness.
- An untracked `crowd_output.zip` (~970MB, dated 2026-08-14) is sitting in
  the repo root (shows as `??` in `git status`). It looks like a manual
  export/backup rather than something the pipeline produces — no action
  was taken on it, still pending the user's call on keep/move/delete.

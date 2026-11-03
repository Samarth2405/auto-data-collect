"""
Run the best India-wide candidates (from discover_india.py) through the pipeline.

Picks tier-A videos, caps how many come from one location so the batch spans
festivals, rallies and protests rather than one city, then runs stages 2..8
once per location so each lands under the right dataset/<location>/ folder.

    python run_india_batch.py --dry-run          # show the plan, spend nothing
    python run_india_batch.py                    # run it (paid-tier defaults)
    python run_india_batch.py --rpm 15 --workers 1   # free tier

Safe to rerun: every stage caches per video, so finished work is skipped.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

CANDIDATES = Path("candidates/india_candidates.json")
OUT_ROOT = Path("crowd_output_india")
LOG = Path("notes/2026-10-19-india-batch.log")
STOP_MARKER = "[stage4] stopping batch"

# Metadata tiering cannot see where a video was shot; these were found by eye.
NOT_INDIA = {
    "g2Gruqos_R0",   # England vs India at Essex Cricket Ground — an English crowd
}


def select(rows: list[dict], max_secs: int, per_location: int, limit: int) -> list[dict]:
    pool = [r for r in rows if r["tier"] == "A" and r["duration_secs"] <= max_secs
            and r["video_id"] not in NOT_INDIA]
    pool.sort(key=lambda r: (r["score"], r["published_at"]), reverse=True)
    taken: dict[str, int] = defaultdict(int)
    picked: list[dict] = []
    for r in pool:
        if taken[r["location"]] >= per_location:
            continue
        taken[r["location"]] += 1
        picked.append(r)
        if len(picked) >= limit:
            break
    return picked


def run_location(location: str, videos: list[dict], args: argparse.Namespace, log) -> bool:
    """Run one location. Returns False if the API quota ran out (stop the batch)."""
    cmd = [
        sys.executable, "stage8_output.py", *[v["url"] for v in videos],
        "--out", str(OUT_ROOT / location),
        "--min-persons", "0",          # YOLO cannot see people from above
        "--rpm", str(args.rpm), "--workers", str(args.workers),
        "--dataset", "dataset", "--location", location,
    ]
    # Unbuffered: piped stdout is block-buffered otherwise, hiding progress and
    # delaying the quota-stop marker by several KB of output.
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace", env=env)
    out_of_quota = False
    for line in proc.stdout:
        log.write(line)
        log.flush()
        if STOP_MARKER in line:
            out_of_quota = True
    proc.wait()
    return not out_of_quota


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--max-secs", type=int, default=300, help="skip videos longer than this")
    ap.add_argument("--per-location", type=int, default=4)
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--rpm", type=int, default=300)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rows = json.loads(CANDIDATES.read_text(encoding="utf-8"))
    picked = select(rows, args.max_secs, args.per_location, args.limit)

    by_loc: dict[str, list[dict]] = defaultdict(list)
    for r in picked:
        by_loc[r["location"]].append(r)
    # Strongest locations first, so a spend cap costs the least promising ones.
    order = sorted(by_loc, key=lambda loc: max(v["score"] for v in by_loc[loc]), reverse=True)

    mins = sum(r["duration_secs"] for r in picked) / 60
    print(f"{len(picked)} videos, {len(order)} locations, {mins:.0f} min "
          f"= ~{round(mins * 60 * 0.95)} Gemini calls "
          f"(rpm={args.rpm}, workers={args.workers})")
    for loc in order:
        for v in by_loc[loc]:
            print(f"  {loc:30s} {v['category']:11s} {v['duration_secs']:4d}s  "
                  f"{v['published_at'][:10]}  {v['title'][:60].encode('ascii', 'replace').decode()}")
    if args.dry_run:
        return 0

    LOG.parent.mkdir(exist_ok=True)
    with LOG.open("a", encoding="utf-8") as log:
        log.write(f"\n=== INDIA BATCH {time.strftime('%Y-%m-%d %H:%M:%S')} "
                  f"{len(picked)} videos ===\n")
        for i, loc in enumerate(order, 1):
            log.write(f"\n=== {loc} ({i}/{len(order)}) {time.strftime('%H:%M:%S')} ===\n")
            print(f"[{i}/{len(order)}] {loc}: {len(by_loc[loc])} video(s)", flush=True)
            if not run_location(loc, by_loc[loc], args, log):
                print(f"API quota exhausted at {loc} — stopping. Rerun to resume from cache.")
                log.write("=== STOPPED: quota exhausted ===\n")
                return 2
        log.write(f"\n=== INDIA BATCH FINISHED {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())

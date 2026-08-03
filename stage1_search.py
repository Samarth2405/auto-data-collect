"""
Stage 1 — Video Discovery
Searches YouTube via Data API v3, fetches full metadata, filters by
duration and live-stream status, saves discovery.json.

Uses ~101 quota units per query (100 for search + 1 for batched videos.list).
Free tier is 10,000 units/day → ~99 queries/day headroom.
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from dotenv import load_dotenv

load_dotenv()
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# ── Constants ────────────────────────────────────────────────────────────────
MAX_RESULTS = 20            # YouTube search page cap is 50; we default to 20
MIN_DURATION_SECS = 30      # skip trailers, shorts, thumbnails
MAX_DURATION_SECS = 3600    # skip multi-hour uploads (livestream VODs, docs)
API_TIMEOUT_SECS = 30

SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
VIDEOS_URL = "https://www.googleapis.com/youtube/v3/videos"


# ── Data contract ────────────────────────────────────────────────────────────
@dataclass
class DiscoveredVideo:
    video_id: str
    url: str
    title: str
    description: str
    channel: str
    duration_secs: float
    view_count: int
    published_at: str
    thumbnail: str
    tags: list[str] = field(default_factory=list)


# ── Internal ─────────────────────────────────────────────────────────────────
def _get_api_key() -> str:
    key = os.environ.get("YOUTUBE_API_KEY")
    if not key:
        raise EnvironmentError("YOUTUBE_API_KEY environment variable not set")
    return key


def _get(url: str, params: dict) -> dict:
    """GET url?params → parsed JSON. Raises RuntimeError on any HTTP failure."""
    full = f"{url}?{urlencode(params)}"
    req = Request(full, headers={"Accept": "application/json"})
    try:
        with urlopen(req, timeout=API_TIMEOUT_SECS) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except HTTPError as e:
        body = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else ""
        raise RuntimeError(f"YouTube API HTTP {e.code}: {body[:500]}") from e
    except URLError as e:
        raise RuntimeError(f"YouTube API network error: {e.reason}") from e


_ISO_DURATION_RE = re.compile(r"^PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?$")


def _parse_iso8601_duration(dur: str) -> float:
    """Parse 'PT1H2M30S' → 3750.0 seconds. Returns 0 on malformed input."""
    if not dur:
        return 0.0
    m = _ISO_DURATION_RE.match(dur)
    if not m:
        return 0.0
    h, mi, s = m.groups()
    return int(h or 0) * 3600 + int(mi or 0) * 60 + int(s or 0)


# ── API calls ────────────────────────────────────────────────────────────────
def search_videos(query: str, max_results: int, order: str) -> list[str]:
    """Return a list of video IDs matching the query. 100 quota units."""
    resp = _get(SEARCH_URL, {
        "part": "snippet",
        "q": query,
        "type": "video",
        "maxResults": min(max_results, 50),
        "order": order,
        "key": _get_api_key(),
    })
    return [item["id"]["videoId"] for item in resp.get("items", [])
            if item.get("id", {}).get("videoId")]


def fetch_video_details(video_ids: list[str]) -> list[DiscoveredVideo]:
    """Batch fetch full metadata for up to 50 IDs. 1 quota unit."""
    if not video_ids:
        return []
    resp = _get(VIDEOS_URL, {
        "part": "snippet,contentDetails,statistics,liveStreamingDetails",
        "id": ",".join(video_ids[:50]),
        "key": _get_api_key(),
    })
    results: list[DiscoveredVideo] = []
    for item in resp.get("items", []):
        snip = item.get("snippet", {})
        details = item.get("contentDetails", {})
        stats = item.get("statistics", {})

        # Skip live / upcoming — we want VODs only
        if snip.get("liveBroadcastContent") in ("live", "upcoming"):
            continue

        vid = item["id"]
        results.append(DiscoveredVideo(
            video_id=vid,
            url=f"https://www.youtube.com/watch?v={vid}",
            title=snip.get("title", ""),
            description=(snip.get("description", "") or "")[:500],
            channel=snip.get("channelTitle", ""),
            duration_secs=_parse_iso8601_duration(details.get("duration", "")),
            view_count=int(stats.get("viewCount", 0)),
            published_at=snip.get("publishedAt", ""),
            thumbnail=snip.get("thumbnails", {}).get("high", {}).get("url", ""),
            tags=snip.get("tags", [])[:20],
        ))
    return results


# ── Public API ───────────────────────────────────────────────────────────────
def run_stage1(
    query: str,
    max_results: int = MAX_RESULTS,
    min_duration_secs: float = MIN_DURATION_SECS,
    max_duration_secs: float = MAX_DURATION_SECS,
    order: str = "relevance",
    output_dir: str = "./crowd_output",
) -> list[DiscoveredVideo]:
    """Search YouTube, fetch details, filter by duration, save discovery.json."""
    print(f"[stage1] searching for {query!r} (max {max_results}, order={order})")
    ids = search_videos(query, max_results, order)
    print(f"[stage1] found {len(ids)} video ids, fetching details")

    videos = fetch_video_details(ids)

    filtered = [
        v for v in videos
        if min_duration_secs <= v.duration_secs <= max_duration_secs
    ]
    dropped = len(videos) - len(filtered)
    if dropped:
        print(f"[stage1] dropped {dropped} outside "
              f"[{min_duration_secs:.0f}s, {max_duration_secs:.0f}s]")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    discovery_path = out / "discovery.json"
    discovery_path.write_text(json.dumps({
        "query": query,
        "order": order,
        "video_count": len(filtered),
        "videos": [asdict(v) for v in filtered],
    }, indent=2))

    print(f"[stage1] done — {len(filtered)} videos → {discovery_path.name}")
    return filtered


# ── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Stage 1: YouTube video discovery")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--out", default="./crowd_output")
    parser.add_argument("--max-results", type=int, default=MAX_RESULTS)
    parser.add_argument("--min-duration", type=int, default=MIN_DURATION_SECS)
    parser.add_argument("--max-duration", type=int, default=MAX_DURATION_SECS)
    parser.add_argument("--order", default="relevance",
                        choices=["relevance", "viewCount", "date", "rating"])
    parser.add_argument("--full-pipeline", action="store_true",
                        help="After discovery, run stages 2..8 on the results")
    parser.add_argument("--min-persons", type=int, default=5, help="Stage 3 threshold")
    parser.add_argument("--rpm", type=int, default=15, help="Stage 4 Gemini RPM")
    parser.add_argument("--gap-secs", type=float, default=3.0, help="Stage 6 segment gap")
    parser.add_argument("--export-clips", action="store_true", help="Stage 8 clip export")
    args = parser.parse_args()

    videos = run_stage1(
        args.query,
        max_results=args.max_results,
        min_duration_secs=args.min_duration,
        max_duration_secs=args.max_duration,
        order=args.order,
        output_dir=args.out,
    )
    for v in videos:
        mins = v.duration_secs / 60
        print(f"  {v.video_id}  ({mins:5.1f}m, {v.view_count:>10,} views)  {v.title[:60]}")

    if args.full_pipeline:
        from stage2_download_prepare import run_stage2
        from stage3_prefilter import run_stage3
        from stage4_gemini_score import run_stage4
        from stage6_segment_builder import run_stage6
        from stage8_output import run_stage8

        urls = [v.url for v in videos]
        v_data = run_stage2(urls, output_dir=args.out)
        prefiltered = run_stage3(v_data, person_count_min=args.min_persons)
        scored = run_stage4(prefiltered, rpm=args.rpm)
        segmented = run_stage6(scored, min_segment_secs=args.gap_secs)
        outputs = run_stage8(segmented, output_dir=args.out, export_clips=args.export_clips)

        print("\n[pipeline] artifacts:")
        for k, v in outputs.items():
            print(f"  {k}: {v}")

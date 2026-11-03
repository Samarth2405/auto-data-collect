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

# ── Vantage heuristics (pre-download bias) ──────────────────────────────────
# Cheap first pass over YouTube metadata. Titles lie, so this only biases the
# download pool — Stage 4 asks Gemini per frame and makes the real decision.
ELEVATED_MARKERS = (
    "aerial", "drone", "overhead", "top view", "top angle", "from above",
    "bird's eye", "birds eye", "birdseye", "high angle", "elevated",
    "rooftop", "roof top", "terrace", "balcony", "upper floor", "flyover",
    "skywalk", "footbridge", "viewpoint", "crane shot", "dji", "mavic",
    "looking down", "above the crowd", "panorama", "panoramic",
    "helicopter", "helicam", "sky view", "top shot", "overbridge", "overpass",
    "ariel view", "arial view", "aireal",      # common misspellings of "aerial"
    "ड्रोन",                  # "drone" in Devanagari
)
GROUND_MARKERS = (
    "walking tour", "walk tour", "pov", "point of view", "first person",
    "handheld", "walk with me", "street food", "food tour", "selfie",
    "vlog", "asmr", "eating", "taste test", "shopping guide", "haul",
)


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
    vantage_hint: str = "unclear"               # likely_elevated | likely_ground | unclear
    vantage_markers: list[str] = field(default_factory=list)


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
def search_videos(
    query: str,
    max_results: int,
    order: str,
    published_after: str | None = None,
    region_code: str | None = None,
) -> list[str]:
    """Return a list of video IDs matching the query. 100 quota units.

    published_after is an RFC 3339 timestamp ("2025-10-19T00:00:00Z"); region_code
    is an ISO 3166 alpha-2 code ("IN") that biases results toward that country.
    Both are optional and leave existing callers' behaviour unchanged.
    """
    params = {
        "part": "snippet",
        "q": query,
        "type": "video",
        "maxResults": min(max_results, 50),
        "order": order,
        "key": _get_api_key(),
    }
    if published_after:
        params["publishedAfter"] = published_after
    if region_code:
        params["regionCode"] = region_code
    resp = _get(SEARCH_URL, params)
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


def _vantage_hint(video: DiscoveredVideo) -> tuple[str, list[str]]:
    """Guess camera height from title/description/tags.

    Returns (hint, matched_markers). Elevated markers win ties: a drone video
    captioned "walking tour of Chandni Chowk" is still worth downloading,
    whereas the reverse costs us only one wasted download that Stage 4 catches.
    """
    haystack = " ".join([
        video.title or "",
        video.description or "",
        " ".join(video.tags or []),
    ]).lower()

    up = [m for m in ELEVATED_MARKERS if m in haystack]
    down = [m for m in GROUND_MARKERS if m in haystack]

    if up:
        return "likely_elevated", up
    if down:
        return "likely_ground", down
    return "unclear", []


# ── Public API ───────────────────────────────────────────────────────────────
def run_stage1(
    query: str,
    max_results: int = MAX_RESULTS,
    min_duration_secs: float = MIN_DURATION_SECS,
    max_duration_secs: float = MAX_DURATION_SECS,
    order: str = "relevance",
    output_dir: str = "./crowd_output",
    require_elevated: bool = False,
    drop_ground: bool = False,
) -> list[DiscoveredVideo]:
    """Search YouTube, fetch details, filter, save discovery.json.

    require_elevated keeps only videos whose metadata names an elevated vantage.
    drop_ground is the softer form: keep elevated *and* unclear, drop only the
    ones that advertise themselves as ground-level walking tours and vlogs.
    """
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

    # Annotate every survivor, then optionally filter on the annotation
    for v in filtered:
        v.vantage_hint, v.vantage_markers = _vantage_hint(v)

    if require_elevated or drop_ground:
        keep = {"likely_elevated"} | (set() if require_elevated else {"unclear"})
        before = len(filtered)
        rejected = [v for v in filtered if v.vantage_hint not in keep]
        filtered = [v for v in filtered if v.vantage_hint in keep]
        mode = "require-elevated" if require_elevated else "drop-ground"
        print(f"[stage1] vantage filter ({mode}): kept {len(filtered)}/{before}")
        for v in rejected:
            why = ", ".join(v.vantage_markers) or "no elevated marker"
            print(f"[stage1]   rejected {v.video_id} [{v.vantage_hint}: {why}] {v.title[:48]}")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    discovery_path = out / "discovery.json"
    discovery_path.write_text(json.dumps({
        "query": query,
        "order": order,
        "video_count": len(filtered),
        "vantage_filter": ("require_elevated" if require_elevated
                           else "drop_ground" if drop_ground else "off"),
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
    parser.add_argument("--workers", type=int, default=1,
                        help="Concurrent Gemini calls (paid tier); --rpm still caps the total")
    parser.add_argument("--gap-secs", type=float, default=3.0, help="Stage 6 segment gap")
    parser.add_argument("--export-clips", action="store_true", help="Stage 8 clip export")
    parser.add_argument("--require-elevated", action="store_true",
                        help="Stage 1: keep only videos whose metadata names an elevated vantage")
    parser.add_argument("--drop-ground", action="store_true",
                        help="Stage 1: keep elevated + unclear, drop advertised ground-level footage")
    parser.add_argument("--any-vantage", action="store_true",
                        help="Stage 6/8: keep ground-level segments too (default is elevated only)")
    parser.add_argument("--location", default="",
                        help="Stage 8: location slug for the dataset tree, e.g. mumbai-crawford-market")
    parser.add_argument("--dataset", default="",
                        help="Stage 8: export a Getty-style dataset tree into this directory")
    args = parser.parse_args()

    videos = run_stage1(
        args.query,
        max_results=args.max_results,
        min_duration_secs=args.min_duration,
        max_duration_secs=args.max_duration,
        order=args.order,
        output_dir=args.out,
        require_elevated=args.require_elevated,
        drop_ground=args.drop_ground,
    )
    for v in videos:
        mins = v.duration_secs / 60
        tag = {"likely_elevated": "^", "likely_ground": "_", "unclear": "?"}[v.vantage_hint]
        print(f"  {tag} {v.video_id}  ({mins:5.1f}m, {v.view_count:>10,} views)  {v.title[:56]}")

    if args.full_pipeline:
        from stage2_download_prepare import run_stage2
        from stage3_prefilter import run_stage3
        from stage4_gemini_score import run_stage4
        from stage6_segment_builder import run_stage6
        from stage8_output import run_stage8

        urls = [v.url for v in videos]
        v_data = run_stage2(urls, output_dir=args.out)
        prefiltered = run_stage3(v_data, person_count_min=args.min_persons)
        scored = run_stage4(prefiltered, rpm=args.rpm, workers=args.workers)
        segmented = run_stage6(scored, min_segment_secs=args.gap_secs,
                               require_elevated=not args.any_vantage)
        outputs = run_stage8(segmented, output_dir=args.out,
                             export_clips=args.export_clips,
                             dataset_dir=args.dataset or None,
                             location=args.location or None)

        print("\n[pipeline] artifacts:")
        for k, v in outputs.items():
            print(f"  {k}: {v}")

"""
discover_india.py - India-wide discovery of crowd footage shot from height.

YouTube Data API only: no downloads, no Gemini. Writes a ranked candidate list
to candidates/ for the pipeline to consume later.

Quota: each query costs 100 (search) + 1 (details). Results are cached per query
under candidates/raw/, so a rerun or a crash never re-spends quota, and the
ranking can be re-tuned for free with --rank-only.

    python discover_india.py --dry-run      # show the plan and quota cost
    python discover_india.py                # search, cache, rank
    python discover_india.py --rank-only    # re-rank from cache, zero quota
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from stage1_search import (
    GROUND_MARKERS,
    DiscoveredVideo,
    _vantage_hint,
    fetch_video_details,
    search_videos,
)

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

OUT_DIR = Path("candidates")
RAW_DIR = OUT_DIR / "raw"
MAX_PER_QUERY = 50          # same quota as 10, so take the maximum
MIN_DURATION_SECS = 20      # aerial crowd shots are often short; Stage 1's 30s floor drops them
MAX_DURATION_SECS = 1800    # every extra minute is 60 Gemini calls at 1 fps
RECENT_DAYS = 365           # applied to protest / rally queries only
REGION = "IN"

# A title can say "drone" and still be empty streets or rooftops, which cost us
# a whole location already: "Scenic Drone Shoot ... Lockdown" scored 0 high-density
# frames across 244 aerial ones. So height alone is never enough; the title must
# also point at people.
CROWD_MARKERS = (
    "crowd", "crowded", "rally", "protest", "procession", "gathering", "jan sabha",
    "jansabha", "yatra", "mela", "visarjan", "immersion", "stampede", "sea of people",
    "lakh", "massive", "huge", "march", "dharna", "aarti", "pooram", "queue", "rush",
    "bheed", "devotees", "pilgrims", "namaz", "celebration", "victory parade",
    "भीड़",              # bheed
    "रैली",              # rally
    "आंदोलन",   # andolan
    "प्रदर्शन",  # pradarshan
    "जनसभा",         # jansabha
    "यात्रा",   # yatra
    "मेला",               # mela
    "विसर्जन",  # visarjan
)
NON_CROWD_MARKERS = (
    "lockdown", "empty", "deserted", "scenic", "skyline", "cityscape", "housing",
    "real estate", "property", "hotel", "resort", "timelapse", "time lapse",
    "hyperlapse", "sunrise", "sunset", "flood", "construction", "plot", "villa",
)

# (location slug, category, query, recent_only)
Q: list[tuple[str, str, str, bool]] = []


def add(category: str, location: str, *queries: str, recent: bool = False) -> None:
    for q in queries:
        Q.append((location, category, q, recent))


# Protests and rallies: recent only, per the brief.
add("protest", "india-protest",
    "India protest drone view huge crowd",
    "India protest march aerial footage",
    "आंदोलन ड्रोन वीडियो भीड़",
    "candlelight march drone view India", recent=True)
add("protest", "india-farmers-protest", "farmers protest drone view crowd India", recent=True)
add("protest", "india-student-protest", "students protest march drone footage India", recent=True)
add("protest", "delhi-ramlila-maidan", "Ramlila Maidan rally drone view", recent=True)
add("protest", "delhi-jantar-mantar", "Jantar Mantar protest crowd drone", recent=True)
add("protest", "mumbai-azad-maidan", "Azad Maidan Mumbai protest crowd aerial", recent=True)
add("protest", "hyderabad-indira-park", "Hyderabad protest rally drone view crowd", recent=True)
add("protest", "bengaluru-freedom-park", "Bengaluru protest march drone view crowd", recent=True)
add("protest", "chennai-marina", "Chennai Marina protest crowd aerial drone", recent=True)
add("rally", "kolkata-brigade-parade-ground", "Brigade Parade Ground rally drone view Kolkata", recent=True)
add("rally", "patna-gandhi-maidan", "Gandhi Maidan Patna rally drone view crowd", recent=True)
add("rally", "lucknow-rally", "Lucknow rally drone view crowd", recent=True)
add("rally", "india-election-rally", "election rally drone view huge crowd India", recent=True)
add("rally", "india-roadshow", "roadshow drone view massive crowd India", recent=True)
add("rally", "india-jan-sabha", "jan sabha drone view crowd", recent=True)

# Festivals and processions.
add("festival", "prayagraj-kumbh-mela", "Maha Kumbh drone view crowd Sangam", "Kumbh Mela aerial crowd drone")
add("festival", "mumbai-ganesh-visarjan", "Ganesh visarjan Mumbai drone view crowd Girgaum",
    "Lalbaugcha Raja crowd drone view")
add("festival", "kolkata-durga-puja", "Durga Puja Kolkata crowd drone view pandal")
add("festival", "puri-rath-yatra", "Rath Yatra Puri drone view crowd")
add("festival", "mathura-holi", "Holi Vrindavan Mathura crowd drone view")
add("festival", "india-kanwar-yatra", "Kanwar Yatra drone view crowd")
add("festival", "patna-chhath-puja", "Chhath Puja ghat crowd drone view")
add("festival", "ahmedabad-navratri", "Navratri garba crowd drone view Ahmedabad")
add("festival", "delhi-jama-masjid", "Jama Masjid Eid namaz crowd drone view Delhi")
add("festival", "madurai-jallikattu", "Jallikattu Alanganallur crowd drone view")
add("festival", "thrissur-pooram", "Thrissur Pooram crowd drone view")
add("festival", "mysuru-dasara", "Mysuru Dasara Jamboo Savari crowd drone view")
add("festival", "guwahati-ambubachi", "Ambubachi Mela Kamakhya crowd drone view")
add("festival", "gangasagar-mela", "Gangasagar Mela crowd drone view")
add("festival", "hyderabad-ganesh-immersion", "Hussain Sagar Ganesh immersion crowd drone view Hyderabad")
add("festival", "india-diwali-market", "Diwali shopping crowd market drone view India")

# Religious gatherings.
add("religious", "tirupati-tirumala", "Tirumala Tirupati crowd drone view queue")
add("religious", "amritsar-golden-temple", "Golden Temple Amritsar crowd drone view")
add("religious", "haridwar-har-ki-pauri", "Haridwar Har Ki Pauri Ganga Aarti crowd drone view")
add("religious", "varanasi-ganga-aarti", "Varanasi Ganga Aarti crowd drone view")
add("religious", "ayodhya-ram-mandir", "Ayodhya Ram Mandir crowd drone view")
add("religious", "ujjain-mahakal", "Ujjain Mahakal crowd drone view")
add("religious", "ajmer-sharif", "Ajmer Sharif Urs crowd drone view")
add("religious", "sabarimala", "Sabarimala crowd drone view")
add("religious", "shirdi", "Shirdi Sai Baba crowd drone view")

# Transit hubs.
add("transit", "mumbai-local-train", "Mumbai local train crowd from above bridge")
add("transit", "kolkata-howrah-station", "Howrah station crowd drone view")
add("transit", "delhi-railway-station", "New Delhi railway station crowd overhead")
add("transit", "india-railway-crowd", "railway station crowd India foot overbridge view")

# Sport and celebration.
add("celebration", "mumbai-marine-drive", "Marine Drive Mumbai victory parade crowd drone view")
add("celebration", "bengaluru-chinnaswamy", "Bengaluru Chinnaswamy celebration crowd drone view")
add("celebration", "ahmedabad-stadium", "Narendra Modi Stadium crowd aerial")
add("celebration", "india-concert", "concert crowd India drone view")

# Markets, streets, beaches, and generic crowds.
add("market", "hyderabad-charminar", "Charminar Ramzan night crowd drone view Hyderabad")
add("market", "jaipur-johari-bazaar", "Jaipur bazaar crowd drone view")
add("market", "india-crowded-market", "crowded market India drone view")
add("beach", "mumbai-juhu-chowpatty", "Juhu Chowpatty beach crowd drone view")
add("crowd", "india-crowd", "huge crowd India drone view", "sea of people India aerial footage")


# ── Search + cache ───────────────────────────────────────────────────────────
def _cache_path(query: str) -> Path:
    return RAW_DIR / f"{hashlib.md5(query.encode('utf-8')).hexdigest()[:12]}.json"


def _search_one(location: str, category: str, query: str, recent: bool) -> list[dict] | None:
    """Return cached or fresh videos for a query. None means quota is gone."""
    path = _cache_path(query)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))["videos"]

    after = None
    if recent:
        after = (datetime.now(timezone.utc) - timedelta(days=RECENT_DAYS)).strftime(
            "%Y-%m-%dT%H:%M:%SZ")
    try:
        ids = search_videos(query, MAX_PER_QUERY, "relevance",
                            published_after=after, region_code=REGION)
        vids = fetch_video_details(ids)
    except RuntimeError as exc:
        if "quotaExceeded" in str(exc) or "HTTP 403" in str(exc):
            print(f"  !! quota exhausted at {query!r} - stopping, partial results kept")
            return None
        print(f"  !! skipped {query!r}: {str(exc)[:90]}")
        return []

    kept = [asdict(v) for v in vids
            if MIN_DURATION_SECS <= v.duration_secs <= MAX_DURATION_SECS]
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "query": query, "location": location, "category": category, "recent": recent,
        "fetched_at": datetime.now(timezone.utc).isoformat(), "videos": kept,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    return kept


# ── Ranking ──────────────────────────────────────────────────────────────────
def _hits(haystack: str, markers: tuple[str, ...]) -> list[str]:
    return [m for m in markers if m in haystack]


def classify(v: dict) -> dict:
    """Tier a video from its metadata. A needs height AND people; C is excluded."""
    dv = DiscoveredVideo(**{k: v[k] for k in DiscoveredVideo.__dataclass_fields__
                            if k in v and k not in ("vantage_hint", "vantage_markers")})
    hint, elev = _vantage_hint(dv)
    hay = " ".join([v["title"], v.get("description", ""), " ".join(v.get("tags", []))]).lower()
    crowd = _hits(hay, CROWD_MARKERS)
    non_crowd = _hits(v["title"].lower(), NON_CROWD_MARKERS)   # title only: descriptions are noisy
    ground = hint == "likely_ground"

    if non_crowd or ground:
        tier = "C"
    elif hint == "likely_elevated" and crowd:
        tier = "A"
    elif hint == "likely_elevated" or crowd:
        tier = "B"
    else:
        tier = "C"

    score = (3 * (hint == "likely_elevated") + min(len(elev), 3)
             + 2 * bool(crowd) + min(len(crowd), 3)
             - 4 * bool(non_crowd) - 3 * ground
             + (1 if v["duration_secs"] <= 600 else 0))
    return {"tier": tier, "score": score, "hint": hint, "elevated_markers": elev,
            "crowd_markers": crowd, "non_crowd_markers": non_crowd}


def collect() -> tuple[list[dict], Counter]:
    """Merge every cached query into one deduplicated, classified list."""
    merged: dict[str, dict] = {}
    per_query: Counter = Counter()
    for f in sorted(RAW_DIR.glob("*.json")):
        raw = json.loads(f.read_text(encoding="utf-8"))
        per_query[(raw["category"], raw["location"])] += len(raw["videos"])
        for v in raw["videos"]:
            entry = merged.setdefault(v["video_id"], {
                **v, "locations": [], "categories": [], "queries": []})
            entry["locations"].append(raw["location"])
            entry["categories"].append(raw["category"])
            entry["queries"].append(raw["query"])

    out = []
    for v in merged.values():
        # Prefer a specific place over a generic "india-*" bucket
        specific = [l for l in v["locations"] if not l.startswith("india-")]
        v["location"] = (specific or v["locations"])[0]
        v["category"] = v["categories"][v["locations"].index(v["location"])]
        v.update(classify(v))
        out.append(v)
    out.sort(key=lambda v: (v["tier"], -v["score"], -v.get("view_count", 0)))
    return out, per_query


# ── Output ───────────────────────────────────────────────────────────────────
CSV_FIELDS = ["tier", "score", "video_id", "url", "location", "category", "duration_secs",
              "view_count", "published_at", "channel", "title", "vantage_hint",
              "elevated_markers", "crowd_markers", "queries"]


def write_outputs(videos: list[dict]) -> None:
    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "india_candidates.json").write_text(
        json.dumps(videos, indent=2, ensure_ascii=False), encoding="utf-8")
    with open(OUT_DIR / "india_candidates.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for v in videos:
            row = {k: v.get(k, "") for k in CSV_FIELDS}
            for k in ("elevated_markers", "crowd_markers", "queries"):
                row[k] = "; ".join(v.get(k, []))
            w.writerow(row)


def report(videos: list[dict]) -> None:
    tiers = Counter(v["tier"] for v in videos)
    print(f"\n{len(videos)} unique videos | tier A={tiers['A']}  B={tiers['B']}  C={tiers['C']} "
          "(C = ground-level or non-crowd, excluded)")
    for t in ("A", "B"):
        sel = [v for v in videos if v["tier"] == t]
        secs = sum(v["duration_secs"] for v in sel)
        print(f"  tier {t}: {len(sel):>3} videos, {secs / 60:6.0f} min of footage "
              f"= {secs:6.0f} Gemini calls = {secs / 15 / 60:5.1f} h at 15 RPM (1 fps)")

    by_cat = defaultdict(list)
    for v in videos:
        if v["tier"] == "A":
            by_cat[v["category"]].append(v)
    print("\nTier A by category:")
    for cat, vs in sorted(by_cat.items(), key=lambda kv: -len(kv[1])):
        print(f"  {cat:12} {len(vs):>3} videos across {len({v['location'] for v in vs})} locations")

    print("\nTop tier-A candidates:")
    for v in [v for v in videos if v["tier"] == "A"][:40]:
        m, s = divmod(int(v["duration_secs"]), 60)
        print(f"  {v['score']:>2} {v['category']:11} {v['location'][:26]:26} {m:>3}:{s:02d} "
              f"{v['published_at'][:7]} {v['title'][:46]}")
        print(f"       {v['url']}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dry-run", action="store_true", help="show the plan and quota cost only")
    ap.add_argument("--rank-only", action="store_true", help="re-rank from cache, zero quota")
    ap.add_argument("--max-queries", type=int, default=len(Q))
    args = ap.parse_args()

    todo = [q for q in Q[:args.max_queries] if not _cache_path(q[2]).exists()]
    cost = len(todo) * 101
    print(f"[discover] {len(Q)} queries planned, {len(Q) - len(todo)} cached, "
          f"{len(todo)} to fetch = {cost:,} quota units (daily budget 10,000)")
    if args.dry_run:
        return 0

    if not args.rank_only:
        for i, (loc, cat, query, recent) in enumerate(Q[:args.max_queries], 1):
            was_cached = _cache_path(query).exists()
            res = _search_one(loc, cat, query, recent)
            if res is None:
                break
            tag = "cached" if was_cached else "fetched"
            print(f"  [{i:>2}/{min(len(Q), args.max_queries)}] {tag:7} {len(res):>2} kept  "
                  f"{cat:11} {query[:56]}")
            if not was_cached:
                time.sleep(0.25)

    videos, _ = collect()
    if not videos:
        print("[discover] nothing cached yet")
        return 1
    write_outputs(videos)
    report(videos)
    print(f"\n[discover] wrote {OUT_DIR / 'india_candidates.json'} and .csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())

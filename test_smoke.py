"""
test_smoke.py — pure-function smoke tests for every stage.
No network, no API keys, no file writes. Verifies data-shape correctness
and the wiring of pure helpers used across the pipeline.

Run:  python test_smoke.py
"""

from __future__ import annotations

from pathlib import Path

# ─── Stage 1 ─────────────────────────────────────────────────────────────────
def test_iso_duration():
    from stage1_search import _parse_iso8601_duration
    cases = [
        ("PT30S", 30.0),
        ("PT2M", 120.0),
        ("PT1H", 3600.0),
        ("PT1H2M30S", 3750.0),
        ("PT15M42S", 942.0),
        ("PT0S", 0.0),
        ("", 0.0),
        ("invalid", 0.0),
        (None, 0.0),
    ]
    for dur, expected in cases:
        got = _parse_iso8601_duration(dur)
        assert got == expected, f"_parse_iso8601_duration({dur!r}) = {got}, expected {expected}"
    print(f"  stage1  iso8601 duration : {len(cases)} cases OK")


# ─── Stage 2 ─────────────────────────────────────────────────────────────────
def test_video_id_regex():
    from stage2_download_prepare import _extract_video_id
    cases = [
        ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://youtu.be/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://youtube.com/embed/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://youtube.com/shorts/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://youtube.com/v/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=42s", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/watch?v=dQw4w9WgXcQ&list=PL_xxx", "dQw4w9WgXcQ"),
        ("not-a-youtube-url", None),
        ("https://vimeo.com/12345", None),
    ]
    for url, expected in cases:
        got = _extract_video_id(url)
        assert got == expected, f"_extract_video_id({url!r}) = {got!r}, expected {expected!r}"
    print(f"  stage2  video_id regex   : {len(cases)} cases OK")


# ─── Stage 6 ─────────────────────────────────────────────────────────────────
def test_fmt_time():
    from stage6_segment_builder import _fmt_time
    cases = [
        (0, "0:00"),
        (5, "0:05"),
        (45, "0:45"),
        (60, "1:00"),
        (112, "1:52"),
        (3599, "59:59"),
        (3600, "1:00:00"),
        (3665, "1:01:05"),
        (7325, "2:02:05"),
    ]
    for secs, expected in cases:
        got = _fmt_time(secs)
        assert got == expected, f"_fmt_time({secs}) = {got!r}, expected {expected!r}"
    print(f"  stage6  time formatting  : {len(cases)} cases OK")


def test_yt_link():
    from stage6_segment_builder import _yt_link
    cases = [
        ("https://www.youtube.com/watch?v=X", 45.0, "https://www.youtube.com/watch?v=X&t=45s"),
        ("https://youtu.be/X", 12.0, "https://youtu.be/X?t=12s"),
        ("https://youtu.be/X#frag", 30.0, "https://youtu.be/X?t=30s"),  # strips fragment
        ("", 30.0, ""),
    ]
    for url, ts, expected in cases:
        got = _yt_link(url, ts)
        assert got == expected, f"_yt_link({url!r}, {ts}) = {got!r}, expected {expected!r}"
    print(f"  stage6  yt_link deep-link: {len(cases)} cases OK")


def test_group_by_gap():
    from stage3_prefilter import FrameScore
    from stage4_gemini_score import GeminiScore
    from stage6_segment_builder import _group_by_gap

    def at(ts):
        return GeminiScore(
            frame=FrameScore(path=Path("."), timestamp=ts, person_count=10, passes_filter=True),
            crowd_score=8,
            reasoning="",
        )

    # gap=3.0 ->[10,11,12] | [20,21] | [40]
    hits = [at(t) for t in (10, 11, 12, 20, 21, 40)]
    groups = _group_by_gap(hits, 3.0)
    assert [len(g) for g in groups] == [3, 2, 1], f"gap=3: {[len(g) for g in groups]}"

    # gap=10.0 ->[10,11,12,20,21] | [40]  (21→40 gap of 19 > 10)
    groups = _group_by_gap(hits, 10.0)
    assert [len(g) for g in groups] == [5, 1], f"gap=10: {[len(g) for g in groups]}"

    # gap=100.0 ->all one group
    groups = _group_by_gap(hits, 100.0)
    assert len(groups) == 1 and len(groups[0]) == 6

    # empty input
    assert _group_by_gap([], 3.0) == []

    # boundary: gap exactly == max_gap should be inclusive (same group)
    hits2 = [at(0), at(3), at(6.001)]
    groups = _group_by_gap(hits2, 3.0)
    assert [len(g) for g in groups] == [2, 1], f"boundary: {[len(g) for g in groups]}"

    print("  stage6  group_by_gap     : 5 scenarios OK")


def test_scene_cuts():
    from stage6_segment_builder import _pick_cuts, _scene_candidates, _shot_boundaries

    fps = 25
    series = [(i / fps, 0.02) for i in range(250)]          # steady footage
    series[50] = (2.0, 0.6)                                   # hard cut: one spike
    for i in range(100, 125):                                 # fast pan: sustained
        series[i] = (i / fps, 0.25)
    for i in range(150, 180):                                 # 60 fps repeats: x, 0, x, 0
        series[i] = (i / fps, 0.22 if i % 2 else 0.0)
    series[200] = (8.0, 0.5)                                  # two cuts around a still
    for i in range(201, 215):                                 # title card (repeats only)
        series[i] = (i / fps, 0.0)
    series[215] = (8.6, 0.45)
    cands = _scene_candidates(series, window=2, floor=0.1, dup_floor=0.01)
    assert _pick_cuts(cands) == [2.0, 8.0, 8.6], f"cuts: {_pick_cuts(cands)}"

    # thresholds re-derive from the cached candidates; black edges are boundaries
    assert _pick_cuts(cands, min_score=0.55) == [2.0]
    assert _shot_boundaries({"candidates": cands, "black": [[9.0, 9.5]]}) == [2.0, 8.0, 8.6, 9.0, 9.5]
    print("  stage6  scene cuts       : spike vs pan vs repeats vs still OK")


def test_shot_aware_segments():
    from stage3_prefilter import FrameScore
    from stage4_gemini_score import GeminiScore
    from stage6_segment_builder import _aggregate, _group_by_gap

    def at(ts, score=8):
        return GeminiScore(
            frame=FrameScore(path=Path("."), timestamp=ts, person_count=0, passes_filter=True),
            crowd_score=score, reasoning="", vantage="aerial",
        )

    hits = [at(10), at(11, 9), at(12), at(13), at(20)]
    # no cuts: the old behaviour, one run [10..13] then [20]
    assert [len(g) for g in _group_by_gap(hits, 3.0)] == [4, 1]

    # cut at 11.2: frame 11 is sampled at 11.5, so it belongs to the next shot
    groups = _group_by_gap(hits, 3.0, [11.2], fps=1.0)
    assert [len(g) for g in groups] == [1, 3, 1], [len(g) for g in groups]
    segs = [_aggregate(g, "https://youtu.be/x", 1.0, [11.2]) for g in groups]
    # first ends at the cut, not at 11; second starts at the cut, not at 11
    assert (segs[0].start, segs[0].end) == (10.0, 11.0)
    assert (segs[1].start, segs[1].end) == (11.2, 14.0), (segs[1].start, segs[1].end)
    assert segs[1].peak_timestamp == 11, "poster comes from the best hit"

    # cut at 12.7 lands inside frame 12's window: the segment stops there
    seg = _aggregate(_group_by_gap(hits[:3], 3.0, [12.7])[0], "", 1.0, [12.7])
    assert (seg.start, seg.end) == (10.0, 12.7), (seg.start, seg.end)
    print("  stage6  shot-aware split : split at cuts, trim to shot, peak frame OK")


def test_dataset_reexport():
    import csv
    import tempfile
    from stage2_download_prepare import VideoData
    from stage3_prefilter import PrefilteredVideo
    from stage4_gemini_score import ScoredVideo
    from stage6_segment_builder import Segment, VideoSegments
    from stage8_output import export_dataset

    def vs(video_id, n_segs):
        video = VideoData(video_id=video_id, url=f"https://youtu.be/{video_id}", title="t",
                          duration=60, video_path=Path("missing.mp4"), frames_dir=Path("f"))
        segs = [Segment(start=i * 10.0, end=i * 10.0 + 3, start_fmt="", end_fmt="",
                        frame_count=3, avg_crowd_score=8, peak_crowd_score=8,
                        person_count_avg=0, person_count_peak=0, yt_link="",
                        vantage="aerial") for i in range(n_segs)]
        return VideoSegments(scored=ScoredVideo(prefiltered=PrefilteredVideo(video=video)),
                             segments=segs)

    def manifest(root):
        with open(root / "manifest.csv", newline="", encoding="utf-8") as f:
            return [(r["asset_id"], r["location"], r["source_video_id"]) for r in csv.DictReader(f)]

    import contextlib, io
    with tempfile.TemporaryDirectory() as td, contextlib.redirect_stdout(io.StringIO()):
        root = Path(td)
        export_dataset([vs("A", 3)], td, "puri-rath-yatra")
        export_dataset([vs("B", 2)], td, "puri-rath-yatra")        # second query, same place
        export_dataset([vs("C", 1)], td, "kolkata-rath-yatra")     # different place, same code RAT
        ids = [r[0] for r in manifest(root)]
        assert ids == [f"IND_RAT_{i:04d}" for i in range(1, 7)], ids

        export_dataset([vs("A", 1)], td, "puri-rath-yatra")        # re-segmented: 3 -> 1
        rows = manifest(root)
        assert sorted(r[2] for r in rows) == ["A", "B", "B", "C"], rows
        assert len({r[0] for r in rows}) == len(rows), "IDs stay unique"
        gone = root / "puri-rath-yatra" / "aerial" / "IND_RAT_0002"
        assert not gone.exists(), "superseded asset folder removed"
        assert (root / "kolkata-rath-yatra" / "aerial" / "IND_RAT_0006").is_dir()
    print("  stage8  dataset re-export: no ID reuse, re-export replaces, stale dirs removed OK")


def _segment(start, end, vantage="aerial"):
    from stage6_segment_builder import Segment
    return Segment(start=start, end=end, start_fmt="", end_fmt="", frame_count=0,
                   avg_crowd_score=8, peak_crowd_score=8, person_count_avg=0,
                   person_count_peak=0, yt_link="", vantage=vantage)


def test_crowd_count_selection():
    from stage2_download_prepare import VideoData
    from stage7_crowd_count import _aggregate, _frames_by_segment

    frames = [Path(f"f/{k:06d}.jpg") for k in range(1, 21)]        # stamped t = k - 1
    video = VideoData(video_id="v", url="", title="", duration=20, video_path=Path("v.mp4"),
                      frames_dir=Path("f"), frame_paths=frames, metadata={"fps_extracted": 1.0})
    # [10, 13) holds frames stamped 10, 11, 12; [11.2, 14) was trimmed at a cut and
    # holds 11, 12, 13 (sampled at t + 0.5, so 11 is inside, 10 is not)
    sel = _frames_by_segment(video, [_segment(10, 13), _segment(11.2, 14)])
    names = [[p.name for p in seg] for seg in sel]
    assert names == [["000011.jpg", "000012.jpg", "000013.jpg"],
                     ["000012.jpg", "000013.jpg", "000014.jpg"]], names

    counts = {"000011.jpg": 100.4, "000012.jpg": 200.6, "000013.jpg": 299.4}
    c = _aggregate(sel[0], counts)
    assert (c.avg, c.peak, c.frames) == (200, 299, 3), c
    assert _aggregate([Path("f/000099.jpg")], counts) is None, "no cached count -> None"
    print("  stage7  count selection  : frames per segment, avg/peak OK")


def test_counts_in_dataset():
    import contextlib, csv, io, json, tempfile
    from stage2_download_prepare import VideoData
    from stage3_prefilter import PrefilteredVideo
    from stage4_gemini_score import ScoredVideo
    from stage6_segment_builder import VideoSegments
    from stage7_crowd_count import CountedVideo, SegmentCount
    from stage8_output import export_dataset

    video = VideoData(video_id="A", url="https://youtu.be/A", title="t", duration=60,
                      video_path=Path("missing.mp4"), frames_dir=Path("f"))
    vs = VideoSegments(scored=ScoredVideo(prefiltered=PrefilteredVideo(video=video)),
                       segments=[_segment(0, 3), _segment(10, 13)])
    cv = CountedVideo(segmented=vs, segment_counts=[SegmentCount(812, 950, 3), None])
    with tempfile.TemporaryDirectory() as td, contextlib.redirect_stdout(io.StringIO()):
        export_dataset([cv], td, "puri-rath-yatra")
        with open(Path(td) / "manifest.csv", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert [(r["crowd_count_avg"], r["crowd_count_peak"]) for r in rows] == \
            [("812", "950"), ("", "")], rows
        meta = json.loads((Path(td) / "puri-rath-yatra/aerial/IND_RAT_0001/metadata.json")
                          .read_text(encoding="utf-8"))["measurements"]
        assert meta["crowd_count_peak"] == 950 and "DM-Count" in meta["crowd_count_method"]
        assert "YOLO" in meta["person_count_method"]
        # a plain VideoSegments (counting skipped) still exports, with blank counts
        export_dataset([vs], td, "puri-rath-yatra")
    print("  stage8  crowd counts     : manifest + metadata fields, blank when skipped OK")


# ─── Data contract chain ─────────────────────────────────────────────────────
def test_composition_chain():
    """Verify VideoSegments.video walks scored ->prefiltered ->video shortcut."""
    from stage2_download_prepare import VideoData
    from stage3_prefilter import PrefilteredVideo
    from stage4_gemini_score import ScoredVideo
    from stage6_segment_builder import VideoSegments

    video = VideoData(
        video_id="test123", url="https://x", title="T", duration=100,
        video_path=Path("v.mp4"), frames_dir=Path("f"),
    )
    pv = PrefilteredVideo(video=video)
    sv = ScoredVideo(prefiltered=pv)
    vs = VideoSegments(scored=sv)

    assert vs.video is video, "VideoSegments.video should equal the underlying VideoData"
    assert vs.scored.prefiltered.video is video, "manual walk should agree"

    # high_density_frames property should default to []
    assert sv.high_density_frames == []
    assert pv.candidate_frames == []
    print("  chain   composition       : VideoSegments ->VideoData walk OK")


# ─── Stage 4 sentinel behavior ───────────────────────────────────────────────
def test_score_clamp_and_sentinel():
    """Ensure GeminiScore.is_high_density uses CROWD_THRESHOLD correctly."""
    from stage3_prefilter import FrameScore
    from stage4_gemini_score import CROWD_THRESHOLD, GeminiScore

    frame = FrameScore(path=Path("."), timestamp=0, person_count=0, passes_filter=True)
    below = GeminiScore(frame=frame, crowd_score=CROWD_THRESHOLD - 1, reasoning="")
    at = GeminiScore(frame=frame, crowd_score=CROWD_THRESHOLD, reasoning="")
    above = GeminiScore(frame=frame, crowd_score=CROWD_THRESHOLD + 2, reasoning="")

    assert not below.is_high_density
    assert at.is_high_density
    assert above.is_high_density
    print(f"  stage4  threshold cutoff : boundary at {CROWD_THRESHOLD} OK")


# ─── Stage 4 failure classification ──────────────────────────────────────────
def test_terminal_error_classification():
    """A spend cap must not be retried; a per-minute 429 must be."""
    from stage4_gemini_score import _is_terminal

    terminal = [
        "429 RESOURCE_EXHAUSTED ... exceeded its monthly spending cap ...",
        "400 API key not valid. Please pass a valid API key.",
        "429 Quota exceeded for metric generate_content_requests per day",
        "403 PERMISSION_DENIED",
    ]
    transient = [
        "429 RESOURCE_EXHAUSTED ... quota metric ... per minute ...",
        "503 Service Unavailable",
        "500 Internal error",
        "read operation timed out",
    ]
    for msg in terminal:
        assert _is_terminal(Exception(msg)), f"should be terminal: {msg!r}"
    for msg in transient:
        assert not _is_terminal(Exception(msg)), f"should be retryable: {msg!r}"
    print(f"  stage4  terminal vs retry: {len(terminal)}+{len(transient)} cases OK")


# ─── Stage 4 concurrency ─────────────────────────────────────────────────────
def test_concurrent_scoring():
    """Workers must not change results; quota/errors must bank partial work."""
    import json
    import tempfile
    import time

    import stage4_gemini_score as s4
    from stage2_download_prepare import VideoData
    from stage3_prefilter import FrameScore, PrefilteredVideo

    def make_pv(root: Path) -> PrefilteredVideo:
        frames_dir = root / "vid" / "frames"
        frames_dir.mkdir(parents=True)
        video = VideoData(video_id="vid", url="https://x", title="T", duration=12,
                          video_path=root / "vid" / "video.mp4", frames_dir=frames_dir)
        frames = [FrameScore(path=frames_dir / f"{i:06d}.jpg", timestamp=float(i),
                             person_count=0, passes_filter=True) for i in range(1, 13)]
        return PrefilteredVideo(video=video, frame_scores=frames)

    calls: list[str] = []

    def fake(path, rpm):
        calls.append(path.stem)
        time.sleep(0.01)
        if path.stem == "000005" and mode["fail"]:
            raise mode["fail"]("boom")
        return s4.CrowdScoreResponse(score=int(path.stem) % 10 + 1,
                                     vantage=s4.CameraVantage.ELEVATED, reasoning="")

    real_score, real_client = s4._score_frame, s4._get_client
    s4._score_frame, s4._get_client = fake, lambda: None
    mode = {"fail": None}
    try:
        outs = {}
        for w in (1, 4):
            with tempfile.TemporaryDirectory() as d:
                sv = s4.score_video(make_pv(Path(d)), rpm=6000, workers=w)
                outs[w] = [(s.frame.timestamp, s.crowd_score) for s in sv.gemini_scores]
        assert outs[1] == outs[4], "workers changed the scores"
        assert [t for t, _ in outs[4]] == sorted(t for t, _ in outs[4]), "not timestamp-ordered"
        assert len(outs[4]) == 12

        for exc_type, expect_none in ((s4.QuotaExhausted, False), (ValueError, True)):
            mode["fail"] = exc_type
            with tempfile.TemporaryDirectory() as d:
                pv = make_pv(Path(d))
                calls.clear()
                raised = False
                try:
                    result = s4.score_video(pv, rpm=6000, workers=2, save_every=1)
                except s4.QuotaExhausted:
                    result, raised = None, True
                assert raised != expect_none, f"{exc_type.__name__}: raised={raised}"
                assert result is None
                cache = json.loads((Path(d) / "vid" / "gemini_scores.json").read_text())
                banked = len(cache["scores"])
                # Frames 1-4 finish before frame 5 fails; a frame already in
                # flight beside it may bank too, so only the bounds are fixed.
                assert 4 <= banked < 12, f"banked {banked}"

                # Resume: only the unbanked frames are scored again.
                mode["fail"] = None
                calls.clear()
                sv = s4.score_video(pv, rpm=6000, workers=4)
                assert len(calls) == 12 - banked and len(sv.gemini_scores) == 12
    finally:
        s4._score_frame, s4._get_client = real_score, real_client
    print("  stage4  concurrency      : workers=1/4 parity, quota+error banking, resume OK")


# ─── Runner ──────────────────────────────────────────────────────────────────
def main() -> int:
    tests = [
        test_iso_duration,
        test_video_id_regex,
        test_fmt_time,
        test_yt_link,
        test_group_by_gap,
        test_scene_cuts,
        test_shot_aware_segments,
        test_dataset_reexport,
        test_crowd_count_selection,
        test_counts_in_dataset,
        test_composition_chain,
        test_score_clamp_and_sentinel,
        test_terminal_error_classification,
        test_concurrent_scoring,
    ]
    print(f"Running {len(tests)} smoke tests...\n")
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
            failed += 1
        except ImportError as e:
            print(f"  SKIP  {t.__name__}: missing dependency ({e.name})")
    print()
    if failed:
        print(f"{failed} test(s) failed")
        return 1
    print("All smoke tests passed.")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

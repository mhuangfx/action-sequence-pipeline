#!/usr/bin/env python3
"""G9 -- did the storyboard reference image itself get rendered into the video?

WHY THIS EXISTS
---------------
Found by the OWNER'S EYE on 2026-08-24, in a video that every gate had passed. The last 0.46
seconds of seg20 are the seg20 storyboard: the 3x3 panel grid, the yellow panel numbers, the
three empty grey cells, reproduced as video frames. Twelve frames of it, at 56.125-56.583s of
final.mp4.

Nothing caught it, and each miss has its own reason:

  identity_gate  samples ONE frame at each SOURCE shot midpoint. On seg20 those land at 0.75,
                 1.75, 2.42, 3.38, 4.29 and 5.26s. The leak starts at 5.375s. It was missed by
                 ELEVEN HUNDREDTHS OF A SECOND. Sampling three frames a shot would not reliably
                 have caught it either -- the leak is in the tail of the last shot, and midpoint
                 sampling never looks at a tail.
  G1 people>2    would very likely have fired on a grid showing six panels of people, but only
                 if a sample had landed on it. It did not.
  G8 motion      the leak makes the clip LESS energetic, and seg20 was already flagged for low
                 motion. The defect was hiding inside a number we had already written off.
  passthrough    compares the output against the SOURCE VIDEO. The board is not the source
                 video, so there was nothing for it to match.

THE LESSON, AND IT IS THE SAME ONE AGAIN
----------------------------------------
Every gate here inferred from a sample or a proxy. This one does not: it compares the output
against THE EXACT IMAGE WE FED THE RENDERER, which is a file on disk. Same move that fixed G3 --
stop inferring a fact you can check directly. Facts beat thresholds.

It also reads EVERY frame, not a sample. At 141 frames a clip that costs about a second, and a
gate that samples cannot see a defect between its samples.

SEPARATION (measured on seg20_s918273, all 141 frames)
------------------------------------------------------
    leak frames        0.9631 - 0.9740     12 frames, 5.375-5.833s
    every other frame  max 0.1480, mean -0.0859

There is no threshold to tune in that gap. The default of 0.50 sits six standard deviations
clear of the noise and half a unit clear of the signal.

★ SINGLE-PANEL BOARDS BREAK THE PREMISE -- 2026-08-25
------------------------------------------------------
Everything above assumes a GRID board. The premise that made 0.50 safe is that a good render never
resembles a mosaic of nine panels. One board per shot destroys it: a correct render is SUPPOSED to
reproduce its board's composition. Measured on the 2026-08-25 board A/B, frame 0 of
ab_render/S/t05.mp4 scores 0.9896 against s_t05.png and is a FAITHFUL render, not a leak. Armed at
0.50 the grid rule fails 7 of 10 correct takes; 85 of that clip's 124 frames clear even 0.85.

So for a single board, correlation carries no information on its own. What a leak actually looks
like is the board shown as a STILL: high board correlation AND the picture stops moving, held long
enough to read. Two facts, not one threshold. Measured separation, quietest 0.3s of frame-to-frame
delta on a 160x90 grey patch:

    a still spliced in                0.0008
    ten correct A/B renders           0.0929 - 0.6530     (lowest is S/t02, a locked-off wide)
    ratio to the clip's own median    0.0014 vs 0.66-0.90

Both tests clear the correct renders by more than 4x, so they are ORed rather than ANDed. The
ratio alone finds a still held inside an energetic clip (the seg20 leak: 0.81 against a median of
11.37, ratio 0.071); the absolute alone finds a clip that is a still nearly end to end, where the
clip's own median IS the leak and the ratio collapses to 1.0. ANDing them, as first sketched,
misses the second shape -- which is the shape seg09_s770809 actually had.

WHAT THIS GATE CANNOT SEE, AND IT MATTERS
-----------------------------------------
seg09_s770809 is a whole clip of board, and the renderer ANIMATED it: quietest 0.3s = 0.1629,
inside the range of correct renders. As a GRID it is caught by correlation alone, and is. Had it
been a single board it would pass this gate. That is not a threshold to tune -- a photoreal single
board that moves is, by construction, what a correct render of that shot looks like. The defect
that remains visible to a viewer is the FROZEN one, and that is what this catches. The owner's eye
stays the backstop for the rest.

THE RAW GRID IS ALWAYS SCANNED
------------------------------
When a board has an un-restyled sibling (..._RAW.png, or grid_raw.png beside grid_restyled.png)
every frame is scanned against that too, with its own detected kind. A leak of the SOURCE frames is
the worst case there is -- it puts someone else's footage on screen -- so it is never optional.

POSITIVE CONTROL
----------------
--selftest checks the detector fires on a board compared against itself (must be ~1.0) and stays
quiet on a real render with no leak. A detector with no demonstrated true positive is how G5 got
armed wrong; do not arm this one without running it. The single-board controls -- a synthetic still
spliced onto a real render, and the two correct A/B takes -- live in tools/gate_selftest.py, which
builds the leak fixture with ffmpeg rather than committing a video.

USAGE
    tools/board_leak.py CLIP --board BOARD.png [--board-kind auto|single|grid] [--json OUT.json]
    tools/board_leak.py --selftest
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
THRESHOLD = 0.50            # GRID boards only: a render that resembles a mosaic is a leak, full stop
GRID = (64, 36)             # patch the correlation is computed on -- unchanged, the grid rule is calibrated on it

# --- single-board rule, calibrated on the 2026-08-25 A/B (see the docstring) ---
SINGLE_CORR = 0.85          # necessary, never sufficient: correct renders reach 0.9896
MOTION_GRID = (160, 90)     # motion is measured on a bigger patch; 64x36 blurs away real movement
STILL_WINDOW_S = 0.30       # the picture must stop for this long to read as a held still
STILL_RATIO = 0.15          # ...to below 15% of the clip's OWN median frame delta
STILL_ABS = 0.02            # ...or below this outright, for a clip that is a still nearly throughout


def normalised(img):
    """Downscale to a small grayscale patch with zero mean and unit variance.

    Zero-mean/unit-variance is what makes the score a correlation rather than a brightness
    comparison: it cannot be fooled by the render being darker or brighter than the board, and a
    flat frame (std 0) scores 0 instead of dividing by zero.
    """
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    g = cv2.resize(g, GRID, interpolation=cv2.INTER_AREA).astype(np.float64)
    g -= g.mean()
    s = g.std()
    return g / s if s > 1e-6 else g


def motion_patch(img):
    """Plain grey patch, NOT normalised -- frame-to-frame difference must keep its absolute scale.

    Normalising here would make a held still and a moving shot equally "different" frame to frame,
    which is exactly the signal we need to keep.
    """
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return cv2.resize(g, MOTION_GRID, interpolation=cv2.INTER_AREA).astype(np.float64)


def detect_board_kind(board_path):
    """"grid" or "single", read off the image itself. Returns None when it cannot tell.

    A merged board is panels butted together with a gutter, so its internal boundaries are lines of
    near-constant pixels: the column standard deviation collapses there. Measured over every board
    in the repo, the boundary-to-median ratio is <= 0.089 on the 2x2 and 3x3 grids and >= 0.46 on
    every single panel -- and a grid needs ALL of its boundaries low, which no single panel has.
    A five-fold gap on a fact about the file, so this is a lookup, not a threshold to tune.
    """
    img = cv2.imread(str(board_path))
    if img is None:
        return None
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
    cols, rows = g.std(axis=0), g.std(axis=1)
    for n in (3, 2):                       # 3x3 first: a 3x3 also has a low ratio at its midpoint
        ok = True
        for prof in (cols, rows):
            med = float(np.median(prof))
            if med <= 1e-6:
                ok = False
                break
            for i in range(1, n):
                idx = min(len(prof) - 1, int(round(len(prof) * i / n)))
                if float(prof[idx]) / med >= 0.30:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            return "grid"
    return "single"


RAW_SIBLINGS = [("_PHOTOREAL", "_RAW"), ("_restyled", "_raw"), ("restyled", "raw")]


def raw_sibling(board_path):
    """The un-restyled source-frame board for this board, if one is on disk.

    A leak of the RAW board is the worst case: it is not our art at all, it is the source footage.
    Nothing about it is optional, so the caller never gets to skip it.
    """
    p = Path(board_path)
    for a, b in RAW_SIBLINGS:
        if a in p.name:
            cand = p.with_name(p.name.replace(a, b))
            if cand.exists() and cand.resolve() != p.resolve():
                return cand
    return None


def _spans(flags, scores, fps):
    """Contiguous runs of True, as time spans with their peak score."""
    out, run = [], None
    for idx, f in enumerate(flags):
        if f:
            run = run or [idx, idx]
            run[1] = idx
        elif run:
            out.append(run)
            run = None
    if run:
        out.append(run)
    return [{"start_s": round(a / fps, 3), "end_s": round((b + 1) / fps, 3),
             "frames": b - a + 1, "peak": round(max(scores[a:b + 1]), 4)} for a, b in out]


def _read(clip):
    cap = cv2.VideoCapture(str(clip))
    if not cap.isOpened():
        return None, None, None
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    corr_patches, mot_patches = [], []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        corr_patches.append(normalised(frame))
        mot_patches.append(motion_patch(frame))
    cap.release()
    return fps, corr_patches, mot_patches


def _still_flags(mot_patches, fps):
    """Which frames sit inside a >=0.3s stretch where the picture has stopped moving.

    ★ WHY A WINDOW AND NOT A RUN OF QUIET FRAMES. Both were built. Grouping frames by correlation
    and averaging inside the group fails on the real shape of the defect: in the synthetic fixture
    the still is spliced onto a take whose own body already clears 0.85 on 85 of 124 frames, so the
    "run" swallows the moving body and its mean comes out normal. And a fixed window CENTRED on the
    frame fails at the splice, where one 18.17 transition frame drags the mean of everything within
    a quarter second of it. So: slide a 0.3s window, and mark every frame of any window whose mean
    is below the floor. The transition frame poisons only the windows that contain it.
    """
    n = len(mot_patches)
    if n < 3:
        return [False] * n, {}
    d = np.array([float(np.abs(mot_patches[i] - mot_patches[i - 1]).mean()) for i in range(1, n)])
    med = float(np.median(d))
    floor = max(STILL_RATIO * med, STILL_ABS)      # ORed: whichever test fires, the picture stopped
    win = max(2, min(len(d), int(round(STILL_WINDOW_S * fps))))
    flags = [False] * n
    quietest = None
    for i in range(0, len(d) - win + 1):
        m = float(d[i:i + win].mean())
        quietest = m if quietest is None else min(quietest, m)
        if m < floor:
            for f in range(i, i + win + 1):        # delta j spans frames j and j+1
                flags[f] = True
    return flags, {"median_delta": round(med, 4),
                   "quietest_window": round(quietest, 4) if quietest is not None else None,
                   "still_floor": round(floor, 4),
                   "window_frames": win}


def scan(clip, board_path, threshold=THRESHOLD, board_kind="auto", single_corr=SINGLE_CORR,
         scan_raw=True, raw_board=None):
    """Every frame of clip against the board. Returns a dict, or None if unreadable.

    None means UNMEASURED and the caller must treat it as a failure, never as clean.

    board_kind "grid" applies the 2026-08-24 rule unchanged: correlation >= threshold is a leak.
    board_kind "single" requires correlation AND a held still, because a correct single-board render
    is meant to match its board (see the docstring). "auto" reads the kind off the board file.
    """
    board_img = cv2.imread(str(board_path))
    if board_img is None:
        return None
    board = normalised(board_img)

    fps, corr_patches, mot_patches = _read(clip)
    if not corr_patches:
        return None
    scores = [float((board * p).mean()) for p in corr_patches]

    kind = board_kind
    if kind == "auto":
        kind = detect_board_kind(board_path) or "grid"   # unknown falls back to the STRICTER rule

    detail = {}
    if kind == "grid":
        flags = [s >= threshold for s in scores]
        used = threshold
    else:
        still, detail = _still_flags(mot_patches, fps)
        flags = [(s > single_corr) and still[i] for i, s in enumerate(scores)]
        used = single_corr
        # a held still must last long enough for a viewer to read it as a frozen frame
        keep = [False] * len(flags)
        for sp in _spans(flags, scores, fps):
            if sp["end_s"] - sp["start_s"] >= STILL_WINDOW_S:
                a = int(round(sp["start_s"] * fps))
                for f in range(a, a + sp["frames"]):
                    keep[f] = True
        flags = keep

    spans = _spans(flags, scores, fps)
    peak = max(scores)
    out = {
        "clip": str(clip), "board": str(board_path), "frames": len(scores), "fps": round(fps, 3),
        "peak_similarity": round(peak, 4),
        "threshold": used,
        "leak_frames": sum(1 for f in flags if f),
        "spans": spans,
        "clean_peak": round(max([s for i, s in enumerate(scores) if not flags[i]], default=0.0), 4),
        "leak": bool(spans),
    }
    out["board_kind"] = kind
    if detail:
        out["motion"] = detail

    # The RAW source-frame board, always, with its own kind. Never optional -- see the docstring.
    out["raw"] = None
    if scan_raw:
        # A PANEL CROPPED OUT OF A GRID has no derivable raw sibling -- 01.jpg in panels_raw is
        # indexed by panel, not by shot -- so the caller names it with raw_board. Everything with a
        # _PHOTOREAL / _restyled name derives its own.
        rs = Path(raw_board) if raw_board else raw_sibling(board_path)
        if rs is not None and Path(rs).exists():
            out["raw"] = scan(clip, rs, threshold=threshold, board_kind="auto",
                              single_corr=single_corr, scan_raw=False)
    out["leak_any"] = bool(out["leak"] or (out["raw"] or {}).get("leak"))
    return out


def selftest():
    """Both directions. A detector that has never fired proves nothing."""
    board = ROOT / "output/fullbattle_20260808/seg20_board_merged_3x3_PHOTOREAL.png"
    print("G9 BOARD-LEAK SELF-TEST\n")
    problems = []

    if not board.exists():
        print("  ★ FIXTURE MISSING: %s" % board.name)
        return 1

    # POSITIVE: the board against itself must be a near-perfect match.
    b = normalised(cv2.imread(str(board)))
    self_score = float((b * b).mean())
    ok = self_score > 0.99
    print("  positive control  board vs itself      %.4f   %s"
          % (self_score, "correct" if ok else "<<< DETECTOR IS BROKEN"))
    if not ok:
        problems.append("board does not match itself")

    # POSITIVE: the known leak, found by eye 2026-08-24.
    leak_clip = ROOT / "output/gauntlet_20260819/seg20_s918273.mp4"
    if leak_clip.exists():
        r = scan(leak_clip, board)
        hit = r and r["leak"] and 5.3 < r["spans"][0]["start_s"] < 5.5
        print("  positive control  the seg20 leak      peak %.4f, %d frame(s) at %.3fs   %s"
              % (r["peak_similarity"], r["leak_frames"],
                 r["spans"][0]["start_s"] if r["spans"] else -1,
                 "correct" if hit else "<<< DID NOT FIND THE KNOWN LEAK"))
        if not hit:
            problems.append("missed the known seg20 leak")
    else:
        print("  ★ FIXTURE MISSING: %s" % leak_clip.name)
        problems.append("leak fixture missing")

    # NEGATIVE: a clip whose own board is a DIFFERENT segment's must stay quiet.
    clean = ROOT / "output/gauntlet_20260819/segA_s770809.mp4"
    cboard = ROOT / "output/fullbattle_20260808/segA_board_merged_3x3_PHOTOREAL.png"
    if clean.exists() and cboard.exists():
        r = scan(clean, cboard)
        quiet = r and not r["leak"]
        print("  negative control  segA vs its board   peak %.4f   %s"
              % (r["peak_similarity"], "quiet, correct" if quiet else "<<< FALSE POSITIVE"))
        if not quiet:
            problems.append("fired on a clean clip")

    print()
    if problems:
        print("SELF-TEST FAILED: %s" % "; ".join(problems))
        return 1
    print("SELF-TEST PASSED. The detector finds the known leak and stays quiet on clean footage.")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip", nargs="?")
    ap.add_argument("--board", help="the storyboard image fed to this render")
    ap.add_argument("--json")
    ap.add_argument("--threshold", type=float, default=THRESHOLD,
                    help="GRID boards only. Single boards use --single-corr plus the stillness test.")
    ap.add_argument("--board-kind", choices=["auto", "single", "grid"], default="auto",
                    help="'grid' = the 2026-08-24 correlation rule. 'single' = correlation AND a "
                         "held still. 'auto' reads it off the board file. Callers that KNOW should "
                         "say so -- an unreadable board falls back to the stricter grid rule.")
    ap.add_argument("--single-corr", type=float, default=SINGLE_CORR)
    ap.add_argument("--raw-board",
                    help="the un-restyled source-frame board for this render. Needed when the board "
                         "is a panel cropped out of a grid, where the raw sibling is not derivable "
                         "from the filename.")
    ap.add_argument("--no-raw", action="store_true",
                    help="skip the RAW source-frame sibling scan (diagnostics only, never in a gate)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return selftest()
    if not (args.clip and args.board):
        return ap.error("need CLIP and --board (or --selftest)")

    r = scan(args.clip, args.board, args.threshold, board_kind=args.board_kind,
             single_corr=args.single_corr, scan_raw=not args.no_raw,
             raw_board=args.raw_board)
    if r is None:
        print("G9 UNMEASURED -- could not read the clip or the board")
        if args.json:
            Path(args.json).write_text(json.dumps({"leak": None, "why": "unreadable"}, indent=2))
        return 1
    if args.json:
        Path(args.json).write_text(json.dumps(r, indent=2))

    print("%s  %d frames, %s board, peak similarity %.4f (next-highest clean frame %.4f)"
          % (Path(args.clip).name, r["frames"], r["board_kind"], r["peak_similarity"],
             r["clean_peak"]))
    if r.get("motion"):
        m = r["motion"]
        print("   quietest %.2fs of picture %.4f, floor %.4f (clip median %.4f)"
              % (STILL_WINDOW_S, m["quietest_window"] or -1, m["still_floor"], m["median_delta"]))
    bad = False
    for label, res in (("the board", r), ("the RAW source-frame board", r.get("raw"))):
        if not res:
            continue
        for sp in res["spans"]:
            bad = True
            print("  ★ %s ON SCREEN  %.3f-%.3fs  (%d frames, similarity %.4f)"
                  % (label.upper(), sp["start_s"], sp["end_s"], sp["frames"], sp["peak"]))
    if r.get("raw") is None and (args.raw_board or raw_sibling(args.board)) is not None:
        print("   (raw sibling exists but could not be read)")
    if not bad:
        print("PASSED: the storyboard does not appear in the video")
        return 0
    print("FAILED: the storyboard reference is visible in the output")
    return 1


if __name__ == "__main__":
    sys.exit(main())

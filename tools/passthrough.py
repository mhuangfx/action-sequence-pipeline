#!/usr/bin/env python3
"""Measure how much of an H3 output is a near-copy of its reference video.

WHY THIS EXISTS
---------------
On 2026-08-18 the owner spotted the source's white-shirted young man -- the character Vesta is
supposed to REPLACE -- appearing in every generation of the 08-09 overnight batch. Pulling frames
showed it is not a leak of one character: for stretches of roughly half a second to three seconds
the renderer stops building our scene and simply replays the reference, source character and flat
cel rendering together. A passthrough, not a leak.

Nothing in the existing scorecard sees this. Direction agreement, cut offset and hit-stop counts all
score a copied frame as a perfect match -- a copy IS the reference, so it agrees with it completely.

WHY COLOUR, NOT STRUCTURE
-------------------------
Composition cannot separate copy from correct behaviour: the storyboard panels ARE restyled
reference frames, so an obedient output matches the reference's shapes by design. Colour does not.
Our palette is ash grey, gunmetal and ember; the source's is flat cel with a white shirt and blue
trousers. So frames are compared in colour at low resolution, searching a window in time because
H3's cut placement drifts by up to half a second.

CALIBRATION
-----------
Measured against segment 1 of the 08-08 battle:

    genC_seg1_board_only          no video reference   10%   <- the floor: board-following alone
    genB_seg1_board_plus_videoref leaky prompt         34%
    genD_seg1_scoped_videoref     the "fixed" prompt   30%   <- the 08-09 fix did NOT fix this
    overnight batch (14 runs)     15% - 49%

The 10% floor is what board-driven composition similarity looks like with no video reference at all.
Anything well above it is the reference bleeding through.

THRESHOLD WARNING
-----------------
Those numbers come from runs whose look was ANIME, matching the source. The pipeline moved to a
photoreal look on 2026-08-18, and the opponent is now deliberately still recognisable. Both changes
move this metric. RE-DERIVE the threshold on the first photoreal segment -- measure a known-good and
a known-bad run -- before gating on it. Do not carry the old 15% figure over unexamined.
"""
import argparse
import subprocess
import sys

import numpy as np

W, H, FPS = 64, 36, 12


def frames(path, fps=FPS):
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"fps={fps},scale={W}:{H}",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True, check=True).stdout
    if not raw:
        raise SystemExit(f"no frames decoded from {path}")
    return np.frombuffer(raw, np.uint8).reshape(-1, H, W, 3).astype(np.float32) / 255


def similarity(out_path, ref_path, window=0.6):
    """Per-output-frame best colour match against the reference, within +/- window seconds."""
    o, r = frames(out_path), frames(ref_path)
    scale = len(r) / len(o)          # output and reference need not be the same length
    w = int(window * FPS)
    best = np.zeros(len(o), dtype=np.float32)
    for i, f in enumerate(o):
        j = int(i * scale)
        cand = r[max(0, j - w): min(len(r), j + w + 1)]
        if not len(cand):
            continue
        mae = np.abs(cand - f).reshape(len(cand), -1).mean(axis=1)
        best[i] = 1 - mae.min()
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("reference", help="the reference clip the output was generated against")
    ap.add_argument("clips", nargs="+", help="one or more H3 outputs to score")
    ap.add_argument("--threshold", type=float, default=0.90,
                    help="a frame at or above this similarity counts as a copy (default 0.90)")
    ap.add_argument("--window", type=float, default=0.6,
                    help="seconds either side to search, absorbing cut drift (default 0.6)")
    ap.add_argument("--max-copy", type=float, default=None,
                    help="exit 1 if any clip's copy fraction exceeds this (e.g. 0.15). "
                         "RE-DERIVE this for the photoreal look before using it as a gate.")
    ap.add_argument("--json", help="write the scores here")
    args = ap.parse_args()

    results, failed = {}, False
    for clip in args.clips:
        b = similarity(clip, args.reference, args.window)
        hot = np.where(b > args.threshold)[0]
        frac = float(len(hot) / len(b))
        # contiguous runs of copied frames -- one long replay reads very differently from
        # scattered single frames, and only the long runs are visible to a viewer
        spans, start = [], None
        for i in range(len(b)):
            is_copy = b[i] > args.threshold
            if is_copy and start is None:
                start = i
            elif not is_copy and start is not None:
                spans.append((start / FPS, i / FPS)); start = None
        if start is not None:
            spans.append((start / FPS, len(b) / FPS))
        longest = max((e - s for s, e in spans), default=0.0)
        results[clip] = {"copy_fraction": round(frac, 3),
                         "longest_copy_seconds": round(longest, 2),
                         "peak_similarity": round(float(b.max()), 3),
                         "spans": [[round(s, 2), round(e, 2)] for s, e in spans if e - s >= 0.2]}
        print("%-40s copy %4.1f%%  longest %.2fs  peak %.3f  spans %s"
              % (clip.split("/")[-1], frac * 100, longest, b.max(),
                 results[clip]["spans"][:6]))
        if args.max_copy is not None and frac > args.max_copy:
            failed = True

    if args.json:
        import json
        with open(args.json, "w") as fh:
            json.dump(results, fh, indent=2)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()

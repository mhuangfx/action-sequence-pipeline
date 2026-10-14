#!/usr/bin/env python3
"""stutter_check.py — detect half-rate stutter and frozen frames in H3 renders.

For each clip, compute the mean absolute difference between consecutive frames (320x180 gray).
  dup_frac  = fraction of frame pairs with mean diff < 0.5 (near-duplicate frames)
  alt_ratio = mean(even-index diffs) / mean(odd-index diffs), larger over smaller.
              ~1.0 = smooth 24 fps. >=1.6 = every second frame barely moves (12 fps
              rendered inside a 24 fps file). Measured 2026-08-28: ~45% of our renders
              show it; it varies by SEED, not by prompt, so re-roll on failure.
Usage: stutter_check.py "renders/*.mp4" [--alt 1.6] [--dup 0.3]
"""
import cv2, sys, glob, argparse, numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("pattern")
ap.add_argument("--alt", type=float, default=1.6)
ap.add_argument("--dup", type=float, default=0.3)
a = ap.parse_args()
bad = 0
for f in sorted(glob.glob(a.pattern)):
    cap = cv2.VideoCapture(f); fps = cap.get(cv2.CAP_PROP_FPS)
    prev = None; diffs = []
    while True:
        ok, fr = cap.read()
        if not ok: break
        g = cv2.cvtColor(cv2.resize(fr, (320, 180)), cv2.COLOR_BGR2GRAY).astype(np.float32)
        if prev is not None: diffs.append(float(np.abs(g - prev).mean()))
        prev = g
    d = np.array(diffs)
    if len(d) < 4: print(f"{f}\tTOO SHORT"); continue
    dup = float((d < 0.5).mean()); ev = d[0::2].mean(); od = d[1::2].mean()
    ratio = max(ev, od) / max(min(ev, od), 1e-6)
    flag = "STUTTER" if ratio >= a.alt else ("FROZEN" if dup >= a.dup else "ok")
    bad += flag != "ok"
    print(f"{f}\tfps={fps:.0f}\tframes={len(d)+1}\tdup_frac={dup:.2f}\tmeandiff={d.mean():.2f}\talt_ratio={ratio:.2f}\t{flag}")
print(f"# flagged {bad}", file=sys.stderr)

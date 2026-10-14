#!/usr/bin/env python3
"""motion_stats.py — summarise a clip's per-shot motion into one small JSON.

The style A/B decides on three things: medium, identity, and whether the motion still reads. The
first two are scored by identity_gate.py. This covers the third as far as a number can: mean motion
energy, and the top-to-bottom RANGE across shots. Range is the metric that matters -- H3 preserves
mean motion while erasing contrast (2026-08-08), and a clip whose shots all move the same amount
reads as floaty even when the average is right.

Direction agreement is deliberately NOT here: it swings 8-79% on seed alone (2026-08-10).

Usage: .venv/bin/python tools/motion_stats.py CLIP --json OUT.json [--work DIR]
"""
import argparse, json, subprocess, sys, tempfile
from pathlib import Path


def measure(clip, work):
    subprocess.run([sys.executable, str(Path(__file__).with_name("shot_breakdown.py")), str(clip),
                    "--out-dir", str(work), "--motion", "--subject"],
                   check=True, capture_output=True, text=True)
    d = json.loads((Path(work) / "shots.json").read_text())
    return d["shots"] if isinstance(d, dict) else d


def summarise(shots):
    m = [s["motion"] for s in shots if s.get("motion") is not None]
    if not m:
        return {"cuts": len(shots)}
    subj = [s.get("subject") or {} for s in shots]
    return {"cuts": len(shots),
            "motion_mean": round(sum(m) / len(m), 2),
            "motion_min": round(min(m), 2),
            "motion_max": round(max(m), 2),
            "motion_range": round(max(m) / max(min(m), 0.01), 2),
            "hitstops": sum(1 for s in subj if s.get("hitstop")),
            "reversals": sum(1 for s in subj if s.get("reversal"))}


# ---------------------------------------------------------------------------
# G8 -- "the segment went flat"
#
# The original G8 gated on motion_range < 2.5x. It NEVER WORKED. Measured on the three 2026-08-19
# noref seeds of segA, against the source:
#
#     seed 770809   mean 14.60   max 27.16   range 3.60    <- good, owner accepted
#     seed 424242   mean 13.54   max 24.50   range 9.80    <- good, owner accepted
#     seed 131313   mean 10.21   max 15.09   range 2.89    <- THE FLAT ONE
#     SOURCE        mean 13.71   max 27.39   range 4.08    <- the target
#
# The flat run scores 2.89 and PASSES a 2.5 gate. Worse, range cannot separate the classes at all:
# the good 770809 (3.60) sits nearer the flat run than it does the good 424242 (9.80), because range
# is dominated by motion_min, which swings 2.50-7.54 on noise.
#
# PEAK is the clean separator: 15.09 against 24.50 / 27.16 / 27.39. The good runs land within 11% of
# the source; the flat one is 45% below it. Mean separates too, if less sharply.
#
# Thresholds sit midway between the flat run and the good ones, so the gate fires on real failure
# and stays quiet on footage the owner has already accepted. That restraint is deliberate: a gate
# that fails known-good footage is what made G4 worthless.
MEAN_FLOOR = 12.0     # flat run 10.21 | good runs 13.54, 14.60
PEAK_FLOOR = 20.0     # flat run 15.09 | good runs 24.50, 27.16
SOURCE_MEAN = 13.71   # segA source -- the ideal, used to rank the fallback
SOURCE_PEAK = 27.39


def gate(out, mean_floor=MEAN_FLOOR, peak_floor=PEAK_FLOOR):
    """G8 verdict plus a distance-from-ideal score for the after-3-retries fallback.

    `distance` is ONE-SIDED: overshooting the source is not a defect, only falling short is. It is
    the fraction by which the clip undershoots the source, summed over mean and peak, so 0.0 means
    "at or above the source on both". Rank rejected candidates by it and keep the smallest.
    """
    mean, peak = out.get("motion_mean"), out.get("motion_max")
    if mean is None or peak is None:
        return {"pass": None, "why": "no motion measured"}
    fails = []
    if mean < mean_floor:
        fails.append("mean %.2f < %.1f" % (mean, mean_floor))
    if peak < peak_floor:
        fails.append("peak %.2f < %.1f" % (peak, peak_floor))
    distance = round(max(0.0, (SOURCE_MEAN - mean) / SOURCE_MEAN)
                     + max(0.0, (SOURCE_PEAK - peak) / SOURCE_PEAK), 4)
    return {"pass": not fails, "failures": fails, "distance_from_source": distance,
            "thresholds": {"mean_floor": mean_floor, "peak_floor": peak_floor}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("--json", required=True)
    ap.add_argument("--work", help="analysis scratch dir; a temp dir by default")
    ap.add_argument("--gate", action="store_true",
                    help="apply G8 and exit 1 on a breach, so a harness can re-roll the seed")
    ap.add_argument("--mean-floor", type=float, default=MEAN_FLOOR)
    ap.add_argument("--peak-floor", type=float, default=PEAK_FLOOR)
    args = ap.parse_args()
    work = Path(args.work) if args.work else Path(tempfile.mkdtemp(prefix="motionstats_"))
    work.mkdir(parents=True, exist_ok=True)
    out = summarise(measure(args.clip, work))
    out["g8"] = gate(out, args.mean_floor, args.peak_floor)
    Path(args.json).write_text(json.dumps(out, indent=1))
    print(json.dumps(out))
    if args.gate and out["g8"]["pass"] is False:
        print("G8 FAILED: %s  (distance from source %.3f)"
              % ("; ".join(out["g8"]["failures"]), out["g8"]["distance_from_source"]),
              file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

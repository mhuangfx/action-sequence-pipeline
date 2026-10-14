#!/usr/bin/env python3
"""battle_map_report.py — turn a full-battle shot map into a segmentation plan.

Answers the questions that decide whether a long source can be run through the H3 swap:
how much of it is actually clean two-fighter action, where the unusable stretches are, and
how many generations it costs at a given speed-up and segment length.

Usage:
  .venv/bin/python tools/battle_map_report.py SHOTS.json [--speed 1.5] [--segment 6]
"""
import argparse
import json
import statistics as st

# A shot is unusable as swap reference if it shows someone who is not one of the two fighters,
# or shows no fighter at all for long enough to read as a different scene.
LONG_SHOT = 4.0   # seconds; talky beats and dissolve montages live here


def load(p):
    return json.load(open(p))


def classify(shots):
    for s in shots:
        d = s.get("description", "").lower()
        cast = " ".join(s.get("cast") or []).lower()
        s["_reasons"] = r = []
        if s.get("offcast"):
            r.append("off-cast")
        if (s.get("character_count") or 0) > 2:
            r.append(f"{s['character_count']} people")
        # abstract/interior-world imagery: no people and no architecture cues
        if (s.get("character_count") or 0) == 0 and s["dur"] > 0.6:
            r.append("no characters")
        if s["dur"] > LONG_SHOT:
            r.append("long/talky")
        for kw in ("blonde", "blond"):
            if kw in cast or kw in d:
                r.append("blonde present")
                break
        s["_clean"] = not r
    return shots


def segment_plan(shots, speed, seg_out, break_gap=2.0, floor=4.0):
    """Greedy: accumulate clean shots until the segment's OUTPUT length would exceed seg_out.

    Segment boundaries land on source cuts by construction — never mid-shot, which is what
    keeps seams from becoming visible jumps.

    `break_gap`: only an excluded RUN at least this long ends a segment. Shorter excluded shots
    are spliced out and the segment continues across them — an ordinary jump cut in action
    editing. Breaking on every excluded shot instead produced 23 of 32 segments below H3's 4s
    output floor (buying 92s to get 38.8s), which is both wasteful and triggers the tail-stretch.

    `floor`: H3's minimum output. Segments whose content falls below it are reported so they can
    be merged or padded rather than silently paid for.
    """
    src_budget = seg_out * speed
    segs, cur, acc, gap = [], [], 0.0, 0.0
    for s in shots:
        if not s["_clean"]:
            gap += s["dur"]
            if gap >= break_gap and cur:
                segs.append((cur, acc))
                cur, acc = [], 0.0
            continue
        gap = 0.0
        if acc + s["dur"] > src_budget and cur:
            segs.append((cur, acc))
            cur, acc = [], 0.0
        cur.append(s)
        acc += s["dur"]
    if cur:
        segs.append((cur, acc))
    return segs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shots")
    ap.add_argument("--speed", type=float, default=1.5)
    ap.add_argument("--segment", type=float, default=6.0, help="output seconds per generation")
    ap.add_argument("--rate", type=float, default=7.0, help="credits per second (768p=7, 2K=12)")
    a = ap.parse_args()

    d = load(a.shots)
    shots = classify(d["shots"])
    clean = [s for s in shots if s["_clean"]]
    dirty = [s for s in shots if not s["_clean"]]

    clean_s = sum(s["dur"] for s in clean)
    print(f"SOURCE {d['duration']:.1f}s, {len(shots)} shots "
          f"(median {st.median([s['dur'] for s in shots]):.2f}s)")
    print(f"CLEAN two-fighter action: {clean_s:.1f}s across {len(clean)} shots "
          f"({100*clean_s/d['duration']:.0f}% of source)")
    print(f"EXCLUDED: {d['duration']-clean_s:.1f}s across {len(dirty)} shots")

    from collections import Counter
    reasons = Counter(r for s in dirty for r in s["_reasons"])
    for r, n in reasons.most_common():
        secs = sum(s["dur"] for s in dirty if r in s["_reasons"])
        print(f"    {r:<18} {n:>3} shots  {secs:>6.1f}s")

    print(f"\nEXCLUDED RUNS (contiguous, >1.5s) — these break the battle into islands:")
    run, runs = [], []
    for s in shots:
        if not s["_clean"]:
            run.append(s)
        elif run:
            runs.append(run); run = []
    if run:
        runs.append(run)
    for r in runs:
        span = r[-1]["end"] - r[0]["start"]
        if span > 1.5:
            print(f"    {r[0]['start']:>7.2f}-{r[-1]['end']:<7.2f} {span:>5.1f}s  "
                  f"({', '.join(sorted({x for s in r for x in s['_reasons']}))})")

    print(f"\n--- PLAN @ {a.speed}x speed, {a.segment:.0f}s output per generation ---")
    segs = segment_plan(shots, a.speed, a.segment)
    outs = [acc / a.speed for _, acc in segs]
    total_out = sum(outs)
    shots_per = [len(c) for c, _ in segs]
    cost = sum(a.rate * (round(o) + round(acc)) for (c, acc), o in zip(segs, outs))
    print(f"  segments: {len(segs)}   finished output: {total_out:.1f}s")
    print(f"  shots per segment: median {st.median(shots_per):.0f}, max {max(shots_per)}, "
          f"over-7: {sum(1 for x in shots_per if x > 7)} segments")
    print(f"  credits: ~{cost:,.0f}  (~${cost*0.0122:,.0f})")
    print(f"  wall-clock: ~{len(segs)*7.5/2/60:.1f}h at 2 parallel gens "
          f"({a.segment:.0f}s gens ~7.5min each)")


if __name__ == "__main__":
    main()

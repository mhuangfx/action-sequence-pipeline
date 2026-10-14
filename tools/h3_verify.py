#!/usr/bin/env python3
"""Score an H3 output against its reference clip on subject screen-direction.

This is recipe step 8, automated. The defect it catches is directional
INCOHERENCE: outputs that move as much as the source, or more, but drive bodies
the OPPOSITE way on some cuts, so momentum contradicts itself across the edit and
the choreography stops reading. Segment 1 shipped with 4 of 9 axes opposite.

Because the prompt dictates the output's cut timings, the output is measured on
the reference's shot list rescaled to the output duration -- not re-detected, so
shot N means the same shot on both sides.

An axis counts only when BOTH sides move meaningfully on it (|v| >= --deadzone);
near-still axes have no direction to agree about and would otherwise inflate the
score. Exit status is 1 when agreement falls under --min-agreement, so an
unattended run can gate on it.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

AXES = ("dx", "dy")


def probe_duration(path):
    return float(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True).stdout.strip())


def load_shots(path):
    d = json.loads(Path(path).read_text())
    return d["shots"] if isinstance(d, dict) else d


def measure(clip, out_dir, ref_shots_path=None, scale=1.0):
    """Analyse a clip. With ref_shots_path the reference boundaries are imposed;
    without it the clip's own cuts are detected."""
    argv = [sys.executable, str(Path(__file__).with_name("shot_breakdown.py")), str(clip),
            "--out-dir", str(out_dir), "--motion", "--camera", "--subject"]
    if ref_shots_path:
        argv += ["--shots-json", str(ref_shots_path), "--shots-scale", "%.9f" % scale]
    subprocess.run(argv, check=True, capture_output=True, text=True)
    return load_shots(Path(out_dir) / "shots.json")


def match_by_overlap(expected, got):
    """Pair each expected shot with the output shot it overlaps most.

    H3 does not place cuts exactly where the prompt asks -- observed offsets run
    to half a second and the shot COUNT can differ -- so comparing shot i to
    shot i, or imposing the reference's boundaries on the output, measures
    motion across the wrong windows and the direction score is meaningless.
    """
    pairs = []
    for e in expected:
        best, best_ov = None, 0.0
        for g in got:
            ov = min(e["end"], g["end"]) - max(e["start"], g["start"])
            if ov > best_ov:
                best, best_ov = g, ov
        if best is not None:
            pairs.append((e, best, best_ov))
    return pairs


def cut_fidelity(expected, got):
    """How closely the output reproduced the prompted cut points."""
    exp_cuts = [e["start"] for e in expected[1:]]
    got_cuts = [g["start"] for g in got[1:]]
    if not exp_cuts:
        return {"expected": 0, "detected": len(got_cuts), "mean_offset": None}
    offsets = [min(abs(c - g) for g in got_cuts) for c in exp_cuts] if got_cuts else []
    return {"expected": len(exp_cuts), "detected": len(got_cuts),
            "mean_offset": round(sum(offsets) / len(offsets), 3) if offsets else None,
            "max_offset": round(max(offsets), 3) if offsets else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("output_clip")
    ap.add_argument("--ref-shots", required=True,
                    help="shots.json for the reference, measured with --subject")
    ap.add_argument("--ref-clip", help="reference clip; defaults to the ref shots' own span")
    ap.add_argument("--out-dir", required=True, help="where to write the output's analysis")
    ap.add_argument("--deadzone", type=float, default=0.35,
                    help="ignore an axis unless both sides exceed this magnitude")
    ap.add_argument("--prompted-shots",
                    help="shot list the PROMPT asked for, if it differs from --ref-shots. "
                         "Cut fidelity is judged against this; direction still uses --ref-shots, "
                         "so two arms cut differently stay comparable on motion.")
    ap.add_argument("--assume-aligned", action="store_true",
                    help="impose the reference boundaries instead of detecting the "
                         "output's own cuts; only valid if cut timing is known to match")
    ap.add_argument("--min-agreement", type=float, default=0.75,
                    help="exit 1 below this pass rate")
    ap.add_argument("--json", help="write the scorecard here")
    args = ap.parse_args()

    ref = load_shots(args.ref_shots)
    if not ref or "subject" not in ref[0]:
        sys.exit("reference shots.json has no subject data -- rerun shot_breakdown --subject")

    ref_span = max(s["end"] for s in ref)
    out_dur = probe_duration(args.output_clip)
    scale = out_dur / ref_span
    print("reference %.3fs -> output %.3fs (shot list rescaled x%.4f)"
          % (ref_span, out_dur, scale))

    expected = [dict(r, start=r["start"] * scale, end=r["end"] * scale) for r in ref]
    got = measure(args.output_clip, args.out_dir,
                  args.ref_shots if args.assume_aligned else None, scale)

    if args.prompted_shots:
        pr = load_shots(args.prompted_shots)
        psc = out_dur / max(x["end"] for x in pr)
        prompted = [dict(x, start=x["start"] * psc, end=x["end"] * psc) for x in pr]
    else:
        prompted = expected
    fid = cut_fidelity(prompted, got)
    print("cut timing: %d prompted, %d produced" % (fid["expected"], fid["detected"]), end="")
    if fid["mean_offset"] is not None:
        print(", mean offset %.2fs (max %.2fs)" % (fid["mean_offset"], fid["max_offset"]))
    else:
        print()

    pairs = match_by_overlap(expected, got)
    rows, agree, oppose = [], 0, 0
    for r, o, ov in pairs:
        if not r.get("subject") or not o.get("subject"):
            continue
        for ax in AXES:
            rv, ovv = r["subject"].get(ax), o["subject"].get(ax)
            if rv is None or ovv is None:
                continue
            if abs(rv) < args.deadzone or abs(ovv) < args.deadzone:
                continue
            ok = (rv > 0) == (ovv > 0)
            agree += ok
            oppose += not ok
            rows.append({"shot": r["index"], "matched": o["index"], "overlap": round(ov, 2),
                         "axis": ax, "ref": rv, "out": ovv, "agree": ok})

    total = agree + oppose
    print("\n ref->out  ovl axis      ref      out")
    for row in rows:
        print("  %3d->%-3d %4.2f %-4s %7.2f  %7.2f   %s"
              % (row["shot"], row["matched"], row["overlap"], row["axis"],
                 row["ref"], row["out"], "ok" if row["agree"] else "OPPOSITE"))

    if not total:
        sys.exit("\nno comparable axes above the deadzone -- nothing to score")
    rate = agree / total
    print("\ndirection agreement: %d/%d axes (%.0f%%), %d opposite"
          % (agree, total, rate * 100, oppose))

    ref_stops = sum(1 for s in ref if (s.get("subject") or {}).get("hitstop"))
    out_stops = sum(1 for s in got if (s.get("subject") or {}).get("hitstop"))
    print("hit-stops: reference %d, output %d" % (ref_stops, out_stops))

    if args.json:
        Path(args.json).write_text(json.dumps(
            {"clip": args.output_clip, "agreement": rate, "agree": agree,
             "oppose": oppose, "axes": rows,
             "hitstops": {"ref": ref_stops, "out": out_stops},
             "cut_fidelity": fid},
            indent=2) + "\n")

    if rate < args.min_agreement:
        sys.exit("FAIL: below --min-agreement %.0f%%" % (args.min_agreement * 100))
    print("PASS")


if __name__ == "__main__":
    main()

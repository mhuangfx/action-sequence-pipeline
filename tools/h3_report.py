#!/usr/bin/env python3
"""Score finished H3 runs and regenerate a markdown report.

Written to be run repeatedly while a queue is still going: it scores only what it
has not scored yet, then rewrites the report from everything on disk. Rewriting
rather than appending keeps it idempotent, so a crash mid-write costs nothing and
a rerun cannot double-count.

The point is a record that outlives the process. If the machine dies at 3am the
report on disk still describes every run that had finished by then.
"""
import argparse, json, subprocess, sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).parent
OUT = Path("output/fullbattle_20260808")
FLOOR = {"pan": 1.2, "tilt": 1.2, "push": 0.8}


def load(p):
    d = json.loads(Path(p).read_text())
    return d["shots"] if isinstance(d, dict) else d


def dom(c):
    if not c or c.get("label") in (None, "locked off"):
        return "locked"
    a = {k: abs(c.get(k) or 0) / FLOOR[k] for k in FLOOR}
    ax = max(a, key=a.get)
    return "locked" if a[ax] == 0 else "%s%s" % (ax, "+" if (c.get(ax) or 0) > 0 else "-")


def mag(c):
    return max((abs(c.get(k) or 0) / FLOOR[k] for k in FLOOR), default=0.0) if c else 0.0


def pair(ref, got, scale):
    out = []
    for r in ref:
        rs, re_ = r["start"] * scale, r["end"] * scale
        best, bo = None, 0.0
        for g in got:
            ov = min(re_, g["end"]) - max(rs, g["start"])
            if ov > bo:
                best, bo = g, ov
        if best:
            out.append((r, best))
    return out


def camera_stats(ref_shots, out_shots, dur):
    scale = dur / max(s["end"] for s in ref_shots)
    m = w = nomove = 0
    rm, om = [], []
    for r, g in pair(ref_shots, out_shots, scale):
        rm.append(mag(r.get("camera")))
        om.append(mag(g.get("camera")))
        dr, dg = dom(r.get("camera")), dom(g.get("camera"))
        if dr == "locked":
            continue
        if dg == "locked":
            nomove += 1
        elif dr == dg:
            m += 1
        else:
            w += 1
    n = m + w + nomove
    return {"matched": m, "wrong": w, "nomove": nomove, "moving": n,
            "ref_mag": sum(rm) / len(rm) if rm else 0,
            "out_mag": sum(om) / len(om) if om else 0}


def score(name, ref_shots_path, prompted_path):
    """Run h3_verify for a clip if it has no score yet. Returns the score dict or None."""
    clip = OUT / (name + ".mp4")
    if not clip.exists():
        return None
    sj = OUT / (name + "_score.json")
    if not sj.exists():
        argv = [sys.executable, str(HERE / "h3_verify.py"), str(clip),
                "--ref-shots", str(ref_shots_path),
                "--out-dir", str(OUT / ("shots_rpt_" + name)),
                "--json", str(sj), "--min-agreement", "0"]
        if prompted_path:
            argv += ["--prompted-shots", str(prompted_path)]
        r = subprocess.run(argv, capture_output=True, text=True)
        if r.returncode != 0 and not sj.exists():
            return {"error": (r.stderr or r.stdout).strip().splitlines()[-1:] or ["failed"]}
    if not sj.exists():
        return None
    d = json.loads(sj.read_text())
    ana = OUT / ("shots_rpt_" + name) / "shots.json"
    if ana.exists():
        d["camera"] = camera_stats(load(ref_shots_path), load(ana), 5.875)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", required=True)
    ap.add_argument("--state")
    ap.add_argument("--ref-shots", required=True)
    ap.add_argument("--arm-prompted", action="append", default=[],
                    help="SUBSTR=path -- prompted shot list for runs whose name contains SUBSTR")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="H3 run report")
    ap.add_argument("--preamble", help="markdown file inlined above the table; not regenerated")
    args = ap.parse_args()

    jobs = json.loads(Path(args.queue).read_text())
    state_path = Path(args.state) if args.state else Path(args.queue).with_suffix(".state.json")
    state = json.loads(state_path.read_text()) if state_path.exists() else {"done": {}, "failed": {}}
    overrides = [tuple(x.split("=", 1)) for x in args.arm_prompted]

    rows = []
    for j in jobs:
        n = j["name"]
        prompted = next((p for sub, p in overrides if sub in n), None)
        rows.append((j, state["done"].get(n), state["failed"].get(n),
                     score(n, args.ref_shots, prompted)))

    done = sum(1 for _, d, _, _ in rows if d)
    failed = sum(1 for _, _, f, _ in rows if f)
    L = ["# %s" % args.title, "",
         "_Regenerated %s. %d of %d complete, %d failed._"
         % (datetime.now().strftime("%Y-%m-%d %H:%M"), done, len(rows), failed), ""]
    if args.preamble and Path(args.preamble).exists():
        L += [Path(args.preamble).read_text().rstrip(), ""]
    L += ["## Results", "",
         "| run | seed | status | mins | cuts | offset | direction | axes | cam match | cam mag out/ref |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for j, d, f, sc in rows:
        seed = j.get("seed", "")
        if f:
            st, mins = "FAILED", "%.0f" % (f["seconds"] / 60)
        elif d:
            st, mins = "done", "%.0f" % (d["seconds"] / 60)
        else:
            st, mins = "pending", ""
        if sc and "agreement" in sc:
            c = sc.get("cut_fidelity", {})
            cam = sc.get("camera", {})
            L.append("| %s | %s | %s | %s | %s/%s | %s | %d%% | %d | %s | %.2f/%.2f |"
                     % (j["name"], seed, st, mins, c.get("detected"), c.get("expected"),
                        c.get("mean_offset"), round(sc["agreement"] * 100),
                        sc["agree"] + sc["oppose"],
                        "%d/%d" % (cam.get("matched", 0), cam.get("moving", 0)) if cam else "-",
                        cam.get("out_mag", 0), cam.get("ref_mag", 0)))
        else:
            L.append("| %s | %s | %s | %s | | | | | | |" % (j["name"], seed, st, mins))

    # Paired summary: the reason this batch exists.
    def agg(sub):
        vals = [(sc["agreement"], sc.get("cut_fidelity", {}).get("mean_offset"))
                for j, d, f, sc in rows if sub in j["name"] and sc and "agreement" in sc]
        return vals
    u, m = agg("_unmerged_"), agg("_merged_s")
    if u or m:
        L += ["", "## Paired arms", "",
              "| arm | n | direction mean | direction range | offset mean |", "|---|---|---|---|---|"]
        for label, v in (("U unmerged (0.50s shots)", u), ("M merged (0.78s shots)", m)):
            if not v:
                continue
            ds = [a for a, _ in v]
            os_ = [o for _, o in v if o is not None]
            L.append("| %s | %d | %.0f%% | %.0f-%.0f%% | %s |"
                     % (label, len(v), 100 * sum(ds) / len(ds), 100 * min(ds), 100 * max(ds),
                        "%.3f" % (sum(os_) / len(os_)) if os_ else "-"))
        L += ["", "_Direction range across seeds is the noise floor: any single-run difference "
              "smaller than it is not evidence of anything._"]

    Path(args.out).write_text("\n".join(L) + "\n")
    print("%s  (%d done, %d failed)" % (args.out, done, failed))


if __name__ == "__main__":
    main()

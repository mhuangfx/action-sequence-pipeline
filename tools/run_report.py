#!/usr/bin/env python3
"""Turn a pass's ledger.json into something a human actually reads.

WHY THIS EXISTS
---------------
The owner watches final.mp4, not ledger.json. Without this, a segment that shipped carrying the
replaced character surfaces as one line in a scrollback buffer that has already gone. The council
review of 2026-08-19 called a run report a requirement rather than a nicety, and it is right: the
whole grading apparatus is worthless if its worst finding is invisible at the moment someone decides
whether to publish.

★ It prints a NOT MEASURED column. That is deliberate and it is the point. This project's signature
failure is a gate that never ran being indistinguishable from a gate that passed -- it has now
happened four separate times (G4's KeyError, G3's key-name bug, grade()'s missing else-branch, and
gate_selftest's vacuous skip). A blank is not a pass, and the report must never let it look like one.

USAGE
-----
    tools/run_report.py output/gauntlet_20260819
    tools/run_report.py output/gauntlet_20260819 --html      # also write index.html

Writes docs/gauntlet_run_YYYYMMDD.md, and index.html next to the ledger when --html is given.
Exit 2 if any shipped segment carries an identity or unmeasured defect -- same convention as
render_pass.py, so a wrapper cannot miss it.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ALL_GATES = ["G1", "G2", "G5", "G6", "G3", "G8"]


def shipped_failures(rec):
    """The failures of the clip that ACTUALLY SHIPPED -- not the last attempt.

    ★ Fixed 2026-08-20, same bug as the pass banner had. This read attempts[-1], so a segment whose
    third try was rejected was reported carrying that try's defects. On the first full pass seg09
    and seg19 were both listed with a costume break their shipped clips do not have, and seg2's
    numbers came from a take that lost. The retry ladder exists to discard those; a report that
    prints them anyway describes a video nobody assembled.

    Prefer the recorded field, and fall back to matching the chosen clip's seed for ledgers written
    before that field existed.
    """
    if "hard_failures" in rec:
        return rec["hard_failures"]
    chosen = Path(rec.get("chosen") or "").stem
    seed = chosen.split("_s")[-1] if "_s" in chosen else None
    for a in rec.get("attempts", []):
        if seed is not None and str(a.get("seed")) == seed:
            return a.get("hard_failures", [])
    return []


def classify(ledger, seg):
    """(shipped, clean, identity_hit, unmeasured_gates) for one segment."""
    rec = ledger.get(seg, {})
    if not rec.get("chosen"):
        return False, False, False, ALL_GATES
    best = shipped_failures(rec)
    identity = any(f.startswith(("G1", "G2")) or "UNMEASURED" in f for f in best)
    measured = rec.get("measured", {})
    unmeasured = [g for g in ALL_GATES if measured and measured.get(g) is not True]
    if not measured:
        unmeasured = ["(not recorded)"]
    return True, bool(rec.get("passed")), identity, unmeasured


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--html", action="store_true")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    ledger_path = run_dir / "ledger.json"
    if not ledger_path.exists():
        sys.exit("no ledger.json in %s -- has a pass run?" % run_dir)
    ledger = json.loads(ledger_path.read_text())
    cfg = json.loads((ROOT / "config/grading.json").read_text())
    order = [s for s in cfg["_order"] if s in ledger]
    stamp = run_dir.name.replace("gauntlet_", "")

    rows, shipped, clean, flagged, identity_segs, unmeasured_segs = [], 0, 0, [], [], []
    for seg in order:
        rec = ledger[seg]
        ship, ok, ident, unmeas = classify(ledger, seg)
        m = rec.get("metrics", {})
        atts = rec.get("attempts", [])
        last = shipped_failures(rec)
        shipped += ship
        clean += ok
        if ship and not ok:
            flagged.append(seg)
        if ident:
            identity_segs.append(seg)
        if ship and unmeas:
            unmeasured_segs.append(seg)
        rows.append({
            "seg": seg, "ship": ship, "ok": ok, "identity": ident,
            "seed": Path(rec.get("chosen") or "").stem.split("_s")[-1] if rec.get("chosen") else "-",
            "attempts": len(atts),
            "failures": last,
            "unmeasured": unmeas,
            "mean": m.get("motion_mean"), "peak": m.get("motion_max"),
            "cut": m.get("cut_offset_mean_s"), "copy": m.get("longest_copy_s"),
            "reports": rec.get("report_only", []),
        })

    L = []
    L.append("# Render pass — %s\n" % stamp)
    L.append("Generated from `%s`. Nothing here is hand-written.\n" % ledger_path)
    L.append("## Verdict\n")
    L.append("**%d of %d segments shipped. %d clean, %d shipped with flags.**\n"
             % (shipped, len(order), clean, len(flagged)))
    if identity_segs:
        L.append("> ## ★★ WATCH THESE BEFORE PUBLISHING\n>\n"
                 "> **%s** shipped carrying an identity defect — the wrong person on screen.\n"
                 "> The owner ranks a brief wrong-character frame as the worst possible defect,\n"
                 "> worse than a soft or flat beat.\n" % ", ".join(identity_segs))
    if unmeasured_segs:
        L.append("> ## ★ GATES THAT NEVER RAN\n>\n"
                 "> **%s** shipped with at least one gate unmeasured. A blank is NOT a pass.\n"
                 "> This project has shipped four separate bugs where an unmeasured gate looked\n"
                 "> exactly like a passing one.\n" % ", ".join(unmeasured_segs))
    if not identity_segs and not unmeasured_segs and shipped == len(order):
        L.append("No identity defects and no unmeasured gates. Every gate ran on every segment.\n")

    L.append("\n## Per segment\n")
    L.append("| seg | shipped | seed | tries | hard failures | NOT MEASURED | mean | peak | cut off | copy |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        def n(v, f="%.2f"):
            return f % v if isinstance(v, (int, float)) else "—"
        L.append("| %s | %s | %s | %d | %s | %s | %s | %s | %s | %s |" % (
            r["seg"],
            "★ IDENTITY" if r["identity"] else ("yes" if r["ok"] else ("flagged" if r["ship"] else "**NO**")),
            r["seed"], r["attempts"],
            "; ".join(r["failures"])[:70] or "none",
            "**" + ", ".join(r["unmeasured"]) + "**" if r["unmeasured"] else "none",
            n(r["mean"]), n(r["peak"]), n(r["cut"], "%.3f"), n(r["copy"])))

    L.append("\n## Report-only, never gated\n")
    L.append("G4, G7 cut offset, direction agreement and camera match are recorded and never acted "
             "on. G4 fires on 4-6 of 7 shots in every arm including the reference arm; G7 fires on 8 "
             "of 11 historical runs including the incumbent; direction agreement swings 8%-79% on "
             "the seed alone. **A single-run delta on any of these is dice.**\n")
    for r in rows:
        if r["reports"]:
            L.append("- **%s** — %s" % (r["seg"], "; ".join(r["reports"])))

    L.append("\n## How to read this\n")
    L.append("Candidates are ranked by lexicographic severity tiers, not a sum, so an identity hit "
             "cannot be out-voted by any number of lesser defects. After three attempts the closest "
             "candidate is kept rather than dropped, so a complete video always exists — which means "
             "**a shipped segment is not necessarily a clean one.** The flags above are the point.\n")

    out = ROOT / ("docs/gauntlet_run_%s.md" % stamp)
    out.write_text("\n".join(L) + "\n")
    print("wrote %s" % out.relative_to(ROOT))

    if args.html:
        h = ["<title>Render pass %s</title>" % stamp,
             "<style>body{font-family:system-ui;max-width:1100px;margin:2rem auto;padding:0 1rem}"
             "table{border-collapse:collapse;width:100%}td,th{border:1px solid #8884;padding:.4rem}"
             "video{width:320px}.bad{background:#c0392b22}.warn{background:#e67e2222}</style>",
             "<h1>Render pass %s</h1>" % stamp,
             "<p><b>%d/%d shipped, %d clean, %d flagged.</b></p>" % (shipped, len(order), clean, len(flagged))]
        if identity_segs:
            h.append("<p class=bad><b>★★ Identity defect shipped in: %s</b> — watch before "
                     "publishing.</p>" % ", ".join(identity_segs))
        h.append("<table><tr><th>seg<th>clip<th>state<th>hard failures<th>not measured<th>metrics</tr>")
        for r in rows:
            rec = ledger[r["seg"]]
            cls = "bad" if r["identity"] else ("warn" if not r["ok"] else "")
            src = rec.get("chosen")
            vid = ("<video src='%s' controls muted></video>"
                   % Path(src).name) if src else "—"
            h.append("<tr class=%s><td>%s<td>%s<td>%s<td>%s<td>%s<td>mean %s / peak %s / cut %s</tr>"
                     % (cls, r["seg"], vid,
                        "identity" if r["identity"] else ("clean" if r["ok"] else "flagged"),
                        "; ".join(r["failures"]) or "none",
                        ", ".join(r["unmeasured"]) or "none",
                        r["mean"], r["peak"], r["cut"]))
        h.append("</table>")
        (run_dir / "index.html").write_text("\n".join(h))
        print("wrote %s" % (run_dir / "index.html"))

    return 2 if (identity_segs or unmeasured_segs) else 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Render all twelve segments, grade each one, retry a failure, keep the closest. Then assemble.

WHAT THIS IS, AND WHAT IT DELIBERATELY IS NOT
---------------------------------------------
This is NOT the gauntlet loop. The loop was designed around a 41-minute render and a candidate
bake-off; measurement on 2026-08-19 killed both premises. A no-reference render takes 11m30s, not
41 minutes -- the video reference was costing 21m40s of every render, and it is gone. One full pass
over all twelve segments costs about 2.3 hours including scoring. So GPU time is not scarce and
nothing needs rationing.

The bake-off is gone for a separate reason: with one render per candidate, author effect and seed
effect are perfectly confounded, and the metrics that would rank them swing 8%-79% on the seed
alone. A tournament that cannot rank is not a tournament.

What survives is small and worth keeping: render, grade against gates that are near-binary and
near-deterministic, re-roll the seed on a real failure, and after three attempts keep the closest
rather than dropping the segment. That is this file.

THE GRADING RULES
-----------------
Hard gates, which can trigger a re-roll:
    G1  a third character on screen        identity_gate
    G2  the replaced character is back     identity_gate
    G3  passthrough, longest copied run    passthrough
    G7  mean cut offset                    h3_verify
    G8  the segment went flat              motion_stats

Never gated, only reported: G4, G5, G6, direction agreement, camera match. See config/grading.json
for why each one is untrustworthy. G4 in particular fires on 4-6 of 7 shots in EVERY arm including
the reference arm -- gating on it would drop all twelve segments and produce nothing. G5 has no
working positive control (2026-08-20). G6 was demoted 2026-08-24: it enforced a "torso covered
completely" rule that the character reference sheet itself contradicts.

After max_attempts, the best candidate is chosen by (hard failure count, G8 distance from source,
cut offset). Nothing is ever dropped, so a complete video exists at the end of every run.

SAFETY
------
Resumable: a candidate whose .mp4 already exists is not re-rendered, so an interrupted run
continues where it stopped. Every render goes through tools/gpu_claim.sh, which evicts Ollama's
vision model from VRAM first -- the lock alone is not enough and a render died out-of-memory
because of exactly that.

It refuses to start unless tools/gate_selftest.py passes, because a broken gate silently wastes a
whole run. That check is the one thing that would have caught G4.

USAGE
-----
    tools/render_pass.py --dry-run          # show every command, render nothing
    tools/render_pass.py                    # the real pass
    tools/render_pass.py --segments segA,seg06
    tools/render_pass.py --assemble-only    # rebuild final.mp4 from what already graded
"""
import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CFG = json.loads((ROOT / "config/grading.json").read_text())
ROSTERS = json.loads((ROOT / "config/board_rosters.json").read_text())["segments"]
SRC = ROOT / "output/fullbattle_20260808"
PY = str(ROOT / ".venv/bin/python3")

VESTA = "characters/char01_vesta/ref_vestia_v3.png"
ARCHER = "characters/<hero>/sheet.png"


def log(msg):
    print("[%s] %s" % (datetime.now().strftime("%H:%M:%S"), msg), flush=True)


def refs_for(seg):
    """Board plus only the characters who actually appear. Naming an absent character is what
    reintroduced the replaced character in the first place."""
    panels = ROSTERS[seg]["panels"]
    up = " ".join(panels).upper()
    layout = CFG["segments"][seg]["board_layout"]
    out = ["output/fullbattle_20260808/%s_board_merged_%s_PHOTOREAL.png" % (seg, layout)]
    if "THE WOMAN" in up or "BOTH" in up:
        out.append(VESTA)
    if "THE SWORDSMAN" in up or "BOTH" in up:
        out.append(ARCHER)
    return out


def render_argv(seg, seed, name, out_dir):
    r = CFG["render"]
    argv = [str(ROOT / "tools/gpu_claim.sh"), PY, "-u", str(ROOT / "tools/h3_local.py"),
            "--name", name,
            "--prompt-parts", "prompts/fullbattle_photoreal/%s_video_photoreal_novideo_parts.json" % seg,
            "--seconds", str(CFG["segments"][seg]["duration"]),
            "--seed", str(seed),
            "--width", str(r["width"]), "--height", str(r["height"]),
            "--steps", str(r["steps"]), "--ref-image-size", r["ref_image_size"],
            "--out-dir", str(out_dir)]
    for ref in refs_for(seg):
        argv += ["--ref-image", ref]
    return argv


def run(argv, **kw):
    return subprocess.run([str(a) for a in argv], cwd=ROOT, capture_output=True, text=True, **kw)


VIDEO_SUFFIXES = (".mp4", ".mov", ".webm", ".avi", ".mkv")


def video_reference_in(workflow_path):
    """G3, asked DIRECTLY. Returns [] for a clean job, a list of offenders for a dirty one, or
    None when it cannot tell -- which the caller must treat as UNMEASURED, never as clean.

    ★ WHY THIS REPLACED A PIXEL THRESHOLD, 2026-08-20.

    G3 exists to catch one thing: a video reference creeping back into the job. It used to infer
    that from how much of the output resembled the source, at longest_copy > 0.50s. On seg15 that
    fired at 2.50s and announced "a reference crept into the job" -- and no reference had. The job
    passed three images and no video, checked in the workflow itself.

    The threshold came from segA, whose source has mean motion 13.71. seg15's source is 6.39 and
    near-static: shifted against ITSELF by one second it reads 91% copied with a 3.50s run. Our
    render reproduces that same still composition because the storyboard is a restyle of those very
    frames. Two controls pinned it -- the same render against an UNRELATED source scores 0.00, so
    the metric discriminates fine; it is the threshold that was calibrated on one segment and armed
    on twelve. That is the G8 mistake exactly, for the third time in this repo.

    So stop inferring. The premise is a fact about the JOB, and the job is on disk. A LoadVideo node
    is present or it is not. No threshold, no per-segment calibration, no false positives on a quiet
    scene. The passthrough number is still measured and still reported -- it is real information --
    but it no longer gates.

    Positive control: eight historical workflows in output/fullbattle_20260808 carry
    LoadVideo.file=...mp4. gate_selftest.py checks this fires on one of them and stays quiet on a
    current one. A detector with no demonstrated true positive is how G5 got armed wrong.
    """
    try:
        wf = json.loads(Path(workflow_path).read_text())
    except Exception:
        return None
    if not isinstance(wf, dict):
        return None
    bad = []
    for node in wf.values():
        if not isinstance(node, dict):
            continue
        cls = node.get("class_type", "?")
        for k, v in (node.get("inputs") or {}).items():
            if isinstance(v, str) and v.lower().endswith(VIDEO_SUFFIXES):
                bad.append("%s.%s=%s" % (cls, k, v))
    return bad


def grade(seg, clip, work):
    """Every gate, on one candidate. Costs about 41 seconds.

    ★ EVERY gate must record an outcome. There is no path where a gate simply does not appear.
    The first version had `if json_exists:` with no else on G8, G3 and G7 -- a crashed scorer wrote
    no JSON, nothing was appended, and the candidate passed BY OMISSION. That is this project's
    signature failure (a gate that never ran being indistinguishable from a gate that passed),
    rebuilt inside the tool written to prevent it. Found by council review 2026-08-19.
    """
    seg_cfg = CFG["segments"][seg]
    g = CFG["hard_gates"]
    shots = SRC / ("shots_%s_merged_src.json" % seg)
    hard, report, metrics, measured = [], [], {}, {}

    def unmeasured(gate, why):
        """An unmeasured gate is a HARD failure in the worst tier. Never a pass."""
        hard.append("%s UNMEASURED -- %s" % (gate, why))
        measured[gate] = False

    # --- identity: G1 G2 hard; G4, G5, G6 report-only -----------------------------------
    ij = work / "identity.json"
    if ij.exists():
        ij.unlink()                      # never trust a stale file from an earlier attempt
    r = run([ROOT / "tools/gpu_claim.sh", PY, ROOT / "tools/identity_gate.py", clip,
             "--shots", shots, "--segment", seg, "--json", ij])
    if ij.exists():
        d = json.loads(ij.read_text())
        hard += [f["gate"] for f in d["hard_failures"]]
        for k in ("G1", "G2", "G5", "G6"):
            measured[k] = True
        if d["report_only"]:
            report.append("G4/G5/G6 x%d (never gated)" % len(d["report_only"]))
    else:
        for k in ("G1", "G2", "G5", "G6"):
            unmeasured(k, "identity_gate wrote no JSON: %s"
                       % (r.stderr.strip().splitlines()[-1:] or ["no stderr"])[0][:120])

    # --- G9: is the STORYBOARD ITSELF on screen? -----------------------------------------
    # Found by the owner's eye 2026-08-24 in a video every gate had passed: the last 0.46s of
    # seg20 IS the seg20 board -- panel grid, yellow numbers, empty grey cells. A rescan of all
    # 23 renders found THREE leaks (13%), one of them a whole 5.875s clip. Only luck kept the
    # other two out of the cut. This compares each frame against the exact image we fed the
    # renderer, so there is nothing to infer and nothing to calibrate.
    bj = work / "board.json"
    if bj.exists():
        bj.unlink()
    board = ROOT / refs_for(seg)[0]          # refs_for puts the board first, always
    # ★ --board-kind grid is stated, not left to auto-detect. Every segment in this pass renders
    # from a merged 2x2/3x3 board, and that is a fact about the job we already hold -- the same
    # move that fixed G3. Auto-detect stays for callers that genuinely do not know. Single-panel
    # boards need the 2026-08-25 stillness rule instead; see board_leak.py.
    r = run([PY, ROOT / "tools/board_leak.py", clip, "--board", board,
             "--board-kind", "grid", "--json", bj])
    if not bj.exists():
        unmeasured("G9", "board_leak wrote no JSON: %s"
                   % (r.stderr.strip().splitlines()[-1:] or ["no stderr"])[0][:120])
    else:
        d = json.loads(bj.read_text())
        if d.get("leak") is None:
            unmeasured("G9", d.get("why", "board_leak could not measure"))
        else:
            measured["G9"] = True
            metrics["board_peak"] = d.get("peak_similarity")
            raw = d.get("raw") or {}
            metrics["raw_board_peak"] = raw.get("peak_similarity")
            # leak_any covers the RAW source-frame board as well. A leak of the un-restyled board
            # is the worse of the two -- it is not our art on screen, it is the source footage.
            for label, res in (("THE STORYBOARD", d), ("THE RAW SOURCE-FRAME BOARD", raw)):
                if res.get("leak"):
                    sp = res["spans"][0]
                    hard.append("G9 %s IS ON SCREEN: %.2f-%.2fs of this clip "
                                "(%d frame(s), similarity %.3f)"
                                % (label, sp["start_s"], sp["end_s"], res["leak_frames"],
                                   res["peak_similarity"]))

    # --- motion: G8, against THIS segment's own source ------------------------------------
    mj = work / "motion.json"
    if mj.exists():
        mj.unlink()
    run([PY, ROOT / "tools/motion_stats.py", clip, "--json", mj,
         "--mean-floor", str(seg_cfg["g8_mean_floor"]),
         "--peak-floor", str(seg_cfg["g8_peak_floor"])])
    if mj.exists():
        d = json.loads(mj.read_text())
        metrics.update(motion_mean=d.get("motion_mean"), motion_max=d.get("motion_max"),
                       cuts=d.get("cuts"))
        g8 = d["g8"]
        measured["G8"] = True
        # shortfall against THIS segment's source, not segA's
        sm, sp = seg_cfg["source_motion_mean"], seg_cfg["source_motion_max"]
        mm, mx = d.get("motion_mean"), d.get("motion_max")
        metrics["g8_shortfall"] = round(max(0.0, (sm - mm) / sm) + max(0.0, (sp - mx) / sp), 4) \
            if (mm is not None and mx is not None) else None
        if g8.get("pass") is False:
            hard.append("G8 flat: " + "; ".join(g8["failures"]))
        elif g8.get("pass") is None:
            unmeasured("G8", "no motion could be measured")
    else:
        unmeasured("G8", "motion_stats wrote no JSON")

    ref_clip = SRC / ("%s_15x.mp4" % seg)          # SCORING input only, never a render input

    # --- G3: was a video reference in the job? Asked of the JOB, not of the pixels ---------
    wf_path = clip.with_name(clip.stem + "_workflow.json")
    offenders = video_reference_in(wf_path)
    if offenders is None:
        unmeasured("G3", "could not read the workflow at %s" % wf_path.name)
    else:
        measured["G3"] = True
        metrics["video_inputs"] = len(offenders)
        if offenders:
            hard.append("G3 a video reference IS in the job: %s" % "; ".join(offenders[:3]))

    # --- passthrough: measured and REPORTED, never gated (see video_reference_in) -----------
    pj = work / "copy.json"
    if pj.exists():
        pj.unlink()
    if not ref_clip.exists():
        report.append("passthrough not measured -- no source clip at %s" % ref_clip.name)
    else:
        run([PY, ROOT / "tools/passthrough.py", ref_clip, clip, "--json", pj])
        if not pj.exists():
            report.append("passthrough wrote no JSON (report only)")
        else:
            # passthrough.py returns {clip_path: {...}}, key longest_copy_SECONDS. Reading it as a
            # flat dict with "longest_copy_s" silently returned None and DISABLED G3.
            d = json.loads(pj.read_text())
            rec = next(iter(d.values())) if isinstance(d, dict) and d and \
                isinstance(next(iter(d.values())), dict) else d
            longest = rec.get("longest_copy_seconds")
            metrics["longest_copy_s"] = longest
            metrics["copy_fraction"] = rec.get("copy_fraction")
            if longest is None:
                report.append("passthrough not measured (report only)")
            elif longest > g["G3_longest_copy_s"]:
                report.append("passthrough %.2fs > %.2fs -- REPORT ONLY. On a near-static segment "
                              "this is the scene holding still, not a copy; G3 itself now checks "
                              "the job for a video input." % (longest, g["G3_longest_copy_s"]))

    # --- cut fidelity: G7 is REPORT ONLY (fires on 8 of 11 historical runs) ---------------
    vj = work / "verify.json"
    if vj.exists():
        vj.unlink()
    if ref_clip.exists():
        run([PY, ROOT / "tools/h3_verify.py", clip, "--ref-shots", shots,
             "--out-dir", work / "verify", "--json", vj])
        if vj.exists():
            d = json.loads(vj.read_text())
            off = d.get("cut_fidelity", {}).get("mean_offset")
            metrics["cut_offset_mean_s"] = off
            thr = CFG["report_only_thresholds"]["G7_cut_offset_mean_s"]
            if off is not None and off > thr:
                report.append("G7 cut offset %.3fs > %.2fs (report only)" % (off, thr))
            # h3_verify calls it "agreement". REPORT ONLY -- swings 8%-79% on the seed alone.
            if d.get("agreement") is not None:
                report.append("direction agreement %.2f (never gated)" % d["agreement"])
        else:
            report.append("G7 not measured (report only, so not a failure)")

    return {"hard_failures": hard, "report_only": report, "metrics": metrics,
            "measured": measured, "passed": not hard}


def tier_of(failure):
    """Which severity tier a failure text belongs to. Unknown/unmeasured -> the WORST tier."""
    t = CFG["severity"]["tiers"]
    head = failure.split()[0] if failure else ""
    if "UNMEASURED" in failure:
        return CFG["severity"]["_unmeasured_tier"]
    return t.get(head, CFG["severity"]["_unmeasured_tier"])


def severity_vector(failures):
    """Count of failures per tier, worst tier first. Compared LEXICOGRAPHICALLY, never summed.

    A sum inverts the owner's rule: additively three drawn shots score 3x50=150 against 100 for one
    wrong-person frame, so the sum PREFERS the clip with the replaced character in it. The owner said
    on 2026-08-19 that a brief wrong-character frame is worse than a soft beat. Tiers cannot be
    out-voted by volume. Caught by the cross-family council seat.
    """
    n = len(CFG["severity"]["_tier_names"])
    vec = [0] * n
    for f in failures:
        vec[min(tier_of(f), n - 1)] += 1
    return tuple(vec)


def rank_key(cand):
    """Lowest wins. Tiers first (lexicographic), then shortfall vs this segment's OWN source."""
    m = cand["metrics"]
    return (severity_vector(cand["hard_failures"]),
            m.get("g8_shortfall") if m.get("g8_shortfall") is not None else 9e9,
            m.get("cut_offset_mean_s") if m.get("cut_offset_mean_s") is not None else 9e9)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="print commands, render nothing")
    ap.add_argument("--segments", help="comma-separated subset; default is all twelve in order")
    ap.add_argument("--out-dir", default=None, help="default output/gauntlet_YYYYMMDD")
    ap.add_argument("--assemble-only", action="store_true")
    ap.add_argument("--seeds",
                    help="comma-separated seed ladder for THIS run, overriding config/grading.json. "
                         "Use it to retry a segment on a seed the standard ladder never reaches -- "
                         "with max_attempts=3 the fourth configured seed is otherwise unreachable.")
    ap.add_argument("--skip-selftest", action="store_true",
                    help="NOT RECOMMENDED. The self-test is what catches a broken gate.")
    args = ap.parse_args()

    segs = args.segments.split(",") if args.segments else CFG["_order"]
    if args.seeds:
        CFG["seeds"] = [int(x) for x in args.seeds.split(",")]
        log("seed ladder overridden for this run: %s" % CFG["seeds"])
    # .resolve() is load-bearing. Every clip path is later stored as clip.relative_to(ROOT), and
    # a RELATIVE --out-dir makes that raise ValueError the instant the first candidate is graded --
    # which is exactly when an unattended run has already spent a render. The default is built from
    # ROOT and is absolute, so this only ever bites a caller who passes the flag. It bit one on
    # 2026-08-20, 20 seconds into a twelve-segment pass.
    out_dir = Path(args.out_dir).resolve() if args.out_dir else \
        ROOT / ("output/gauntlet_%s" % datetime.now().strftime("%Y%m%d"))
    out_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = out_dir / "ledger.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {}

    if not (args.dry_run or args.assemble_only or args.skip_selftest):
        log("running gate self-test before touching the card")
        if subprocess.run([PY, str(ROOT / "tools/gate_selftest.py")], cwd=ROOT).returncode != 0:
            sys.exit("gate self-test FAILED -- fix the gate before rendering. "
                     "Override with --skip-selftest only if you know why it fails.")

    if not args.assemble_only:
        for seg in segs:
            if ledger.get(seg, {}).get("chosen"):
                log("%s already decided -- skipping" % seg)
                continue
            cands = []
            for attempt in range(CFG["max_attempts"]):
                seed = CFG["seeds"][attempt % len(CFG["seeds"])]
                name = "%s_s%d" % (seg, seed)
                clip = out_dir / (name + ".mp4")
                argv = render_argv(seg, seed, name, out_dir)

                if args.dry_run:
                    print("  " + " ".join(str(a) for a in argv))
                    break

                if clip.exists():
                    log("%s attempt %d: clip exists, re-using" % (seg, attempt + 1))
                else:
                    log("%s attempt %d (seed %d) rendering..." % (seg, attempt + 1, seed))
                    try:
                        r = run(argv, timeout=CFG.get("render_timeout_s", 3600))
                    except subprocess.TimeoutExpired:
                        log("   render TIMED OUT after %ss -- treating as a failed attempt"
                            % CFG.get("render_timeout_s", 3600))
                        continue
                    if r.returncode != 0 or not clip.exists():
                        log("   render FAILED: %s" % r.stderr.strip().splitlines()[-1:])
                        continue

                work = out_dir / ("%s_scores" % name)
                work.mkdir(exist_ok=True)
                v = grade(seg, clip, work)
                v.update(seed=seed, attempt=attempt + 1, clip=str(clip.relative_to(ROOT)))
                cands.append(v)
                if v["passed"]:
                    log("   PASS  %s" % v["metrics"])
                    break
                log("   fail: %s" % "; ".join(v["hard_failures"]))

            if args.dry_run:
                continue
            if not cands:
                ledger[seg] = {"chosen": None, "why": "every render failed to produce a clip"}
            else:
                best = sorted(cands, key=rank_key)[0]
                ledger[seg] = {
                    "chosen": best["clip"],
                    "passed": best["passed"],
                    "severity_tiers": list(severity_vector(best["hard_failures"])),
                    "hard_failures": best["hard_failures"],
                    "measured": best.get("measured", {}),
                    "why": "clean" if best["passed"] else
                           "closest after %d attempts: %s" % (len(cands),
                                                              "; ".join(best["hard_failures"])),
                    "metrics": best["metrics"],
                    "report_only": best["report_only"],
                    "attempts": [{"seed": c["seed"], "passed": c["passed"],
                                  "hard_failures": c["hard_failures"]} for c in cands]}
                log("%s -> %s (%s)" % (seg, Path(best["clip"]).name, ledger[seg]["why"]))
            ledger_path.write_text(json.dumps(ledger, indent=2))

    if args.dry_run:
        log("dry run: nothing rendered")
        return

    # --- assemble -------------------------------------------------------------------------
    # ★ ASSEMBLY ALWAYS COVERS THE WHOLE FILM, never the --segments subset.
    #
    # It used to use `segs`, so `--segments seg20` cheerfully printed "assembled 1/1 segments"
    # and OVERWROTE final.mp4 with a single 5.88s clip, destroying the finished twelve-segment
    # cut. Measured 2026-08-24, while fixing one segment. The duration assertion did not save it
    # either, because the expected total was computed from the same subset -- 5.88s against an
    # expected 5.87s reads as a perfect result.
    #
    # Re-rendering one segment is the NORMAL way to fix a film, so the normal outcome has to be
    # the complete film with that segment replaced. Anything still missing is reported below by
    # the never-drop check.
    asm = [x for x in CFG["_order"] if x in ledger or x in segs]
    if asm != segs:
        log("assembling the full film (%d segments), not just the %d rendered this run"
            % (len(asm), len(segs)))
    segs = asm
    missing = [x for x in segs if not ledger.get(x, {}).get("chosen")]
    if missing:
        log("★ NEVER-DROP VIOLATED: no clip for %s. final.mp4 would be INCOMPLETE."
            % ", ".join(missing))
    chosen = [ledger[x]["chosen"] for x in segs if ledger.get(x, {}).get("chosen")]
    if not chosen:
        log("nothing to assemble")
        return 1

    listfile = out_dir / "_concat.txt"
    listfile.write_text("".join("file '%s'\n" % (ROOT / c) for c in chosen))
    final = out_dir / "final.mp4"
    r = run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0",
             "-i", listfile, "-an", "-c", "copy", final])
    if r.returncode != 0 or not final.exists():
        log("★ ASSEMBLY FAILED: %s" % (r.stderr.strip()[-300:] or "ffmpeg wrote nothing"))
        return 1

    # Duration sanity. Silent truncation by concat has to be loud.
    want = sum(CFG["segments"][x]["duration"] for x in segs if ledger.get(x, {}).get("chosen"))
    pr = run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
              "-of", "csv=p=0", final])
    got = float(pr.stdout.strip()) if pr.stdout.strip() else 0.0
    log("assembled %d/%d segments -> %s  (%.2fs, expected about %.2fs)"
        % (len(chosen), len(segs), final, got, want))
    # Overshoot and undershoot are DIFFERENT faults and only one of them is a dropped segment.
    # h3_local rounds every render UP to the frame grid (align_frames rounds up), so a full pass
    # runs a little long by design -- 12 segments came in +3.04s on the first real pass, and the
    # old wording called that "something was dropped" and sent the reader looking for a missing
    # clip. Only a SHORTFALL can mean a drop.
    if got < want - 1.5:
        log("★ DURATION SHORTFALL: %.2fs against an expected %.2fs. A segment may have been "
            "dropped -- check the concat list." % (got, want))
    elif got > want + 1.5:
        log("note: %.2fs against a nominal %.2fs (+%.2fs). Expected: every render is rounded UP "
            "to the frame grid, so the pass runs slightly long. Not a drop."
            % (got, want, got - want))

    flagged = [x for x in segs if ledger.get(x, {}).get("chosen")
               and not ledger[x].get("passed")]
    # ★ THE SHIPPED clip, not every attempt. Measured 2026-08-20: this scanned
    # ledger[x]["attempts"], so a defect the retry ladder REJECTED still raised the banner. The
    # first full pass ended with "IDENTITY OR UNMEASURED DEFECT SHIPPED IN: seg2, seg09" when
    # neither shipped clip carried one -- seg2's G2 hit and seg09's UNMEASURED were both on
    # discarded attempts, which is the ladder working. The loudest safety signal in the pass was
    # announcing the one thing that had not happened, and a real identity defect would have been
    # indistinguishable from that noise.
    # severity_tiers is computed from the CHOSEN candidate, and tier 0 is exactly
    # "identity or unmeasured" -- see config/grading.json.
    identity = [x for x in flagged if (ledger[x].get("severity_tiers") or [0])[0] > 0]
    if flagged:
        log("SHIPPED WITH FLAGS: %s" % ", ".join(flagged))
    # The owner reads a report, not a ledger. Always write one.
    rr = run([PY, ROOT / "tools/run_report.py", out_dir, "--html"])
    for line in rr.stdout.strip().splitlines():
        log(line)
    if rr.returncode not in (0, 2):
        log("run_report failed: %s" % rr.stderr.strip()[-200:])

    if identity:
        log("")
        log("=" * 72)
        log("★★ IDENTITY OR UNMEASURED DEFECT SHIPPED IN: %s" % ", ".join(identity))
        log("   The owner ranks a brief wrong-character frame as the WORST defect.")
        log("   Watch these segments before publishing anything. See ledger.json.")
        log("=" * 72)
        return 2                      # nonzero, so an unattended caller cannot miss it
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)

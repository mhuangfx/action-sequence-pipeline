#!/usr/bin/env python3
"""Check every hard gate reproduces the PUBLISHED verdict table, before the loop starts.

WHY THIS EXISTS
---------------
G4 shipped as a hard gate and fired on 4-6 of 7 shots in EVERY arm, including the reference arm.
Had the loop been started with it armed, all twelve segments would have been dropped and the run
would have produced nothing. Nobody noticed because G4 had never actually run -- three of the twelve
segments were missing from config/board_rosters.json, so it raised KeyError instead of a verdict.

WHAT IT ACTUALLY CHECKS, AND WHY NOT "THE GATES SHOULD BE SILENT"
----------------------------------------------------------------
The obvious test -- point the gates at accepted footage and require silence -- is WRONG here, and
writing it that way produced two false alarms before this file settled. The owner accepted the noref
arm as a whole while knowing it carried real defects: seed 424242 has 2 drawn shots and 1 shot with
the replaced character in it. Both are printed in the verdict's own comparison table. A gate that
went silent on that clip would be BROKEN, not healthy.

So the fixture below is the verdict table itself, transcribed. The test is that each gate returns
the number that was published. That is strictly stronger than "stay quiet": it catches a gate that
has gone blind AND a gate that has started crying wolf, and it ties the code to a document a human
signed off.

Source of the G2 number: docs/style_ab_20260819_verdict.md, the "Verdict" table, cross-checked
against the noref_baseline block of config/gauntlet_incumbent.json.

★ THE G5 NUMBERS CHANGED ON 2026-08-20, and the authority for them changed with it. The verdict
table's "drawn shots" column (0 / 2 / 0) was produced by a gate whose answer turned out to be
decided by whether the vision model computed on the graphics card or the processor -- the same
frame, 4/4 "drawing" one way and 15/15 "photograph" the other. The owner then reviewed all 21
shots of all three clips by eye and confirmed every one has the proper style. So the expected
count is now 0 on all three, the owner's eye is the authority, and G5 is REPORT-ONLY: it is still
measured and still checked here, but it can no longer trigger a re-roll. It has no working
positive control -- it calls 6 of 7 frames of the ANIME SOURCE "rendered" -- and a gate that
cannot recognise the defect it guards must not spend an 11-minute render on a hunch.

IT HAS ALREADY CAUGHT ONE REGRESSION
------------------------------------
G6 was first added as a fifth key inside the calibrated identity question. That silently moved G5 on
seed 424242 from 2 drawn shots to 1, and suppressed the G2 hit at shot 5 entirely -- two real
defects hidden by adding an unrelated question. G6 now gets its own separate call, and both numbers
read correctly again. Do not merge new questions into QUESTION in identity_gate.py.

★ G6 IS REPORT-ONLY AS OF 2026-08-24, owner's decision. It demanded a torso "covered completely"
while the reference sheet handed to every render, characters/char01_vesta/ref_vestia_v3.png, shows
an open-front cuirass -- so it failed the takes that copied the reference faithfully. Every prompt
now describes the sheet's real costume and G6's question was rewritten to match. Its notes are
counted and PRINTED here, but they are not a problem to be fixed until the new question has been
calibrated against footage the owner has approved by eye. See _G6_note in config/grading.json.

Run this before every loop start, and after ANY change to a gate, a prompt or a threshold.

    tools/gate_selftest.py

Exit 0 = the gates reproduce the record and are safe to arm.
Exit 1 = a gate disagrees with the record. Find out which is wrong before rendering anything.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output/fullbattle_20260808"
SHOTS_SEGA = OUT / "shots_segA_merged_src.json"

# The verdict table, transcribed. Do not "tidy" these to zero -- see the docstring.
CASES = [
    {"clip": OUT / "ovn_noref_s770809.mp4", "segment": "segA", "shots": SHOTS_SEGA,
     "g5_cel": 0, "g2_replaced": 0, "g8_pass": True,
     "note": "clean run, mean 14.60 peak 27.16. Owner verified all 7 shots by eye 2026-08-20"},
    {"clip": OUT / "ovn_noref_s424242.mp4", "segment": "segA", "shots": SHOTS_SEGA,
     "g5_cel": 0, "g2_replaced": 1, "g8_pass": True,
     "note": "carries the known replaced-character shot. Owner verified all 7 shots have the "
             "proper STYLE by eye 2026-08-20 -- the old '2 drawn shots' was the broken G5"},
    {"clip": OUT / "ovn_noref_s131313.mp4", "segment": "segA", "shots": SHOTS_SEGA,
     "g5_cel": 0, "g2_replaced": 0, "g8_pass": False,
     "note": "THE FLAT SEED -- G8 must fail this one and only this one"},
]


def run(argv):
    return subprocess.run([str(a) for a in argv], cwd=ROOT, capture_output=True, text=True)


def identity(case, tmp):
    js = tmp / (case["clip"].stem + "_id.json")
    r = run([ROOT / "tools/gpu_claim.sh", ROOT / ".venv/bin/python3",
             ROOT / "tools/identity_gate.py", case["clip"],
             "--shots", case["shots"], "--segment", case["segment"], "--json", js])
    if not js.exists():
        # ★ Print WHY. The first version reported every no-JSON outcome as "it crashed", which hid
        # a placement guard doing its job and sent the reader hunting for a bug that was not there.
        tail = (r.stderr.strip() or r.stdout.strip() or "no output").splitlines()[-3:]
        print("     no JSON. Last output:")
        for line in tail:
            print("       %s" % line[:150])
        return None
    return json.loads(js.read_text())


def motion(case, tmp):
    """Uses the segment's CONFIGURED floors, not motion_stats' own defaults.

    The defaults were hard-coded from segA and render_pass never passed the config values through,
    so config/grading.json's G8 thresholds were dead config. Testing the defaults would have tested
    something the pass does not use.
    """
    cfg = json.loads((ROOT / "config/grading.json").read_text())["segments"][case["segment"]]
    js = tmp / (case["clip"].stem + "_motion.json")
    run([ROOT / ".venv/bin/python3", ROOT / "tools/motion_stats.py", case["clip"], "--json", js,
         "--mean-floor", str(cfg["g8_mean_floor"]), "--peak-floor", str(cfg["g8_peak_floor"])])
    return json.loads(js.read_text()) if js.exists() else None


G3_CASES = [
    # G3 asks the JOB, not the pixels, as of 2026-08-20. A detector with no demonstrated true
    # positive is how G5 got armed wrong, so both directions are fixtures here.
    (OUT / "cal_segA_photoreal_workflow.json", True,
     "a real historical job that DID carry a video reference"),
    (OUT / "genJ_seg1_directed_refmax_workflow.json", True,
     "another one, different prompt and settings"),
    (ROOT / "output/gauntlet_20260819/segA_s770809_workflow.json", False,
     "a current no-reference job -- must stay quiet"),
]


def check_g3(problems):
    import importlib.util
    spec = importlib.util.spec_from_file_location("rp", ROOT / "tools/render_pass.py")
    rp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rp)
    print("  G3 video-reference detector")
    for path, should_fire, note in G3_CASES:
        if not Path(path).exists():
            print("     ★ FIXTURE MISSING: %s" % Path(path).name)
            problems.append(("G3", "fixture missing: %s" % Path(path).name))
            continue
        r = rp.video_reference_in(path)
        fired = bool(r)
        ok = (r is not None) and fired == should_fire
        print("     %-9s %s -- %s" % ("FIRES" if fired else "quiet",
                                      "correct" if ok else "<<< DISAGREES", note))
        if not ok:
            problems.append(("G3", "%s: expected %s" % (Path(path).name,
                                                        "a hit" if should_fire else "silence")))
    if rp.video_reference_in(ROOT / "no_such_workflow.json") is not None:
        print("     ★ a missing workflow did NOT read as unmeasured")
        problems.append(("G3", "missing workflow must return None, not clean"))
    else:
        print("     None      correct   -- an unreadable job is UNMEASURED, never clean")
    print()


def check_g9(problems):
    """G9, both directions, against real fixtures. Delegates to the tool's own self-test, which
    proves it FINDS the known seg20 leak and stays quiet on a clean clip."""
    print("  G9 storyboard-leak detector")
    r = run([ROOT / ".venv/bin/python3", ROOT / "tools/board_leak.py", "--selftest"])
    for line in (r.stdout or "").strip().splitlines():
        if line.strip() and not line.startswith("G9 BOARD"):
            print("   %s" % line.strip()[:110])
    if r.returncode != 0:
        problems.append(("G9", "board_leak self-test failed"))
    print()


AB = ROOT / "output/trailer_test/ab_render"
AB_BOARDS = ROOT / "output/trailer_test/ab_boards"

# ★ THE SINGLE-BOARD CONTROLS, 2026-08-25. G9 was built for 3x3 grids on the premise that a good
# render never resembles a mosaic. One board per shot inverts it: a CORRECT render is supposed to
# match its board -- frame 0 of ab_render/S/t05.mp4 scores 0.9896 against s_t05.png and is a
# faithful take. Armed at 0.50 the grid rule would have failed 7 of the 10 A/B renders, so the
# 23-render pass would have re-rolled correct footage until the credits ran out.
#
# The leak fixture is BUILT here rather than committed: 0.6s of the board held as a still, spliced
# onto the tail of the real S/t05 render -- the exact shape of the seg20 defect, in the single-board
# world. A detector with no demonstrated true positive is how G5 got armed wrong.
SINGLE_LEAK_MIN_S = 0.30


def build_single_leak(tmp):
    """S/t05 with 0.6s of its own board frozen on the end. Returns the path, or None."""
    clip = AB / "S/t05.mp4"
    board = AB_BOARDS / "s_t05.png"
    if not (clip.exists() and board.exists()):
        return None
    still, a_ts, b_ts, out = (tmp / n for n in ("still.mp4", "a.ts", "b.ts", "single_leak.mp4"))
    dims = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                           "stream=width,height,r_frame_rate", "-of", "csv=p=0", str(clip)],
                          capture_output=True, text=True).stdout.strip().split(",")
    if len(dims) < 3:
        return None
    w, h, rate = dims[0], dims[1], dims[2]
    def ff(argv):
        return subprocess.run(["ffmpeg", "-v", "error", "-y"] + argv,
                              capture_output=True, text=True).returncode
    if ff(["-loop", "1", "-framerate", rate, "-t", "0.6", "-i", str(board),
           "-vf", "scale=%s:%s:force_original_aspect_ratio=decrease,"
                  "pad=%s:%s:(ow-iw)/2:(oh-ih)/2,setsar=1" % (w, h, w, h),
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(still)]):
        return None
    for src, dst in ((clip, a_ts), (still, b_ts)):
        if ff(["-i", str(src), "-c", "copy", "-bsf:v", "h264_mp4toannexb", "-f", "mpegts",
               str(dst)]):
            return None
    if ff(["-i", "concat:%s|%s" % (a_ts, b_ts), "-c", "copy", "-bsf:v", "h264_mp4toannexb",
           "-f", "mp4", str(out)]):
        return None
    return out if out.exists() else None


def check_g9_single(problems, tmp):
    """G9 on SINGLE-panel boards: a synthetic held still must fail, correct takes must pass."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("bl", ROOT / "tools/board_leak.py")
    bl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bl)

    print("  G9 single-panel boards (2026-08-25)")

    # The kind is read off the file. If that goes wrong the wrong RULE runs, so it is a fixture too.
    for path, want in ((AB_BOARDS / "s_t05.png", "single"),
                       (AB_BOARDS / "g_t05.png", "single"),
                       (AB_BOARDS / "grid_restyled.png", "grid"),
                       (ROOT / "output/fullbattle_20260808/seg20_board_merged_3x3_PHOTOREAL.png",
                        "grid"),
                       (ROOT / "output/fullbattle_20260808/seg09_board_merged_2x2_PHOTOREAL.png",
                        "grid")):
        if not path.exists():
            print("     ★ FIXTURE MISSING: %s" % path.name)
            problems.append(("G9", "board fixture missing: %s" % path.name))
            continue
        got = bl.detect_board_kind(path)
        ok = got == want
        print("     kind      %-38s -> %-6s %s" % (path.name, got,
                                                   "correct" if ok else "<<< DISAGREES"))
        if not ok:
            problems.append(("G9", "%s detected as %s, it is a %s board" % (path.name, got, want)))

    # POSITIVE: the board held as a still on the tail of a real render.
    leak = build_single_leak(tmp)
    if leak is None:
        print("     ★ could not build the synthetic single-board leak (ffmpeg or fixtures missing)")
        problems.append(("G9", "synthetic single-board leak could not be built"))
    else:
        r = bl.scan(leak, AB_BOARDS / "s_t05.png", board_kind="single")
        hit = bool(r and r["leak"] and r["spans"]
                   and (r["spans"][0]["end_s"] - r["spans"][0]["start_s"]) >= SINGLE_LEAK_MIN_S)
        print("     positive  s_t05 held as a still on the tail of S/t05 -- %s   %s"
              % (("%d frame(s) at %.3fs, picture %.4f vs floor %.4f"
                  % (r["leak_frames"], r["spans"][0]["start_s"], r["motion"]["quietest_window"],
                     r["motion"]["still_floor"])) if hit else "NOT FOUND",
                 "correct" if hit else "<<< DID NOT FIND A HELD STILL"))
        if not hit:
            problems.append(("G9", "missed the synthetic single-board leak"))

    # NEGATIVE: the two correct A/B takes. Both match their board almost perfectly and BOTH ARE
    # FINE -- this is the case the grid rule got wrong, so silence here is the whole point.
    for arm in ("S", "G"):
        clip = AB / arm / "t05.mp4"
        board = AB_BOARDS / ("%s_t05.png" % arm.lower())
        if not (clip.exists() and board.exists()):
            print("     ★ FIXTURE MISSING: %s/t05.mp4" % arm)
            problems.append(("G9", "A/B fixture missing: %s/t05.mp4" % arm))
            continue
        r = bl.scan(clip, board, board_kind="single")
        quiet = bool(r and not r["leak_any"])
        print("     negative  %s/t05, a correct take, board match %.4f   %s"
              % (arm, r["peak_similarity"], "quiet, correct" if quiet else "<<< FALSE POSITIVE"))
        if not quiet:
            problems.append(("G9", "fired on the correct %s/t05 take" % arm))
        # and the same clip under the OLD grid rule, to keep on the record why this changed
        rg = bl.scan(clip, board, board_kind="grid")
        print("        (the grid rule at 0.50 would fail it on %d of %d frames)"
              % (rg["leak_frames"], rg["frames"]))
    print()


def main():
    print("GATE SELF-TEST -- do the gates still reproduce the published verdict table?\n")
    problems = []
    check_g3(problems)
    check_g9(problems)

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        check_g9_single(problems, tmp)
        for case in CASES:
            name = case["clip"].stem
            print("  %s\n     %s" % (name, case["note"]))
            if not case["clip"].exists():
                # ★ NOT a skip. render_pass.py refuses to start unless this passes, so a silent
                # skip would make that guard hollow exactly when the fixtures are gone -- the
                # safety check itself failing silently. Missing fixture = failed self-test.
                print("     ★ FIXTURE MISSING: %s -- cannot verify the gates\n" % case["clip"])
                problems.append((name, "fixture clip is missing, gates unverified"))
                continue

            ident = identity(case, tmp)
            if ident is None:
                print("     identity_gate produced no JSON -- it crashed\n")
                problems.append((name, "identity_gate crashed"))
                continue

            hard = [f["gate"] for f in ident["hard_failures"]]
            counts = {"G2": sum(g.startswith("G2") for g in hard),
                      "G1": sum(g.startswith("G1") for g in hard)}
            # G6 is REPORT-ONLY as of 2026-08-24, so like G5 it never reaches hard_failures. Read
            # it off the report instead -- a demoted gate still has to be watched here, or it
            # drifts unnoticed and comes back armed on a number nobody re-verified.
            counts["G6"] = sum(1 for r in ident["report_only"]
                               if r["gate"].startswith("G6"))
            # G5 is REPORT-ONLY as of 2026-08-20, so it never reaches hard_failures. Read it from
            # the style answer instead -- a report-only gate still has to be checked, or it drifts
            # unnoticed and comes back armed on a number nobody re-verified.
            counts["G5"] = sum(1 for sh in ident["shots"]
                               if (sh.get("style") or {}).get("style") == "cel_anime")

            for gate, want, label in (("G5", case["g5_cel"], "cel-animation shots (REPORT ONLY)"),
                                      ("G2", case["g2_replaced"], "replaced-character shots")):
                got = counts[gate]
                ok = got == want
                print("     %-3s %d %s, verdict table says %d   %s"
                      % (gate, got, label, want, "correct" if ok else "<<< DISAGREES"))
                if not ok:
                    problems.append((name, "%s reads %d, table says %d" % (gate, got, want)))

            if counts["G1"]:
                print("     G1  FIRED %d time(s) on this clip -- not in the record"
                      % counts["G1"])
                problems.append((name, "G1 fired unexpectedly"))
            else:
                print("     G1  quiet   correct")

            # G6 is uncalibrated on purpose since 2026-08-24 -- the costume it used to demand is
            # not the costume the reference sheet shows. Its count is PRINTED, never a problem.
            print("     G6  %d note(s) on this clip (REPORT ONLY, uncalibrated since 2026-08-24)"
                  % counts["G6"])

            print("     G4/G5/G6  %d report-only note(s) in total, never counted"
                  % len(ident["report_only"]))

            m = motion(case, tmp)
            if not m:
                print("     G8  could not measure")
                problems.append((name, "motion_stats failed"))
            else:
                got, want = bool(m["g8"]["pass"]), case["g8_pass"]
                detail = "; ".join(m["g8"]["failures"]) or \
                    "mean %.2f peak %.2f" % (m["motion_mean"], m["motion_max"])
                print("     G8  %s, expected %s -- %s   %s"
                      % ("pass" if got else "FAIL", "pass" if want else "FAIL",
                         detail, "correct" if got == want else "<<< DISAGREES"))
                if got != want:
                    problems.append((name, "G8 verdict disagrees with the known outcome"))
            print()

    if problems:
        print("SELF-TEST FAILED -- a gate no longer matches the record:")
        for name, issue in problems:
            print("   %s: %s" % (name, issue))
        print("\nEither the gate drifted or the record is wrong. Settle which BEFORE rendering.\n"
              "Do not start a loop on gates that disagree with the document the owner signed off.")
        return 1

    print("SELF-TEST PASSED. Every gate reproduces the published verdict table, and G8 separates\n"
          "the flat seed from the two good ones. The gates are safe to arm.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

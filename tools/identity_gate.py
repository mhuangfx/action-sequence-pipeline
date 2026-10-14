#!/usr/bin/env python3
"""Check, shot by shot, that the right characters are on screen -- and only them.

★ G1 and G2 are the HARD gates here and they alone set the exit status. They are per-frame
properties and do not care which window a frame belongs to.

★ G5 (style) is REPORT-ONLY as of 2026-08-20 and G6 (costume) as of 2026-08-24 -- see the notes at
each of them, and the never_gate block of config/grading.json.

★ G4 IS NOT SOUND, and is report-only as of 2026-08-19. As of 2026-08-19 the exit status NO LONGER
COUNTS IT -- a G4-only clip now exits 0. Before that fix every caller that gated on the exit code
would have dropped all twelve segments, because G4 fires on 4-6 of 7 shots in every arm including
the reference arm. G4 compares roster panel i against
the midpoint of imposed reference window i, which assumes output shot i is reference shot i. H3 does
not honour prompted cut timings -- on segA the reference has 7 windows and the output has 10
detected shots. This is the same error h3_verify.py fixed with match_by_overlap on 2026-08-09; this
tool never got that fix. Measured over three arms, G4 fires on 4-6 of 7 shots in every arm including
the reference arm, with frame readings that are correct by eye. Fixing it needs design, not a patch:
overlap matching still leaves output shots with no panel, and panel ORDER preservation is untested.

WHY THIS EXISTS
---------------
This is the defect class nothing else catches. On 2026-08-18 a storyboard put the wrong character
in two of seven panels: the swordsman's own close-ups came back rendered as the heroine. Every
existing measure scored that board as fine. Cut offset does not look at who is in frame. Direction
agreement does not either. `passthrough.py` does not -- a correctly-composed frame with the wrong
person in it is not a copy of anything. It was caught by eye, and eyes do not scale to an
unattended 72-hour run.

It also enforces the owner's hard rule: these are 1-v-1 battles, so a THIRD character on screen in
any frame fails the segment outright. A stranger reads to a viewer as a new character, not as a
mistake.

G6 (costume) was listed as a hard gate from the start but had NO IMPLEMENTATION anywhere until
2026-08-19; the costume rule existed only inside the board prompt. It is a per-frame property like
G1/G2, so it lives here. It is REPORT-ONLY as of 2026-08-24: the rule it enforced ("torso covered
completely") contradicted the reference image the renderer is handed, so it fired on renders that
were faithfully copying that image. A vision gate that fires on footage we already accepted is
worse than no gate -- the whole lesson of G4 -- so it stays on the report until it is calibrated
against owner-approved footage with tools/gate_selftest.py.

THE TRAP
--------
`ollama run <prompt> <image>` SILENTLY DROPS THE IMAGE and answers from the prompt text alone --
confidently, and wrongly. It must go through the HTTP API with the image in the `images` array.
This has cost this project time before; do not "simplify" it back to the CLI.

REPRODUCIBLE
------------
Run twice on the same clip by accident on 2026-08-18, it returned byte-identical verdicts -- same
people counts, same character flags, same medium call on every shot. It runs at temperature 0, and
that determinism is the point: a gate that flips its verdict between runs is worse than no gate,
because the escalation ladder would burn renders chasing failures that are not there.

GPU
---
The vision model wants the same card as H3, which loads a 32B text encoder. They cannot both be
resident. Run this between renders, under the same lock:

    flock output/state/gpu.lock tools/identity_gate.py ...
"""
import argparse, base64, json, shutil, subprocess, sys, tempfile, time, urllib.request
from pathlib import Path

OLLAMA = "http://127.0.0.1:11434/api/generate"
MODEL = "qwen3-vl:8b-instruct-q8_0"

QUESTION = """Look at this single frame from a live-action fantasy fight scene and answer about what you can actually see.

Answer with JSON only, no other text, using exactly these keys:
{"people": <how many distinct human figures are visible, as an integer>,
 "red_haired_youth": <true if a YOUNG MAN with spiky red-orange hair wearing a WHITE, GREY or LIGHT-COLOURED shirt is visible, otherwise false>,
 "armoured_woman": <true if a WOMAN with long dark-red hair in dark armour is visible, otherwise false>,
 "white_haired_man": <true if a MAN with white or silver hair wearing a long DEEP-RED coat is visible, otherwise false>,
 "medium": <"photograph" if this looks like live-action photography, or "drawing" if it looks like anime, cartoon or illustrated art>}"""


# G6 is asked in its OWN call, deliberately.
#
# It was first added as a fifth key inside QUESTION above. That changed the model's answer to a
# DIFFERENT question: on ovn_noref_s424242 the calibrated prompt calls 2 of 7 shots "drawing"
# (matching the figure recorded in config/gauntlet_incumbent.json), and the five-key version called
# only 1. Adding a key perturbed G5. The docstring's promise of determinism is load-bearing -- a
# gate that shifts its verdict burns renders chasing failures that are not there -- so QUESTION is
# now frozen and anything new gets its own call.
COSTUME_QUESTION = """Look at this single frame and answer only about the WOMAN with long dark-red hair.

Her armour is an open-front blackened-iron cuirass: shaped chest plates rising to a high collar at the throat, the centre of the sternum and the midriff left as bare skin between the plates, plated pauldrons on both shoulders, forearm gauntlets, a runed hip belt, and long ragged charcoal-black skirt panels. Answer whether what she is wearing matches that.

Answer with JSON only, no other text, using exactly these keys:
{"costume_matches": <true if what she wears matches that description, false if she is wearing some other outfit>,
 "what_she_wears": "<a short phrase naming what covers her upper body>"}"""


def ask_costume(frame_path):
    """G6 only (REPORT-ONLY as of 2026-08-24). Asked separately so it cannot disturb QUESTION."""
    return _post(frame_path, COSTUME_QUESTION)


# G5 is asked in its OWN call as of 2026-08-20, and on a DIFFERENT question. QUESTION above is
# untouched and still frozen; its "medium" key is now recorded but never gated on.
#
# WHY. QUESTION offers the model two choices -- "live-action photography" or "anime, cartoon or
# illustrated art". Our renders are neither. They are painterly three-dimensional imagery, and
# forcing that into a binary it does not fit put the answer on a knife edge: the same frame came
# back "drawing" with the model wholly on the graphics card and "photograph" with it partly on the
# processor, both perfectly reproducible. A gate whose verdict turns on floating-point noise is not
# measuring the picture.
#
# The owner settled it on 2026-08-20 by eye, over all 21 shots of all three published clips: every
# one has the proper style. So the old reading was wrong on three of those frames, and the fix is
# not a threshold -- it is to ask about the defect we actually fear.
#
# That defect is concrete. The SOURCE material is 2D cel animation, and the failure is the render
# sliding back toward it. So this asks about the physical marks of cel animation -- ink outlines,
# flat colour fills, hard-edged shadow bands -- and explicitly says that dark, painterly or
# stylised imagery is not the defect. Calibrated against 21 owner-verified negatives AND a positive
# control, the anime source clip itself, which it must fire on. See tools/gate_selftest.py.
STYLE_QUESTION = """Look at this single frame and decide how the image was made.

There are only two possibilities:
  "cel_anime" - traditional 2D cel animation. Figures are drawn with hard dark outlines, large
                areas are filled with one flat colour, and shadows are two or three hard-edged
                bands rather than a smooth gradient.
  "rendered"  - a photographic or three-dimensionally rendered image. There are no drawn outlines
                around figures, colour varies continuously across surfaces, and light falls off
                smoothly.

A dark, painterly, grainy or heavily stylised image is still "rendered" as long as it has no drawn
outlines and its shading is continuous. Judge how the image was MADE, not how pretty or how
realistic it is.

Answer with JSON only, no other text, using exactly these keys:
{"style": <"cel_anime" or "rendered">,
 "ink_outlines": <true if the figures have hard drawn outlines around them, otherwise false>,
 "flat_colour_fills": <true if large areas are one uniform flat colour, otherwise false>,
 "why": "<a short phrase naming the strongest visual evidence>"}"""


def ask_style(frame_path):
    """G5 only. Its own call, so it cannot disturb the calibrated QUESTION."""
    return _post(frame_path, STYLE_QUESTION)


def _post(frame_path, question):
    with open(frame_path, "rb") as fh:
        img = base64.b64encode(fh.read()).decode()
    body = json.dumps({"model": MODEL, "prompt": question, "images": [img],
                       "stream": False, "format": "json",
                       "options": {"temperature": 0}}).encode()
    req = urllib.request.Request(OLLAMA, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        raw = json.load(r)["response"]
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"_unparsed": raw[:200]}


def assert_model_on_gpu():
    """★ THE GATE'S ANSWER DEPENDS ON WHERE THE MODEL COMPUTES. Verify, do not assume.

    Measured 2026-08-19, and it cost most of a session. On shot 7 of ovn_noref_s770809 -- a frame
    that is byte-identical between two clips, checked with framemd5 -- this gate returns:

        model fully on the card    "drawing"     4 runs out of 4
        model mostly on the CPU    "photograph"  15 runs out of 15

    Both readings are internally perfectly reproducible. The flip is not temperature (which is 0)
    and not the seed. It is that ollama fixes its GPU/CPU layer split AT LOAD TIME from whatever
    VRAM happens to be free, and a different split means different floating-point numerics, which
    flips a genuinely borderline token. ComfyUI idles holding 6.5GB and holds 13.8GB just after a
    render, so the split -- and therefore the VERDICT -- depended on what else had run first.

    That is this project's signature bug in its purest form: an input nobody was looking at,
    silently deciding a gate's answer. tools/gpu_claim.sh now empties the card before the gate, so
    the model always loads whole. This function CHECKS that it actually did, because "we asked for
    it" is not evidence, and a wrong-but-confident verdict is worse than a loud failure.
    """
    try:
        with urllib.request.urlopen("http://127.0.0.1:11434/api/ps", timeout=30) as r:
            models = json.load(r).get("models", [])
    except Exception as e:
        return "could not read ollama /api/ps to verify model placement: %s" % e
    for m in models:
        if m.get("model") != MODEL and m.get("name") != MODEL:
            continue
        total, vram = m.get("size", 0), m.get("size_vram", 0)
        if not total:
            return "ollama reported no size for %s" % MODEL
        if vram < total * 0.95:
            return ("%s is only %.1f of %.1f GB on the card -- %.1f GB is on the CPU. Different "
                    "placement gives DIFFERENT VERDICTS on borderline frames, so this reading "
                    "would not be comparable to the calibration. Free the card first "
                    "(tools/gpu_claim.sh does this) and re-run."
                    % (MODEL, vram / 2**30, total / 2**30, (total - vram) / 2**30))
        return None
    return "%s is not resident after a call -- cannot verify where it computed" % MODEL


def evict_model():
    """Drop the vision model out of VRAM so the next call reloads it onto a drained card."""
    body = json.dumps({"model": MODEL, "keep_alive": 0}).encode()
    req = urllib.request.Request(OLLAMA, data=body, headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=60).read()
    except Exception:
        pass


def ensure_on_gpu_or_die(frame_path, first_answer, allow_split, tries=3):
    """Verify the model computed wholly on the card, and RECOVER if it did not.

    ★ Aborting here was wrong, and it cost a self-test run on 2026-08-20. Ollama fixes its
    GPU/CPU split at load time, so a model that loaded while ComfyUI still held the card stays
    split for its whole life -- and the first version of this check called sys.exit(), which
    gate_selftest could only report as "identity_gate crashed". A recoverable condition was being
    treated as fatal, and its reason was being thrown away.

    Evicting the model and calling again reloads it onto a card that has since drained, which is
    the actual fix. Only after several failed reloads is the card genuinely too full to trust, and
    THEN refusing is right -- a verdict from a split model is not comparable to the calibration.
    """
    why = assert_model_on_gpu()
    for attempt in range(tries):
        if not why:
            return first_answer
        print("   placement not verified (try %d/%d): %s" % (attempt + 1, tries, why))
        evict_model()
        time.sleep(8)
        first_answer = _post(frame_path, QUESTION)
        why = assert_model_on_gpu()
    if allow_split:
        print("★ WARNING, grading with a split model because --allow-split-model was given")
        return first_answer
    sys.exit("MODEL PLACEMENT UNVERIFIED after %d reloads -- refusing to grade.\n  %s\n"
             "Something else is holding the card. Free it and re-run. Override with "
             "--allow-split-model only for a throwaway measurement, never for a verdict."
             % (tries, why))


def ask(frame_path):
    return _post(frame_path, QUESTION)


def frame_at(clip, t, dest):
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", "%.3f" % t, "-i", str(clip),
                    "-frames:v", "1", "-q:v", "2", str(dest)], check=True)


def expected_from_roster(line):
    """Turn a roster line into who SHOULD be there. Roster lines start with the cast, in caps."""
    up = line.upper()
    if up.startswith("BOTH"):                return {"woman": True,  "swordsman": True}
    if up.startswith("THE WOMAN ALONE"):     return {"woman": True,  "swordsman": False}
    if up.startswith("THE SWORDSMAN ALONE"): return {"woman": False, "swordsman": True}
    # "THE SCENE ALONE" and "NEITHER CHARACTER" both mean a panel with no fighter in it -- an
    # establishing plate or an abstract inner-world insert. Both phrasings are in use across the
    # twelve rosters, and returning None for either one silently skipped G4 on that panel.
    if up.startswith("THE SCENE ALONE"):     return {"woman": False, "swordsman": False}
    if up.startswith("NEITHER CHARACTER"):   return {"woman": False, "swordsman": False}
    return None


def g4_notes(roster_line, a):
    """G4, REPORT-ONLY. Kept in its own function so it can be read on ONE frame while the hard
    gates read several -- its note count then does not move when the sample rate does."""
    want = expected_from_roster(roster_line)
    if not want:
        return []
    out = []
    if want["woman"] and not a.get("armoured_woman"):
        out.append("G4 the woman is missing from a shot that is hers")
    if not want["woman"] and a.get("armoured_woman"):
        out.append("G4 the woman appears in a shot that is not hers")
    if want["swordsman"] and not a.get("white_haired_man"):
        out.append("G4 the swordsman is missing from a shot that is his")
    if not want["swordsman"] and a.get("white_haired_man"):
        out.append("G4 the swordsman appears in a shot that is not his")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip")
    ap.add_argument("--shots", required=True, help="the segment's merged shot list")
    ap.add_argument("--segment",
                    help="segment name, to check against config/board_rosters.json. NOTE: this "
                         "enables G4, which is REPORT-ONLY as of 2026-08-19 -- it pairs reference "
                         "window i with output shot i, and H3 does not honour prompted cut timings, "
                         "so the two are not the same units. See the module docstring.")
    ap.add_argument("--rosters", default="config/board_rosters.json")
    ap.add_argument("--json", help="write the per-shot findings here")
    ap.add_argument("--frames-per-shot", type=int, default=1, metavar="N",
                    help="how many frames to read per shot. MUST BE ODD, so the centre sample is "
                         "exactly the single sample a 1-frame run takes and any new finding "
                         "provably comes from a NEW frame, never from a changed reading of the old "
                         "one. Default 1 -- the sample rate the published verdict table was "
                         "measured at. Raising it changes the counts, so re-verify by eye and "
                         "re-baseline tools/gate_selftest.py before arming it.")
    ap.add_argument("--dump-frames", metavar="DIR",
                    help="save every frame that trips a gate here, for a human to judge by eye")
    ap.add_argument("--allow-split-model", action="store_true",
                    help="NOT FOR VERDICTS. Grade even when the vision model is split across the "
                         "card and the CPU. The split changes the answers on borderline frames.")
    args = ap.parse_args()
    n_per = args.frames_per_shot
    if n_per < 1 or n_per % 2 == 0:
        sys.exit("--frames-per-shot must be odd and >= 1 (got %d). An even count has no centre "
                 "sample, so the result would not be comparable to the calibrated 1-frame run."
                 % n_per)

    shots = json.loads(Path(args.shots).read_text())
    shots = shots["shots"] if isinstance(shots, dict) else shots
    roster = None
    if args.segment:
        roster = json.load(open(args.rosters))["segments"][args.segment]["panels"]
        if len(roster) != len(shots):
            sys.exit("roster has %d panels but the shot list has %d" % (len(roster), len(shots)))

    findings, failures, reported = [], [], []
    checked_placement = [False]
    keep = Path(args.dump_frames) if args.dump_frames else None
    if keep:
        keep.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        for i, s in enumerate(shots):
            span = s["end"] - s["start"]
            times = [s["start"] + span * (k + 0.5) / n_per for k in range(n_per)]
            centre = n_per // 2      # odd n_per => this is EXACTLY the old single 0.5 sample
            frames, bad, report = [], {}, []
            # `bad` maps gate-text -> the time it was FIRST seen, so a shot contributes at most ONE
            # entry per gate however many frames flag it. That keeps the counts in SHOT units,
            # which is the unit the published verdict table -- and gate_selftest -- are written in.
            for k, t in enumerate(times):
                f = Path(td) / ("s%02d_f%d.jpg" % (i + 1, k + 1))
                frame_at(args.clip, t, f)
                a = ask(f)
                if not checked_placement[0]:
                    checked_placement[0] = True
                    a = ensure_on_gpu_or_die(f, a, args.allow_split_model)
                    rec = {"t": round(t, 2), **a}
                rec = {"t": round(t, 2), **a}
                hits = []

                # ★ A reply we could not parse must FAIL LOUDLY. Before this check it returned
                # {"_unparsed": ...}, and every a.get() below then defaulted to 0/None -- so the
                # frame silently CLEARED G1, G2, G5 and G6 at once. An unreadable answer is not a
                # clean frame, and with several frames a shot it is still not one: that frame was
                # not measured, and unmeasured is never a pass in this project.
                if "_unparsed" in a or "medium" not in a:
                    hits.append("G0 UNMEASURED -- the vision model returned an unreadable answer")
                else:
                    if a.get("people", 0) > 2:
                        hits.append("G1 third character: %d people visible" % a["people"])
                    if a.get("red_haired_youth"):
                        hits.append("G2 the replaced character is on screen")
                    # ★ G5 IS REPORT-ONLY as of 2026-08-20. Owner's call, and the reason is a
                    # measured limitation of the vision model, not a threshold that needs tuning.
                    #
                    # Asked as "live-action photography or anime/cartoon", the model put the SAME
                    # frame on either side depending only on whether it computed wholly on the
                    # graphics card or partly on the processor -- both answers perfectly
                    # reproducible. Re-asked about the physical marks of cel animation instead
                    # (ink outlines, flat colour fills, hard shadow bands) it is clean on all 21
                    # owner-verified frames -- but it also calls SIX OF SEVEN FRAMES OF THE ANIME
                    # SOURCE ITSELF "rendered". So it has no usable positive control: it cannot
                    # reliably recognise the very material this gate exists to detect a slide back
                    # toward. A gate with no demonstrated true positive cannot be armed.
                    #
                    # Still measured, still on the report, still ranked on. Style is a judgement
                    # for the owner's eye at review until a detector that passes a positive
                    # control exists.
                    st = ask_style(f)
                    rec["style"] = st
                    if "_unparsed" in st or "style" not in st:
                        report.append("G5 style unreadable (report only)")
                    elif st.get("style") == "cel_anime":
                        report.append("G5 looks like 2D cel animation: %s (report only)"
                                      % st.get("why", "?"))
                    # ★ G6 IS REPORT-ONLY as of 2026-08-24. Owner's decision: the character
                    # reference sheet characters/char01_vesta/ref_vestia_v3.png is the truth, and
                    # it shows an OPEN-FRONT cuirass. The old G6 asked whether her torso was
                    # covered completely -- a rule that came from the prompt text, never from the
                    # picture the renderer was actually given. So the gate was failing renders for
                    # faithfully copying the reference: six costume breaks, and on seg20 it pushed
                    # the pass into keeping the take that carried the storyboard leak instead.
                    #
                    # The question now describes the real costume and asks whether the render
                    # matches it. It is still measured, still on the report, still ranked on -- but
                    # it cannot burn an 11-minute re-roll until it has been calibrated against
                    # footage the owner has approved by eye.
                    #
                    # G6 only matters when she is actually on screen, and a second call on a frame
                    # without her would only invite a hallucinated answer.
                    if a.get("armoured_woman"):
                        costume = ask_costume(f)
                        rec["costume"] = costume
                        if "_unparsed" in costume or "costume_matches" not in costume:
                            report.append("G6 costume unreadable (report only)")
                        elif not costume.get("costume_matches"):
                            report.append("G6 costume differs from the reference sheet: %s "
                                          "(report only)"
                                          % costume.get("what_she_wears", "?"))
                    # G4 is REPORT-ONLY and unsound. Read it on the CENTRE frame ONLY, so its note
                    # count does not move when the sample rate does. One less thing changing.
                    if roster and k == centre:
                        report += g4_notes(roster[i], a)

                rec["failures"] = hits
                frames.append(rec)
                for h in hits:
                    if h not in bad:
                        bad[h] = round(t, 2)
                        if keep:
                            shutil.copy(f, keep / ("%s_shot%02d_t%05.2f_%s.jpg"
                                                   % (Path(args.clip).stem, i + 1, t,
                                                      h.split()[0])))
                if n_per > 1:
                    print("     frame %d/%d @%5.2fs  people=%-2s youth=%-5s woman=%-5s "
                          "sword=%-5s %-11s %s"
                          % (k + 1, n_per, t, a.get("people", "?"), a.get("red_haired_youth"),
                             a.get("armoured_woman"), a.get("white_haired_man"),
                             a.get("medium", "?"), "; ".join(hits) or "ok"))

            mid = frames[centre]
            rec = {"shot": i + 1, "t": round(times[centre], 2), "frames": frames,
                   "failures": list(bad), "report_only": report,
                   "frames_sampled": n_per}
            # The centre frame's flat keys stay at the top level so anything reading the old shape
            # still works, and so the centre reading is directly comparable to a 1-frame run.
            rec.update({k: v for k, v in mid.items() if k not in ("failures", "t")})
            findings.append(rec)
            failures += [(i + 1, b, bad[b]) for b in bad]
            reported += [(i + 1, r) for r in report]
            note = "; ".join("%s @%.2fs" % (b, bad[b]) for b in bad) if bad else \
                   ("ok" + ("  [report-only: %d]" % len(report) if report else ""))
            print("shot %2d @%5.2fs  people=%-2s youth=%-5s woman=%-5s sword=%-5s fit=%-5s "
                  "%-11s %s"
                  % (i + 1, times[centre], mid.get("people", "?"), mid.get("red_haired_youth"),
                     mid.get("armoured_woman"), mid.get("white_haired_man"),
                     (mid.get("costume") or {}).get("costume_matches"),
                     mid.get("medium", "?"), note))
    if args.json:
        Path(args.json).write_text(json.dumps(
            {"clip": str(args.clip), "segment": args.segment, "frames_per_shot": n_per,
             "hard_failures": [{"shot": n, "gate": g, "t": t} for n, g, t in failures],
             "report_only": [{"shot": n, "gate": g} for n, g in reported],
             "shots": findings}, indent=2))

    if reported:
        print("\nREPORT-ONLY notes (G4, G5, G6 -- not failures): %d note(s) across %d shot(s). "
              "G4 pairs reference window i with output shot i and H3 does not honour prompted cut "
              "timings, so it fires on good footage too. G5 has no working positive control. G6 is "
              "uncalibrated since the 2026-08-24 costume decision. Put them on the review page; do "
              "not re-roll on them." % (len(reported), len({s for s, _ in reported})))
    if failures:
        print("\nFAILED: %d HARD issue(s) across %d shot(s)"
              % (len(failures), len({s for s, _, _ in failures})))
        sys.exit(1)
    print("\nPASSED: %d shots, no hard-gate failure" % len(shots))


if __name__ == "__main__":
    main()

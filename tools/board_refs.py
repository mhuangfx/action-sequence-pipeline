#!/usr/bin/env python3
"""Which reference sheets a trailer shot gets, and the <Picture N> preamble that binds them.

WHY THIS EXISTS
---------------
Two defects from the 2026-08-25 board A/B, both caused by the same thing: the rule for "which
sheets does this shot get" lived in whatever script happened to be running, so the board driver and
the render driver disagreed with each other and with the prompts.

  1. output/trailer_test/boards/_runs/run_boards.py attached the hero sheet
     characters/char01_vesta/ref_vestia_v3.png to EVERY shot, unconditionally. t02 is typed
     LANDSCAPE -- "the composition holds land, sky and falling ash alone" -- and the hero is in the
     board anyway. Handing a model a character sheet is an assertion that the character is in
     frame; that is the same mechanism that reintroduced the replaced character in August, and
     render_pass.refs_for() already carried the fix for the fullbattle segments. The trailer just
     never inherited it.

  2. The blade sheet characters/char01_vesta/ref_vesta_blades_v1.png went to 14 of the 23 shots.
     The A/B measured what that costs: EVERY hero shot carrying it framed wide to show both
     blades, and t05 -- a shot whose whole point is one hand raised open-palmed -- came back with a
     blade in the open hand. So the blade sheet is not a costume accessory to be handed out with
     the hero sheet. It goes only where the blade is the SUBJECT of the composition.

THE RULE
--------
    HERO, HERO+ARMY                       hero sheet
    LANDSCAPE, OBJECT, VFX, DRAGON, ARMY  no hero sheet -- nobody named who is not in frame
    blade is the shot's subject           blade sheet, and only then

BLADE_SUBJECT is written down rather than inferred, because "the prompt mentions a blade" is not
the question -- eleven prompts mention the blades as part of the costume paragraph. The question is
whether the blade is what the frame is ABOUT, and that is read off the composition sentence:

    t06  "framed on a pair of gloved hands closing around the twin hilts ... the composition is
         cropped at the forearms and holds the hands and the hilts alone"      -- OBJECT
    t10  the blade insert. Cut from the 96s edit (render_dur 0), kept here so the rule survives if
         it comes back.
    t11  THE STRIKE. "she kneels ... with one blade driven upright into the ground in front of
         her", and its own note calls it "Parent of the t10 insert: the blade-into-the-ash moment
         must land cleanly inside this take". This is the shot the insert was cut into.

t12 is deliberately absent even though a blade made the crack: its note says "No character and no
character sheet in this render", and the blade is not in frame -- only the fracture it left.

THE <Picture N> PREAMBLE
------------------------
trailer_prompts.json carries no reference tags at all. h3_local passes --ref-image in <Picture N>
order, so an untagged prompt leaves every attached sheet dangling -- the model is handed images and
never told what they are for. The A/B worked around it by prepending three fixed lines, which is
wrong the moment a shot's ref list is not exactly [board, hero, blade]: on t06 the blade is
<Picture 2>, and a hard-coded "<Picture 3>" would point at nothing.

So the preamble is DERIVED from the list actually attached, in the order it is attached. Fix the
refs and the preamble follows. It is built here, in the render path, and never written back into
trailer_prompts.json -- the JSON stays the creative document, not a copy of the plumbing.

USAGE
    from board_refs import refs_for_board, refs_for_render, bind_preamble, render_prompt
    tools/board_refs.py --table          # print the whole roster and eyeball it
"""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROMPTS_JSON = ROOT / "output/trailer_test/trailer_prompts.json"

HERO_SHEET = "characters/char01_vesta/ref_vestia_v3.png"
BLADE_SHEET = "characters/char01_vesta/ref_vesta_blades_v1.png"

HERO_TYPES = {"HERO", "HERO+ARMY"}
BLADE_SUBJECT = {"t06", "t10", "t11"}          # see the docstring -- read off the composition line

# The binding lines, by the role a reference plays. %d is filled with its real 1-based position in
# the --ref-image list, which is what h3_local turns into <Picture N>.
BIND = {
    "board": "本片的构图、镜头角度、取景范围、人物在画面中的大小与位置、以及场景一律以分镜参考图 <Picture %d> 为准。",
    "hero": "女主角的面容、发型、肤色与战甲以 <Picture %d> 为准。",
    "blade": "她手持的一对短弯刀以 <Picture %d> 为准。",
}

_CACHE = {}


def shots(prompts_json=PROMPTS_JSON):
    key = str(prompts_json)
    if key not in _CACHE:
        _CACHE[key] = json.loads(Path(prompts_json).read_text())["shots"]
    return _CACHE[key]


def shot_type(shot, prompts_json=PROMPTS_JSON):
    return shots(prompts_json)[shot]["shot_type"]


def refs_for_board(shot, prompts_json=PROMPTS_JSON):
    """The character sheets that go to BOARD generation, as structure refs.

    Mirrors render_pass.refs_for(): only the characters the shot's own prompt puts in frame. The
    board itself is what this produces, so it is not in the list.
    """
    out = []
    if shot_type(shot, prompts_json) in HERO_TYPES:
        out.append(HERO_SHEET)
    if shot in BLADE_SUBJECT:
        out.append(BLADE_SHEET)
    return out


def refs_for_render(shot, board, prompts_json=PROMPTS_JSON):
    """The full --ref-image list for the H3 render: board FIRST, then the same sheets.

    Board first is load-bearing -- render_pass reads refs_for(seg)[0] as the board to scan for G9.
    """
    return [str(board)] + refs_for_board(shot, prompts_json)


def role_of(ref):
    name = Path(ref).name.lower()
    if "blade" in name:
        return "blade"
    if Path(ref).as_posix().startswith("characters/") or "ref_vestia" in name:
        return "hero"
    return "board"


def bind_preamble(refs):
    """The <Picture N> binding lines for exactly this ref list, in attach order.

    Derived, never hard-coded: a shot that gets [board, blade] must bind the blade to <Picture 2>.
    """
    lines = []
    for i, ref in enumerate(refs, start=1):
        tmpl = BIND.get(role_of(ref))
        if tmpl:
            lines.append(tmpl % i)
    return "".join(lines)


def render_prompt(shot, refs, prompts_json=PROMPTS_JSON):
    """The prompt text as it should reach h3_local: preamble + the Chinese body.

    The English gloss after "EN:" is a note to the reader, not to the model, and is dropped -- the
    settled finding is that Chinese prompts are the default.
    """
    body = shots(prompts_json)[shot]["h3_prompt"]
    body = re.split(r"\nEN:", body)[0].strip()
    return bind_preamble(refs) + body


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", action="store_true", help="print every shot's refs and check them")
    ap.add_argument("--prompts", default=str(PROMPTS_JSON))
    args = ap.parse_args()
    if not args.table:
        return ap.error("nothing to do -- try --table")

    print("%-5s %-11s %-6s %-6s %s" % ("shot", "type", "hero", "blade", "board-gen refs"))
    for shot, v in shots(args.prompts).items():
        r = refs_for_board(shot, args.prompts)
        cut = " (cut from the 96s edit)" if not v.get("render_dur") else ""
        print("%-5s %-11s %-6s %-6s %s%s"
              % (shot, v["shot_type"], "yes" if HERO_SHEET in r else "-",
                 "yes" if BLADE_SHEET in r else "-",
                 ", ".join(Path(x).name for x in r) or "none", cut))
    print("\npreamble check, t06 (board + blade, no hero -- the blade must be <Picture 2>):")
    print("  " + bind_preamble(refs_for_render("t06", "output/trailer_test/boards/t06.png",
                                               args.prompts)))

    # The two defects the A/B actually found, as assertions. A rule nobody runs is how G4 shipped
    # broken; these cost nothing and they fail loudly if the old behaviour comes back.
    bad = []
    if HERO_SHEET in refs_for_board("t02", args.prompts):
        bad.append("t02 is LANDSCAPE and must not carry the hero sheet")
    if BLADE_SHEET in refs_for_board("t05", args.prompts):
        bad.append("t05 is a raised-open-hand shot and must not carry the blade sheet")
    if BLADE_SHEET not in refs_for_board("t06", args.prompts):
        bad.append("t06 IS the blade shot and must carry the blade sheet")
    if "<Picture 2>" not in bind_preamble(["board.png", BLADE_SHEET]):
        bad.append("the preamble is not deriving positions from the ref list")
    print()
    if bad:
        for b in bad:
            print("  ★ %s" % b)
        return 1
    print("  rule checks pass: no hero sheet on t02, no blade sheet on t05, blade sheet on t06,\n"
          "  and the preamble numbers itself off the list it was given.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

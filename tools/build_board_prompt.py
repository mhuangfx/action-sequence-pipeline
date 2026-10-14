#!/usr/bin/env python3
"""Assemble a board-restyle prompt for one segment: shared blocks + its roster + its setting.

Everything except the roster and the setting is identical across segments. That is deliberate --
the roster is the part that has to be written per segment, because it is the part that makes the
restyle work, and the setting is the part that changes when the battle moves location.

Positive phrasing only. On this model family a prohibition tends to summon what it forbids: a
prompt containing "No moon" put a moon in three of four panels.

COSTUME, 2026-08-24 (owner's decision): characters/char01_vesta/ref_vestia_v3.png IS the truth, and
the prompt now describes what that sheet actually shows. The previous clause asked for a torso
"covered completely" while the reference image handed to the same render showed an open-front
cuirass -- prompt and picture pulled against each other and the costume broke six times. Keep this
paragraph matched to the sheet; if the sheet changes, this changes with it.
"""
import argparse, json, re, sys

ROSTERS = 'config/board_rosters.json'

TEMPLATE = """Restyle this {cells}-cell storyboard grid, keeping the {grid} layout, the white panel borders and the yellow panel numbers. Any empty cell is flat grey and stays empty flat grey. Preserve each panel's exact composition, camera angle, framing, staging and body pose.

There are exactly two characters in this story: THE WOMAN and THE SWORDSMAN. Which of them appears in which panel is fixed:

{roster}

THE WOMAN replaces the red-haired young man in the white and blue shirt, in exactly his pose, including the airborne and foreshortened angles. She is taken from her character reference sheet. She wears her base combat regalia exactly as her reference sheet shows it, identical in every panel: an open-front blackened-iron cuirass -- sculpted contoured chest plates that follow the bust and rise to a high runed collar at the throat, with the centre of the sternum and the midriff left as bare skin between the plates; layered blackened-iron pauldrons on both shoulders; bare upper arms; segmented forearm gauntlets with plated fingerless gloves; a wide runed hip belt with a round ember cabochon at the front; and long ragged layered skirt panels of charcoal-black cloth that fall to the ankles and part down the front over bare legs, with plated shin guards and dark boots. The blackened iron is etched with fine runic channels that glow molten ember-orange deep in the cracks, and ember-crystal points glow at the throat, the sternum and the navel. Deep crimson hair worn long, loose and wind-blown past the shoulder blades, warm light-tan skin with freckles, molten amber-gold eyes that give off their own light, glowing ember runic glyphs across her forehead and both cheekbones. She wields a pair of short curved blades with ember-lit edges. Her power stays restrained: the dormant rune glow, the ember crystals, and a few ember motes.

THE SWORDSMAN is the white-haired man in the long deep-red coat. He stays himself in every panel he appears in. He is taken from his own character reference sheet exactly as he appears there -- same face, same white upswept hair, same deep red coat, same silver harness and emblem, same paired black-and-white curved blades. His presence reads as steel and cloth, lit by the room around him.

Every blue and cyan energy effect becomes molten ember-orange instead.

Both characters and the whole scene are rendered in one identical style: photoreal live-action cinematography, as if filmed on a real set with real actors. Real skin with visible pores, real woven fabric with weight and drape, real pitted steel. Sharp photographic detail throughout. Deep unlit shadow, figures carried by rim light. The stone and ground are ash grey, gunmetal and bone; the swordsman's crimson coat is deep dyed cloth that reflects light; the only light that emits in frame is live fire. Long-shutter motion blur on the fastest limbs and blades. Fine film grain, slightly crushed blacks.

The setting is {setting}.
"""

GRIDS = {4: ("4", "2-column by 2-row"), 9: ("9", "3-column by 3-row"), 12: ("12", "3-column by 4-row")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('segment')
    ap.add_argument('--panels', type=int, required=True, help='actual panel count for this board')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    d = json.load(open(ROSTERS))
    seg = d['segments'][a.segment]
    if len(seg['panels']) != a.panels:
        sys.exit("roster has %d panels but the board has %d -- re-read the board before spending "
                 "credits on it" % (len(seg['panels']), a.panels))
    cells = 4 if a.panels <= 4 else (9 if a.panels <= 9 else 12)
    roster = "\n".join("- Panel %d -- %s" % (i, p) for i, p in enumerate(seg['panels'], 1))
    text = TEMPLATE.format(cells=GRIDS[cells][0], grid=GRIDS[cells][1], roster=roster,
                           setting=d['settings'][seg['setting']])

    bad = re.findall(r"\b(no|not|never|without|avoid|don't|neither|nothing)\b", text, re.I)
    if bad:
        sys.exit("prompt contains prohibitions %s -- this model family tends to render what a "
                 "prohibition names. Rewrite positively." % sorted(set(w.lower() for w in bad)))
    open(a.out, 'w').write(text)
    print("%s -> %s  (%d panels, %s, %d chars)"
          % (a.segment, a.out, a.panels, seg['setting'], len(text)))


if __name__ == '__main__':
    main()

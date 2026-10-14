#!/usr/bin/env python3
"""Build the 2026-10-10 overnight H3 batch: part-boards, prompts (official ref2va format), queue.

Run from anywhere: python3 build.py. Writes boards/, prompts/, queue.json next to this file.
"""
import json
from pathlib import Path
from PIL import Image

HERE = Path(__file__).resolve().parent
SHOW = HERE.parent
(HERE / "boards").mkdir(exist_ok=True)
(HERE / "prompts").mkdir(exist_ok=True)

GREY = (58, 58, 58)
PW, PH = 1280, 720  # one panel in the part-boards

C2_BOARD = SHOW / "d1_action/board/v7/board_proposed.png"
C2_CELLS = json.load(open(SHOW / "d1_action/board/v6/cells.json"))
C2_CHAR = SHOW / "d1_action/refs/char_v1/char_v1_1.png"
C1_BOARD = SHOW / "d7_opener/video3/upload/Image1.png"
C1_CHAR = SHOW / "d7_opener/video3/upload/Image3.png"
C1_X = [(0, 847), (854, 1703), (1710, 2560)]
C1_Y = [(0, 472), (480, 951), (959, 1429)]


def c1_cell(n):
    r, c = divmod(n - 1, 3)
    return (C1_X[c][0], C1_Y[r][0], C1_X[c][1], C1_Y[r][1])


def part_board(src, cell_fn, panels, cols, out):
    """Panels in reading order on a cols-wide grid, thin white gaps, padded to 16:9 with grey."""
    im = Image.open(src).convert("RGB")
    rows = -(-len(panels) // cols)
    gap = 12
    w = cols * PW + (cols + 1) * gap
    h = rows * PH + (rows + 1) * gap
    grid = Image.new("RGB", (w, h), (255, 255, 255))
    for i in range(cols * rows):
        r, c = divmod(i, cols)
        x, y = gap + c * (PW + gap), gap + r * (PH + gap)
        if i < len(panels):
            grid.paste(im.crop(cell_fn(panels[i])).resize((PW, PH), Image.LANCZOS), (x, y))
        else:
            grid.paste(Image.new("RGB", (PW, PH), GREY), (x, y))
    W = max(w, round(h * 16 / 9))
    H = round(W * 9 / 16)
    canvas = Image.new("RGB", (W, H), GREY)
    canvas.paste(grid, ((W - w) // 2, (H - h) // 2))
    canvas.save(out)
    return out


def c2_cell(n):
    return tuple(C2_CELLS[str(n)])


# v2 (Mike, 2026-10-10 late): full 9-panel boards for C2 and C1; only the C2 ending uses a part-board.
boards = {
    "c2_full": C2_BOARD,
    "c1_full": C1_BOARD,
    "c2_end": part_board(C2_BOARD, c2_cell, [7, 8, 9], 3, HERE / "boards/c2_end_p7-9.png"),
}

# ---------- shared text ----------
C2_WOMAN = ("a young woman survivor, slender and athletic, pale freckled skin, violet eyes, long black hair "
            "with a violet sheen in a tight low ponytail, a dark rune tattoo on her right cheekbone only, a patched "
            "dark-brown leather jacket over a charcoal hoodie, a black chest rig with magazine pouches, torn olive "
            "cargo pants, duct-taped boots, fingerless gloves, a holster on each thigh and two black pistols")
C2_KEEP = ("her face, freckles, violet eyes, black ponytail, the rune tattoo on her right cheekbone, the leather "
           "jacket, chest rig, olive cargo pants, taped boots and the two black pistols are retained; the "
           "four-view sheet layout and grey studio background do not appear")
C2_ZOMBIES = ("Zombies are fast modern runner zombies of full adult human size, the same scale as her: grey "
              "mottled skin, clouded eyes, torn dirty everyday clothes, bare empty hands with no weapons; "
              "frightening but not gory, no blood.")
C2_PLACE = ("An abandoned ruined city at midday: a narrow alley between old brick tenements with broken doorways, "
            "rusted fire escapes with drop ladders, weeds, rubble and a burnt-out car; flat tar rooftops with "
            "water towers and narrow gaps between buildings. High sun, pale hazy sky, hard bright daylight.")
STYLE = ("Live-action, cinematic, photographic realism, shot on an ARRI Alexa 35, real skin with visible pores, "
         "real cloth and metal, physically accurate daylight, real motion blur, fine film grain. 16:9.")
C1_WOMAN = ("a strikingly beautiful young space-pirate woman with long thick wavy deep-violet hair, pale violet "
            "eyes, dark plum lipstick, freckles, a violet ink tattoo of angular circuit-like glyphs on her left "
            "forehead, temple and left cheek only, battle-worn black and gunmetal sci-fi armour with a small "
            "violet light at the collar and belt buckle, and a pistol holstered on each hip")
C1_KEEP = ("her face, violet hair, violet eyes, plum lips, freckles, the glyph tattoo on her left side and the "
           "black and gunmetal armour are retained; the sheet layout and grey studio background do not appear")
C1_STYLE = ("Live-action, cinematic, photographic realism, shot on a large-format cinema camera, real fire and "
            "smoke, debris with real weight, physically accurate light, fine film grain. 16:9. Every explosion "
            "and flame is real billowing turbulent fire in violet and magenta, never orange.")
C1_CITY = ("A spotless white and glass future city: curved white towers, a tall slender white landmark tower "
           "with sky gardens, a raised glass transit track, a polished pale plaza, a clear blue sky.")


def ts(sec):
    return "00:%06.3f" % sec


def shots_text(shots):
    out = []
    for i, (t, body) in enumerate(shots):
        out.append("[Shot 1] " + body if i == 0 else "[Shot %d] At %s, the shot cuts to %s" % (i + 1, ts(t), body))
    return "\n".join(out)


def prompt(subject_lines, summary, retention_lines, style, shots, soundscape):
    return ("subject_definitions:\n" + "\n".join(subject_lines) +
            "\n\nsummary:\n" + summary +
            "\n\nretention_analysis:\n" + "\n".join(retention_lines) +
            "\n\ndetailed_description:\n" + style + "\n" + shots_text(shots) +
            "\n\noverall_soundscape:\n" + soundscape +
            "\n\nnon_diegetic_music:\nN/A\n")


def board_line(n_shots, empty_cell):
    tail = " The last grid cell is empty grey and is not a shot." if empty_cell else ""
    return ("<Picture 1> is a storyboard reference for [Shot 1] to [Shot %d], one panel per shot read left to "
            "right and top to bottom, defining each shot's viewpoint, framing, subject placement and order.%s"
            % (n_shots, tail))


BOARD_KEEP = ("<Picture 1> (storyboard): partially_preserved - shot order, viewpoint, framing and subject "
              "placement are followed; its white grid borders, grey padding and painted texture do not appear.")

# ---------- parkour style (from action-sequence-pipeline parkour-follow-cam, the fast chase Mike liked) ----------
PARKOUR = ("A kinetic chase-thriller: the spectacle is aggressive camera movement and relentless speed. The camera "
           "mostly chases her from behind and struggles to keep up; its distance never stays the same - tight on "
           "her back, thrown far behind as she pulls away, surging to catch up, skimming low over the ground, arcing "
           "around her, whipping around corners. Never smooth or game-like. Heavy motion blur and handheld shake "
           "throughout. She never stops running: every zombie is dealt with mid-stride and every shot she fires "
           "lands. Natural full speed except the one slow-motion moment of the gap jump, which ramps down for the "
           "leap and snaps back to full speed on the landing.")

C2_FULL_CUTS = [
    (0, "behind her, tight and handheld: <Subject 1> sprints flat out down the narrow alley, a pistol in each hand; the camera is thrown back as she pulls away; one zombie sprints after her."),
    (1.6, "a low skim past the chasing zombie's shoulder: she slides across the burnt-out car hood on her hip, twists back mid-slide and shoots it; it drops; she lands running."),
    (3.2, "a whip from the side at a broken doorway: a zombie bursts out at her; she ducks under its arms without slowing and shoots it point-blank as she passes."),
    (4.6, "a front tracking surge, the camera racing backward: she charges at the lens, firing back over her shoulder; the zombie behind her falls; two more burst out of doorways and chase."),
    (6.2, "a fast close push on her gloved hand: she leaps and snatches the bottom rung of a rusted fire escape ladder."),
    (7.4, "a high overhead rush down the ladder: she scrambles up and hauls herself over the roof edge; two zombies climb fast below her."),
    (8.8, "a front tracking surge across the rooftop: she sprints at the lens, large in the frame; three zombies chase close behind, water towers flashing past."),
    (10.4, "slow motion, a side view arcing around her: she leaps the gap between two buildings, left to right, twists back in mid-air and fires both pistols; each of the three zombies at the roof edge is hit and topples into the gap."),
    (13.6, "a high overhead at full speed again: she lands on the far roof in a roll, springs up and dusts off her jacket as the zombies fall away below."),
]
C2_END_CUTS = [
    (0, "behind her, then surging past to a front tracking shot: <Subject 1> sprints flat out across a flat tar rooftop at the lens, glancing back; three zombies chase close behind, water towers whipping past; the camera is thrown back and surges again."),
    (4, "slow motion, a side view arcing around her: she leaps from the roof edge across the narrow gap, left to right, twists her whole body back in mid-air and fires both pistols; every shot lands: each of the three zombies at the edge is hit, jerks back and topples into the gap. Muzzle flashes, casings tumbling."),
    (11.5, "a high overhead, snapping back to full speed: she lands on the far roof in a roll, springs up and walks away calmly without looking back, holstering one pistol, as the zombies fall away below."),
]
C2_FULL_ONE = ("[Shot 1] The camera chases <Subject 1> from behind, tight and handheld, as she sprints flat out down a narrow ruined alley, "
    "a pistol in each hand, one zombie sprinting after her; she pulls away and the camera is thrown back, then surges to catch up; "
    "she slides across the hood of a burnt-out car on her hip, twisting back mid-slide to shoot the zombie, which drops; the camera "
    "skims low around the car and whips past a broken doorway as a second zombie bursts out - she ducks under its arms without "
    "slowing and shoots it point-blank; the camera arcs ahead of her and races backward as she charges at the lens, firing back over "
    "her shoulder at two more zombies spilling from doorways; she leaps and catches a fire escape ladder and the camera pedestals up "
    "fast beside her as she scrambles to the top and over the roof edge, two zombies climbing below; on the rooftop the camera "
    "trucks along with her as three zombies chase her; at the edge time slows: the camera arcs around her as she leaps the gap, "
    "twists back in mid-air and fires both pistols, each of the three zombies hit and toppling into the gap; time snaps back to "
    "full speed as she lands in a roll on the far roof, springs up and dusts off her jacket. No cuts: the entire video is a single "
    "unbroken shot.")
C2_END_ONE = ("[Shot 1] The camera chases <Subject 1> from behind, tight and handheld, as she sprints flat out across a flat tar rooftop, "
    "three zombies close behind her, water towers whipping past; she pulls away and the camera is thrown back, then surges and "
    "arcs ahead of her as she charges at the lens, glancing back; at the roof edge time slows: the camera arcs around her as she "
    "leaps across the narrow gap between two buildings, travelling left to right, twists her whole body back in mid-air and fires "
    "both pistols; every shot lands, each of the three zombies at the edge is hit, jerks back and topples into the gap, muzzle "
    "flashes and casings tumbling; time snaps back to full speed as she lands in a roll on the far roof, springs up and walks away "
    "calmly without looking back, holstering one pistol, the camera rising high overhead as the zombies fall away below. No cuts: "
    "the entire video is a single unbroken shot.")

C2_SOUND = ("Ragged breathing and pounding footsteps throughout; snarls and shrieks close behind; sharp pistol shots, each followed "
            "by a body hitting the ground; boots on car metal; a splintering door; a metal clang and rattling ladder; slowed, deep "
            "gunshots and a long whoosh during the slow-motion jump; a thud and a roll on gravel.")
C2_SOUND_END = ("Pounding feet on the roof and snarls; slowed, deep gunshots and a long whoosh during the slow-motion jump; bodies "
                "falling away; a thud and a roll on gravel; calm footsteps and quiet wind at the end.")

# ---------- C1 (opener style, cuts), whole piece in 15 s ----------
C1_FULL = [
    (0, "a wide establishing shot, slow push forward: the pristine white and glass city plaza and the landmark tower in bright midday light, a few distant pedestrians. <Subject 1> is not in this shot."),
    (1.6, "a medium-wide shot on the plaza: a narrow column of violet light beams down and <Subject 1> materialises inside it, facing the camera; a ring of violet light fades at her feet; pedestrians turn."),
    (3.4, "an extreme close-up of her lips, locked camera. She whispers coldly, <d>[English] Boom.</d>"),
    (4.4, "a medium-wide shot: she raises one hand beside her head and snaps her fingers with a cold half smile; the landmark tower behind her erupts in a massive explosion of violet fire."),
    (6.4, "a wide shot from behind her: the tower collapses toward her and a wave of debris and violet fire engulfs her; a transparent violet force-field dome forms around her; she stands untouched. Pedestrians flee."),
    (8.4, "an extreme wide high-angle shot: violet fire explosions burst at tower after tower across the skyline; blue sky between the smoke columns."),
    (10.2, "a high overhead wide shot descending toward her: she stands in the broken plaza among violet flames, head tilted back, laughing a slow, cold laugh."),
    (11.8, "an extreme close-up of her lips in violet firelight, ash drifting. She whispers coldly, <d>[English] Are you ready?</d>"),
    (13.2, "a wide shot from behind her silhouette: the white and glass city burning and crumbling in violet flames under a smoke-streaked blue sky."),
]
C1_SOUND = ("A quiet city ambience, then a deep rising hum for the beam; a whisper; a sharp finger snap and a huge blast; a roaring "
            "collapse with debris hammering the force field; distant blasts across the city; her slow cold laugh; a whisper; fire roar.")


def c2_job(body, sound, summary, board_shots=0, empty_cell=False, continuous=False):
    if board_shots:
        subj = ["<Subject 1> is the survivor woman in <Picture 2>, " + C2_WOMAN + ".", board_line(board_shots, empty_cell)]
        keep = ["<Subject 1> (appears in every shot): fully_preserved - " + C2_KEEP + ".", BOARD_KEEP]
    else:
        subj = ["<Subject 1> is the survivor woman in <Picture 1>, " + C2_WOMAN + "."]
        keep = ["<Subject 1> (appears throughout): fully_preserved - " + C2_KEEP + "."]
    style = STYLE + " " + C2_PLACE + " " + C2_ZOMBIES + " She is the only living person. " + PARKOUR
    text = ("subject_definitions:\n" + "\n".join(subj) + "\n\nsummary:\n[reference generation] " + summary +
            "\n\nretention_analysis:\n" + "\n".join(keep) + "\n\ndetailed_description:\n" + style + "\n" +
            (body if continuous else shots_text(body)) +
            "\n\noverall_soundscape:\n" + sound + "\n\nnon_diegetic_music:\nN/A\n")
    return text


def c1_job(with_board):
    if with_board:
        subj = ["<Subject 1> is the space-pirate woman in <Picture 2>, " + C1_WOMAN + ".", board_line(9, False)]
        keep = ["<Subject 1> (appears in every shot she is in): fully_preserved - " + C1_KEEP + ".", BOARD_KEEP]
    else:
        subj = ["<Subject 1> is the space-pirate woman in <Picture 1>, " + C1_WOMAN + "."]
        keep = ["<Subject 1> (appears in every shot she is in): fully_preserved - " + C1_KEEP + "."]
    return prompt(subj, "[reference generation] A space pirate beams into a pristine white city, snaps her fingers and destroys it with violet fire, untouched inside a force field, and laughs as the city burns.",
                  keep, C1_STYLE + " " + C1_CITY + " A fast, punchy pace: nine quick shots in fifteen seconds.", C1_FULL, C1_SOUND)


S_C2 = "A survivor sprints through a ruined alley and over the rooftops from runner zombies, shooting them mid-stride, and escapes with a slow-motion twisting leap across a gap."
S_END = "On a ruined rooftop she outruns three zombies, shoots all three in a slow-motion twisting leap across a gap, lands in a roll and walks away calmly."

jobs = [
    ("01_c2_board_parkour", c2_job(C2_FULL_CUTS, C2_SOUND, S_C2, board_shots=9), [boards["c2_full"], C2_CHAR], 770809),
    ("02_c2_ref_parkour_onetake", c2_job(C2_FULL_ONE, C2_SOUND, S_C2, continuous=True), [C2_CHAR], 770809),
    ("03_c1_board", c1_job(True), [boards["c1_full"], C1_CHAR], 770809),
    ("04_c1_ref", c1_job(False), [C1_CHAR], 770809),
    ("05_c2_end_board_parkour", c2_job(C2_END_CUTS, C2_SOUND_END, S_END, board_shots=3), [boards["c2_end"], C2_CHAR], 770809),
    ("06_c2_end_ref_parkour_onetake", c2_job(C2_END_ONE, C2_SOUND_END, S_END, continuous=True), [C2_CHAR], 770809),
    # extra ~3 h (Mike): second seeds of the two continuous takes and of the main board job
    ("07_c2_ref_parkour_onetake_s2", c2_job(C2_FULL_ONE, C2_SOUND, S_C2, continuous=True), [C2_CHAR], 4242),
    ("08_c2_end_ref_parkour_onetake_s2", c2_job(C2_END_ONE, C2_SOUND_END, S_END, continuous=True), [C2_CHAR], 4242),
    ("09_c2_board_parkour_s2", c2_job(C2_FULL_CUTS, C2_SOUND, S_C2, board_shots=9), [boards["c2_full"], C2_CHAR], 4242),
]

for old in (HERE / "prompts").glob("*.txt"):
    old.unlink()
queue = []
for name, text, refs, seed in jobs:
    pf = HERE / "prompts" / (name + ".txt")
    pf.write_text(text)
    queue.append({"name": "ovn_" + name, "prompt-file": str(pf), "ref-image": [str(r) for r in refs],
                  "seconds": 15, "seed": seed, "out-dir": str(HERE / "renders")})
json.dump(queue, open(HERE / "queue.json", "w"), indent=1)
print("\n".join("%s  refs=%d" % (q["name"], len(q["ref-image"])) for q in queue))

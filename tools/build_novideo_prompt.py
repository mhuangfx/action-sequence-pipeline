#!/usr/bin/env python3
"""Build a segment's no-video-reference H3 prompt, in the hosted mention-chip parts format.

WHY THIS EXISTS
---------------
On 2026-08-19 nine of the twelve segments had a verified board, a merged shot list and full motion
measurements -- and NO video prompt at all. Nobody had written what happens in each shot, so there
was nothing to render. segA, seg2 and seg3 had prompts, but all of them still carried a `<Video 1>`
chip from before the reference was dropped.

This assembles the whole prompt from three sources that already exist, so the shape cannot drift
between segments by hand-editing:

  config/board_rosters.json      -- who is in each panel, and which setting
  config/shot_descriptions.json  -- what happens in each shot, in Chinese
  tools/shot_prompt_lines.py     -- the mechanical clauses, measured off the source

THE RULES IT ENFORCES, AND WHY
------------------------------
1. NO `<Video 1>` chip, ever. Owner decision 2026-08-19: the reference carried the drawn medium.

2. The style is stated ONCE, globally, at the end of the scene block -- never per shot. Measured:
   a per-shot style statement produced 7/7 drawn shots against 5/7 for one global statement. The
   medium is the single exception to this pipeline's per-item rule.

3. A character who does not appear in ANY panel is not named, and their reference image is not
   passed. seg20 has no swordsman; seg22 has no woman. Asserting a character is present is exactly
   what reintroduced the replaced character in the first place, and an "only these two characters"
   line in a one-character segment is that same assertion wearing a hat.

4. The description count must equal the shot count. A mismatch means the roster, the descriptions
   and the shot list disagree, and zipping them silently misassigns every shot after the first --
   the failure mode h3_build_prompt.py's docstring already warns about.

USAGE
-----
    .venv/bin/python3 tools/build_novideo_prompt.py --segment seg06 --duration 5.29
    .venv/bin/python3 tools/build_novideo_prompt.py --all          # all twelve, print a summary

It prints the exact `--ref-image` list the segment needs, because that list now varies by segment.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ROSTERS = ROOT / "config/board_rosters.json"
DESCRIPTIONS = ROOT / "config/shot_descriptions.json"
SHOTS_DIR = ROOT / "output/fullbattle_20260808"
OUT_DIR = ROOT / "prompts/fullbattle_photoreal"

BOARD = "output/fullbattle_20260808/%s_board_merged_%s_PHOTOREAL.png"
VESTA_SHEET = "characters/char01_vesta/ref_vestia_v3.png"
ARCHER_SHEET = "characters/<hero>/sheet.png"

# Lifted from segA_video_photoreal_novideo_parts.json, the prompt the owner accepted on
# 2026-08-19. Do not reword these: the wording is the part that has been tested.
#
# COSTUME CLAUSE REWRITTEN 2026-08-24 (owner's decision). It used to read 黑铁符文战甲（胸腹完全覆盖）
# -- "torso and belly fully covered" -- while the reference sheet handed to the very same render,
# characters/char01_vesta/ref_vestia_v3.png, shows an open-front cuirass. The sheet is the truth,
# so the words now say what the picture says. English gloss of the Chinese below: an open-front
# blackened-iron runic cuirass, contoured chest plates to a high runed collar with the centre
# sternum and midriff as bare skin, pauldrons on both shoulders, forearm gauntlets, a runed hip
# belt and long ragged charcoal skirt panels; long loose crimson hair.
WOMAN_DESC = ("深红色长发披散及背，脸上有熔岩符文发光，黑铁符文战甲：胸甲沿胸型分片、正中敞开，"
              "胸骨中线与小腹为肌肤本色，高领护颈，双肩甲、护臂手甲与符文腰封，层叠深色破边裙摆，"
              "双持一对短弯刀，刀刃燃着余烬红光")
MAN_DESC = ("白色短刺发，深红色长披风，黑色上衣配白色滚边与胸前徽记，白色腰带，一对黑白弯刀")

SETTINGS = {
    "HALL": "月光下的废墟大厅，高大拱窗，地面遍布碎石",
    "PLAIN": "燃烧的焦土平原，炽红天空，遍地竖插的刀剑，远处巨大的铁齿轮在热浪中转动，空中飘着灰烬与余烬火星",
}

# Stated ONCE. See rule 2 above.
STYLE = ("真人实拍电影摄影风格：真实皮肤有毛孔、真实织物有重量与垂坠、真实金属有物理反光，"
         "画面如同真人演员在真实场景中拍摄，细腻胶片颗粒，细节丰富。")

# Board layout per panel count -- 2x2 and 3x3 are 16:9; anything else gets re-framed and the
# composition is lost, which is why the boards were tiled this way in the first place.
LAYOUT = {3: "2x2", 4: "2x2", 5: "3x3", 6: "3x3", 7: "3x3"}


def cast_of(panels):
    up = " ".join(panels).upper()
    return ("THE WOMAN" in up or "BOTH" in up,
            "THE SWORDSMAN" in up or "BOTH" in up)


def clause_lines(shots_path, duration, hitstop_floor):
    """The mechanical clauses, straight out of shot_prompt_lines.py. Each line holds a {DESC}."""
    argv = [sys.executable, str(ROOT / "tools/shot_prompt_lines.py"), str(shots_path),
            "--duration", "%.3f" % duration]
    if hitstop_floor is not None:
        argv += ["--hitstop-floor", str(hitstop_floor)]
    r = subprocess.run(argv, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit("shot_prompt_lines failed for %s:\n%s" % (shots_path, r.stderr))
    return [ln for ln in r.stdout.splitlines() if ln.strip()]


def build(segment, duration, hitstop_floor=None):
    rosters = json.loads(ROSTERS.read_text())
    panels = rosters["segments"][segment]["panels"]
    setting_key = rosters["segments"][segment]["setting"]

    descs = json.loads(DESCRIPTIONS.read_text()).get(segment)
    if descs is None:
        sys.exit("no descriptions for %s in %s" % (segment, DESCRIPTIONS))

    shots_path = SHOTS_DIR / ("shots_%s_merged_src.json" % segment)
    lines = clause_lines(shots_path, duration, hitstop_floor)

    if not (len(panels) == len(descs) == len(lines)):
        sys.exit("%s disagrees: %d panels, %d descriptions, %d shot clauses. "
                 "Zipping these would misassign every shot after the first."
                 % (segment, len(panels), len(descs), len(lines)))

    has_woman, has_man = cast_of(panels)
    if not (has_woman or has_man):
        sys.exit("%s has no characters in any panel" % segment)

    shot_text = " ".join(line.replace("{DESC}", d) for line, d in zip(lines, descs))

    # --- the board block -------------------------------------------------------------------
    if has_woman and has_man:
        cast_clause = "；画面中自始至终只有分镜图里的这两名角色。"
    else:
        cast_clause = "；画面中自始至终只有分镜图里的这一名角色。"

    parts = [
        {"t": "本片的画面内容完全以分镜网格图 "},
        {"m": "Image1"},
        {"t": " 为准：每一格对应下列同编号镜头，构图、镜头角度、人物姿势、人物造型、服装、"
              "面容与场景一律照该分镜图执行" + cast_clause},
    ]

    refs = [BOARD % (segment, LAYOUT[len(panels)])]
    idx = 2

    # --- character blocks, only for characters who actually appear ------------------------
    if has_woman:
        parts.append({"t": "女主角的面容、发型与战甲以 "})
        parts.append({"m": "Image%d" % idx})
        tail = " 为准——%s。" % WOMAN_DESC
        refs.append(VESTA_SHEET)
        idx += 1
        if has_man:
            tail += "白发战士的面容、发色、服装与武器以 "
            parts.append({"t": tail})
            parts.append({"m": "Image%d" % idx})
            refs.append(ARCHER_SHEET)
            idx += 1
            tail = " 为准——%s。" % MAN_DESC
        parts.append({"t": tail + SETTINGS[setting_key] + "。" + STYLE + " " + shot_text})
    else:
        parts.append({"t": "白发战士的面容、发色、服装与武器以 "})
        parts.append({"m": "Image%d" % idx})
        refs.append(ARCHER_SHEET)
        parts.append({"t": " 为准——%s。" % MAN_DESC
                           + SETTINGS[setting_key] + "。" + STYLE + " " + shot_text})

    # --- closing consistency line ---------------------------------------------------------
    if has_woman and has_man:
        closing = ("全片自始至终只有这两名角色同框出场：深红色长发、脸上有熔岩符文的女主角，"
                   "与白色短刺发、深红色长披风的战士。")
    elif has_woman:
        closing = "全片自始至终只有这一名角色出场：深红色长发、脸上有熔岩符文的女主角。"
    else:
        closing = "全片自始至终只有这一名角色出场：白色短刺发、深红色长披风的战士。"
    parts.append({"t": " " + closing +
                       "每一个镜头都保持真人实拍电影摄影风格，真实光影、真实皮肤与织物质感、"
                       "细腻胶片颗粒，人物造型与服装在每一格中完全一致。"})

    return parts, refs, len(panels)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--segment")
    ap.add_argument("--duration", type=float)
    ap.add_argument("--hitstop-floor", type=float, default=None,
                    help="passed through to shot_prompt_lines.py; its default is 3.0")
    ap.add_argument("--all", action="store_true", help="build every segment in the duration table")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()

    # Output lengths, from the deliverable table in prompts/gauntlet_loop_prompt.md §0.
    TABLE = {"segA": 5.80, "seg2": 5.49, "seg3": 4.35, "seg06": 5.29, "seg09": 5.48,
             "seg15": 5.53, "seg16": 5.64, "seg18": 5.37, "seg19": 5.02, "seg20": 5.87,
             "seg22": 4.19, "seg23": 5.18}

    if args.all:
        todo = sorted(json.loads(DESCRIPTIONS.read_text()).keys())
        todo = [s for s in todo if not s.startswith("_")]
    elif args.segment:
        todo = [args.segment]
    else:
        sys.exit("give --segment or --all")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for seg in todo:
        dur = args.duration if (args.segment and args.duration) else TABLE[seg]
        parts, refs, n = build(seg, dur, args.hitstop_floor)
        dest = out_dir / ("%s_video_photoreal_novideo_parts.json" % seg)
        dest.write_text(json.dumps(parts, ensure_ascii=False, indent=1) + "\n")
        chars = sum(len(p["t"]) for p in parts if "t" in p)
        print("%-6s %d shots  %.2fs  %d refs  %d chars  -> %s"
              % (seg, n, dur, len(refs), chars, dest.relative_to(ROOT)))
        for r in refs:
            print("          --ref-image %s" % r)


if __name__ == "__main__":
    main()

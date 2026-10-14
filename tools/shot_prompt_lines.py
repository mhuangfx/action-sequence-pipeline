#!/usr/bin/env python3
"""shot_prompt_lines.py — turn a reference clip's shots.json into per-shot prompt clauses.

Why: H3 reproduces WHAT happens and the cut timing, but measured against the source our outputs
drove 2 of 5 comparable shots the OPPOSITE way across frame (2026-08-09). Momentum stopped carrying
across cuts, which is what made the choreography read weightless. Everything the analysis already
measures — screen direction, impact reversal, hit-stop, camera move, relative speed — was being
thrown away instead of written into the prompt. This emits those clauses mechanically so they can't
be forgotten, leaving only the content description to write by hand.

Camera especially: H3 attenuates camera movement ~7x in pan, so it must be stated explicitly or it
does not survive at all.

Usage:
  .venv/bin/python tools/shot_prompt_lines.py SHOTS.json --duration 4.0
"""
import argparse, json

CAM = {"pan right": "摄影机向右横移", "pan left": "摄影机向左横移",
       "tilt down": "向下俯摇", "tilt up": "向上仰摇",
       "push in": "镜头推近", "pull out": "镜头拉远"}
DIR = {"screen right": "画面右侧", "screen left": "画面左侧", "up": "画面上方", "down": "画面下方"}

# Detection floors from shot_breakdown.camera_motion() (pan/tilt 1.2, push 0.8). Raw magnitudes sit
# on different scales, so "dominant axis" only means something after dividing by them.
AXIS_FLOOR = {"pan": 1.2, "tilt": 1.2, "push": 0.8}
# Calibrated 2026-08-09 against segments 1-3: FAST fires on 3/11, 4/8, 1/5 shots and LARGE on
# 2/11, 3/8, 2/5. A minority, deliberately -- qualifying every shot re-flattens dynamic range the
# same way naming every shot "fast" does.
LARGE_AMPLITUDE = 2.5   # rate x duration, in units of "just detectable"
FAST_SPEED = 2.5        # rate
# Hit-stop fires on 75% of the 56 merged shots if every detection is written. That flattens impact
# exactly the way naming every shot "fast" flattens speed, and merging is what inflated it -- a
# longer window is likelier to contain some dip in the motion curve. Ratios run 1.8 to 44.7, median
# 4.35; everything under 3 is a lull rather than a blow. This floor puts the plain clause on ~27% of
# shots and the heavy one on ~23%, leaving half of them with no impact language at all.
HITSTOP_FLOOR = 3.0
HITSTOP_HEAVY = 6.0


def cam_clause(c, dur):
    """One primary camera move per shot, qualified by amplitude and speed.

    Previously this joined EVERY detected axis, so a shot could read
    "摄影机向右横移、向下俯摇、镜头推近" -- three moves at equal weight, which is the documented
    cause of "motion is mush, no readable camera move". Amplitude and speed are separate official
    dimensions; a bare move is read as medium amplitude at normal speed, which is half of why we
    measured camera attenuating ~7x and never addressed it.

    2D optical flow cannot tell a pan from a truck from a tracking shot -- all three produce
    image-plane translation. For a fight most horizontal movement is really the camera following a
    fighter, so this is the one clause per shot worth overriding by hand.
    """
    if not c or c.get("label") in (None, "locked off"):
        return "摄影机固定"
    axes = {k: abs(c.get(k) or 0) / AXIS_FLOOR[k] for k in AXIS_FLOOR}
    axis = max(axes, key=axes.get)
    rate = axes[axis]
    travel = rate * dur
    if axis == "pan":
        name = "pan right" if (c.get("pan") or 0) > 0 else "pan left"
    elif axis == "tilt":
        name = "tilt down" if (c.get("tilt") or 0) > 0 else "tilt up"
    else:
        name = "push in" if (c.get("push") or 0) < 0 else "pull out"
    quals = []
    if travel >= LARGE_AMPLITUDE:
        quals.append("幅度大")
    if rate >= FAST_SPEED:
        quals.append("速度快")
    return CAM[name] + ("，" + "、".join(quals) if quals else "")


def dir_clause(sub):
    if not sub or sub.get("drive") in (None, "holds position"):
        return None
    tgt = "、".join(DIR.get(p, p) for p in sub["drive"].split(" + "))
    return f"动作整体朝{tgt}推进"


def rev_clause(sub):
    r = (sub or {}).get("reversal")
    if not r:
        return None
    return f"前半段朝{DIR.get(r['from'], r['from'])}，受击后反向被推向{DIR.get(r['to'], r['to'])}"


def hit_clause(sub, floor=HITSTOP_FLOOR):
    """Impact held long enough to be representable, plus the stop and the shake.

    "About two frames" is 83ms. The text encoder samples the reference at 2 fps and the video VAE
    compresses it to ~7 latent fps (~140ms), so an 83ms freeze is below the resolution of BOTH
    conditioning paths -- we were asking for something the model cannot represent. 0.2s clears it.

    Weight lives in the STOP, not the swing, so the stop is now stated. Screen shake is a native
    camera-motion type we had never used.
    """
    h = (sub or {}).get("hitstop")
    if not h or h["ratio"] < floor:
        return None
    heavy = h["ratio"] >= HITSTOP_HEAVY
    bits = ["命中瞬间定格约0.2秒，伴随白色闪光"]
    if heavy:
        bits.append("画面剧烈震动")
    bits += ["命中即刻停住，重心一次落定", "随后猛然反弹"]
    if heavy:
        bits.append("冲击极重")
    return "，".join(bits)


def settle_clause(sub, dur, floor=HITSTOP_FLOOR):
    """Give a long beat its aftermath.

    H3 reads beat duration literally as event speed. An impact occupies 0.4-0.6s; a 1.2s beat whose
    description names only the impact leaves ~0.7s unaccounted and the model fills it with drift --
    the "weightless" symptom by another name. Secondary motion landing a beat late is the cue our
    prompts never carried.
    """
    h = (sub or {}).get("hitstop")
    if dur < 0.8 or not h or h["ratio"] < floor:
        return None
    return "余波中碎石与尘土向外扬起，发梢与裙摆比身体晚一拍才落定"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("shots"); p.add_argument("--duration", type=float, required=True,
                                            help="target H3 output length in seconds")
    p.add_argument("--hitstop-floor", type=float, default=HITSTOP_FLOOR,
                   help="ignore hit-stops weaker than this ratio; keeps impact language a minority")
    a = p.parse_args()
    d = json.load(open(a.shots))
    # The merged per-segment lists are bare arrays with no "duration"; shot_breakdown's own output
    # is a dict. Accept both, and take the span from the last cut when it is not stated.
    shots = d["shots"] if isinstance(d, dict) else d
    span = d.get("duration") if isinstance(d, dict) else None
    span = span or max(s["end"] for s in shots)
    if not any(s.get("motion") is not None for s in shots):
        raise SystemExit(
            f"{a.shots} carries no motion/camera/subject data, so there are no action clauses to "
            "write. Re-measure it:\n"
            "  .venv/bin/python3 tools/shot_breakdown.py <seg>_15x.mp4 --out-dir DIR \\\n"
            "      --motion --camera --subject --shots-json <this file> --shots-scale 1.0\n"
            "then merge motion/camera/subject back into this file. With no video reference these "
            "clauses are the ONLY thing carrying the action.")
    k = a.duration / span
    ms = [s.get("motion") or 0 for s in shots]
    hi, lo = max(ms), min(ms)
    for i, s in enumerate(shots):
        sub = s.get("subject")
        dur = s["end"] - s["start"]
        bits = [cam_clause(s.get("camera"), dur)]
        m = s.get("motion") or 0
        # Only the extremes get a speed word. H3 flattens motion dynamic range (round 2), and
        # naming every shot "fast" would flatten it right back.
        if hi and m >= 0.75 * hi:
            bits.append("动作极快")
        elif hi and m <= max(0.15 * hi, lo):
            bits.append("几乎静止，屏息般的停顿")
        for c in (dir_clause(sub), rev_clause(sub), hit_clause(sub, a.hitstop_floor),
                  settle_clause(sub, dur, a.hitstop_floor)):
            if c:
                bits.append(c)
        print(f"镜头{i+1}（{s['start']*k:.2f}–{s['end']*k:.2f}）" + "，".join(bits) + "。{DESC}")


if __name__ == "__main__":
    main()

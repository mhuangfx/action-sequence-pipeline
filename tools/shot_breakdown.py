#!/usr/bin/env python3
"""
shot_breakdown.py — split a reference clip into shots, then describe each shot.

Two separate jobs, deliberately handled by two different tools:

  TIMING      PySceneDetect's ContentDetector (HSV content deltas + min_scene_len).
              Measured against ffmpeg's `scdet` on a 15s anime fight window (2026-08-08):
              both agree on the high-contrast cuts, but scdet MISSED 4 real cuts in a
              darker dialogue stretch, collapsing 5 shots into one 5.34s block (14 shots
              vs the correct 17). Cut timing drives the whole per-shot prompt, so the
              extra dependency is worth it. `--engine ffmpeg` keeps the zero-install
              path available for a rough pass.

  DESCRIPTION a VLM, one call per shot, on that shot's representative frame.
              A VLM is the wrong tool for timing (it samples frames and guesses
              timestamps; it cannot resolve a 0.13s cut) and the right tool for
              "what kind of shot is this".

Output: JSON shot table {index, start, end, dur, score, frame, description}, ready to
render into the SHOT 1 / SHOT 2 prompt form that Seedance validated (docs/prompt_findings.md).

  MOTION      inter-frame luma delta per shot (--motion), an objective stand-in for "how fast
              does this move". Added after the probe left "the output feels slow" unmeasured.

Usage:
  python3 tools/shot_breakdown.py CLIP.mp4 --out-dir output/shots/foo
  python3 tools/shot_breakdown.py CLIP.mp4 --out-dir ... --describe --motion \
      --model qwen3-vl:8b-instruct-q8_0
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

SCD_RE = re.compile(r"lavfi\.scd\.score: *([0-9.]+), lavfi\.scd\.time: *([0-9.]+)")


def probe_duration(clip: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(clip)],
        capture_output=True, text=True, check=True).stdout.strip()
    return float(out)


def detect_cuts(clip: Path, threshold: float):
    """Return [(time, score)] of raw scdet detections."""
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-i", str(clip),
         "-vf", f"scdet=threshold={threshold}", "-f", "null", "-"],
        capture_output=True, text=True)
    hits = [(float(t), float(s)) for s, t in SCD_RE.findall(proc.stderr)]
    return sorted(hits)


def detect_shots_pyscene(clip: Path, threshold: float, min_shot: float):
    """ContentDetector shot list -> [{start,end,dur,score}]. The accurate path."""
    from scenedetect import ContentDetector, detect
    rate = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=r_frame_rate", "-of", "default=noprint_wrappers=1:nokey=1", str(clip)],
        capture_output=True, text=True, check=True).stdout.strip()
    num, _, den = rate.partition("/")          # e.g. "2997/100" -> 29.97 fps
    fps = float(num) / float(den or 1)
    scenes = detect(str(clip), ContentDetector(threshold=threshold,
                                               min_scene_len=max(1, int(min_shot * fps))))
    shots = []
    for i, (s, e) in enumerate(scenes, 1):
        start, end = s.seconds, e.seconds
        shots.append({"index": i, "start": round(start, 3), "end": round(end, 3),
                      "dur": round(end - start, 3), "score": None})
    return shots


def cluster(hits, min_shot: float, duration: float):
    """Collapse detections closer than min_shot, keeping the highest-scoring one.

    A hard cut and the VFX flash right after it land within a few frames of each other;
    without this floor a 15s action clip yields ~21 'shots', most of them 2 frames long.
    """
    boundaries = []
    for t, s in hits:
        if boundaries and t - boundaries[-1][0] < min_shot:
            # same cut event — keep whichever frame scored higher
            if s > boundaries[-1][1]:
                boundaries[-1] = (t, s)
            continue
        boundaries.append((t, s))

    shots, prev, prev_score = [], 0.0, 0.0
    for t, s in boundaries:
        if t - prev >= min_shot:
            shots.append({"start": round(prev, 3), "end": round(t, 3),
                          "dur": round(t - prev, 3), "score": round(prev_score, 1)})
            prev, prev_score = t, s
    if duration - prev >= 0.05:
        shots.append({"start": round(prev, 3), "end": round(duration, 3),
                      "dur": round(duration - prev, 3), "score": round(prev_score, 1)})
    for i, sh in enumerate(shots, 1):
        sh["index"] = i
    return shots


def extract_frames(clip: Path, shots, out_dir: Path, duration: float):
    """Grab one representative frame per shot (40% in — past the cut, before the next).

    The detector's last shot can end a hair past the real last decodable frame, so a naive
    seek writes no file at all. Clamp, then walk the timestamp backwards until a frame lands.
    """
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    kept = []
    for sh in shots:
        fp = frames_dir / f"shot_{sh['index']:02d}.jpg"
        base = min(sh["start"] + sh["dur"] * 0.4, max(0.0, duration - 0.05))
        for ts in (base, sh["start"] + 0.02, max(0.0, base - 0.25), max(0.0, base - 0.5)):
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{ts:.3f}",
                            "-i", str(clip), "-frames:v", "1", "-q:v", "3", str(fp)],
                           capture_output=True)
            if fp.exists() and fp.stat().st_size > 0:
                sh["frame"] = str(fp)
                kept.append(sh)
                break
        else:
            print(f"  WARN: no frame extractable for shot {sh['index']} "
                  f"({sh['start']:.2f}-{sh['end']:.2f}) — dropping")
    return kept


VLM_PROMPT = (
    "This is one shot from a fight sequence. Reply with JSON only, no prose, with keys: "
    '"description" (1-2 dense sentences: shot size — extreme close-up/close-up/medium/wide — '
    "camera angle and movement, the on-screen character's body action, and the dominant visual "
    'effect), "character_count" (integer, how many distinct people are visible; 0 if none), '
    '"cast" (array of short visual descriptors for each visible person, e.g. '
    '["red hair, white and blue shirt"]), and "impact_vfx" (array of PHYSICAL impact and '
    "environmental effects actually visible, chosen from: dust cloud, flying debris, ground "
    "crack, ground slam, shockwave, sparks, explosion, fire, smoke, water spray, energy trail, "
    "motion blur, speed lines, screen shake, falling rubble, wind gust, blood. Empty array if "
    'none). Be concrete and visual. '
    "Do not name any anime, character, or franchise."
)


def describe(shots, model: str, host: str):
    """Describe each shot's frame via the ollama HTTP API.

    NOTE: use the API, not `ollama run <prompt> <image>`. The CLI form silently drops the
    image and answers from the text prompt alone — verified 2026-08-08, it invented a
    punch-to-the-jaw for a frame that was a wide flame-arc sword swing. The API's
    base64 `images` field is grounded correctly.
    """
    import base64
    import json as _json
    import urllib.request

    for sh in shots:
        img = base64.b64encode(Path(sh["frame"]).read_bytes()).decode()
        payload = _json.dumps({"model": model, "prompt": VLM_PROMPT, "images": [img],
                               "format": "json", "stream": False}).encode()
        req = urllib.request.Request(f"{host}/api/generate", data=payload,
                                     headers={"Content-Type": "application/json"})
        sh["character_count"], sh["cast"], sh["impact_vfx"] = None, [], []
        try:
            raw = _json.loads(urllib.request.urlopen(req, timeout=300).read())["response"].strip()
            parsed = _json.loads(raw)
            sh["description"] = re.sub(r"\s+", " ", str(parsed.get("description", "")))
            sh["character_count"] = parsed.get("character_count")
            sh["cast"] = [str(c) for c in (parsed.get("cast") or [])]
            sh["impact_vfx"] = [str(v).lower() for v in (parsed.get("impact_vfx") or [])]
        except Exception as e:  # keep the table usable even if one shot fails
            sh["description"] = f"[VLM failed: {e}]"
        cast = "; ".join(sh["cast"])[:60]
        vfx = ", ".join(sh["impact_vfx"])[:50] or "-"
        print(f"  shot {sh['index']:02d} ({sh['dur']:.2f}s) [{sh['character_count']}p {cast}] "
              f"vfx[{vfx}]: {sh['description'][:70]}", flush=True)
    return shots


META_RE = re.compile(r"pts_time:([0-9.]+)\s*\nlavfi\.signalstats\.YAVG=([0-9.]+)")


def motion_energy(clip: Path, shots, out_dir: Path):
    """Mean inter-frame luma delta per shot — a cheap, objective proxy for how fast a clip moves.

    Why this exists: after the 2026-08-08 probe, "the output feels slow" was the whole open
    question and there was no number behind it. Cut points alone don't answer it — G5b's cuts
    landed ~9% AHEAD of the source and it still read slow, which points at within-shot motion.

    Method: normalize scale and frame rate (so a 2560x1440/24fps output is comparable to a
    720p/23.976 source), difference each frame against the previous, and take the average
    brightness of that difference image. More movement -> brighter difference -> higher number.

    Two caveats, both real:
      - Frames straddling a cut produce a huge spike, so 0.15s either side of a shot boundary
        is excluded.
      - Texture affects the value. A painterly-photoreal render and a flat-cel anime source
        differ in absolute terms even at identical motion, so cross-STYLE comparison is
        directional only. Comparing our own outputs to each other (1.0x ref vs 1.5x ref) is
        the reliable use.
    """
    # metadata=print needs an explicit file sink: with `-f null -` taking stdout, the default
    # sink produces nothing at all.
    meta = out_dir / "motion_yavg.txt"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-v", "error", "-y", "-i", str(clip), "-vf",
         "scale=640:-2,fps=24,tblend=all_mode=difference,signalstats,"
         f"metadata=print:key=lavfi.signalstats.YAVG:file={meta}",
         "-f", "null", "-"], capture_output=True, text=True)
    samples = [(float(t), float(v))
               for t, v in META_RE.findall(meta.read_text() if meta.exists() else "")]
    if not samples:
        print("  WARN: motion energy unavailable (no signalstats output)")
        return shots, None

    # The cut spike lands on a shot's FIRST frame (differenced against the previous shot's last
    # frame), so only the head needs a guard. Guarding the tail as well threw away half the
    # samples on sub-second shots and left the spike in via the short-shot fallback.
    guard = 0.15
    for sh in shots:
        vals = [v for t, v in samples if sh["start"] + guard <= t <= sh["end"]]
        if not vals:  # shot shorter than the guard — take whatever is inside it
            vals = [v for t, v in samples if sh["start"] < t <= sh["end"]]
        sh["motion"] = round(sum(vals) / len(vals), 2) if vals else None

    scored = [sh["motion"] for sh in shots if sh.get("motion") is not None]
    overall = round(sum(scored) / len(scored), 2) if scored else None
    print(f"  motion energy: overall {overall} | " +
          " ".join(f"s{sh['index']}={sh.get('motion')}" for sh in shots))
    return shots, overall


LM_RE = re.compile(r"\(LM (-?\d+) (-?\d+) (\d+) (\d+) (\d+) ([\d.]+) ([\d.]+)\)")
FRAME_RE = re.compile(r"^Frame (\d+) \(List \d+ \[(.*)\]\)$", re.M)


def camera_motion(clip: Path, shots, out_dir: Path, fps=24, width=640):
    """Measure per-shot CAMERA movement (pan / tilt / push) via vidstabdetect motion vectors.

    Why measured and not asked: the VLM sees ONE frame per shot, so any camera MOVEMENT it reports
    is inferred from motion blur, not observed. Checked 2026-08-08 against the source clip — it
    called a locked-off shot "dynamic camera movement" because the fighters were moving, and left
    8 of 11 shots with no movement note at all. It is reliable for shot ANGLE (low/high), not motion.

    Method: vidstabdetect emits local motion vectors per frame. The median across all vectors
    approximates global camera translation (the subject is a minority of the frame). Projecting each
    vector onto the radial direction from frame centre gives push/pull: vectors diverging from centre
    = pushing in.

    ★ SIGN CONVENTION IS CALIBRATED AGAINST SYNTHETIC GROUND TRUTH (2026-08-08), because the naive
    reading is inverted on all three axes. Clips with a known camera move measured:
        camera pans RIGHT -> pan +12.63      (so pan  > 0 = RIGHT)
        camera tilts DOWN -> tilt +6.32      (so tilt > 0 = DOWN)
        camera pushes IN  -> push -1.29      (so push < 0 = IN)
    Do not "fix" these signs by reasoning about which way content moves — re-run the synthetic test.

    Known confound: a large fast subject filling the frame contaminates the estimate, and the
    push/pull axis is the first to go. When one axis dominates in magnitude, trust that one.
    """
    trf = out_dir / "camera.trf"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(clip), "-vf",
         f"scale={width}:-2,fps={fps},vidstabdetect=shakiness=10:accuracy=15:result={trf}",
         "-f", "null", "-"], capture_output=True)
    if not trf.exists():
        print("  WARN: camera motion unavailable (vidstabdetect produced no output)")
        return shots
    txt = trf.read_text()

    cx = width / 2
    ys = [int(m[3]) for m in LM_RE.findall(txt)] or [180]
    cy = max(ys) / 2

    per_frame = {}
    for fnum, body in FRAME_RE.findall(txt):
        vs = LM_RE.findall(body)
        if len(vs) < 4:
            continue
        rad = []
        for dx, dy, x, y, *_ in vs:
            dx, dy, x, y = int(dx), int(dy), int(x), int(y)
            rx, ry = x - cx, y - cy
            n = (rx * rx + ry * ry) ** 0.5
            if n > 20:
                rad.append((dx * rx + dy * ry) / n)
        import statistics
        per_frame[int(fnum)] = (statistics.median(int(v[0]) for v in vs),
                                statistics.median(int(v[1]) for v in vs),
                                statistics.median(rad) if rad else 0.0)

    for sh in shots:
        f0, f1 = int(sh["start"] * fps) + 2, int(sh["end"] * fps)
        fs = [per_frame[f] for f in range(f0, f1 + 1) if f in per_frame]
        if not fs:
            sh["camera"] = None
            continue
        pan = round(sum(f[0] for f in fs) / len(fs), 2)
        tilt = round(sum(f[1] for f in fs) / len(fs), 2)
        push = round(sum(f[2] for f in fs) / len(fs), 2)
        parts = []
        if abs(pan) > 1.2:
            parts.append("pan right" if pan > 0 else "pan left")
        if abs(tilt) > 1.2:
            parts.append("tilt down" if tilt > 0 else "tilt up")
        if abs(push) > 0.8:
            parts.append("push in" if push < 0 else "pull out")
        sh["camera"] = {"pan": pan, "tilt": tilt, "push": push,
                        "label": " + ".join(parts) if parts else "locked off"}
        print(f"  shot {sh['index']:02d}: {sh['camera']['label']:<34} "
              f"(pan {pan:+.1f} tilt {tilt:+.1f} push {push:+.1f})")
    return shots


# Up to three modifiers may sit between the colour and the noun ("red spiky hair", "long white
# hair tied back"). Matching only the immediately-preceding word captures "spiky" and flags both
# intended fighters as off-cast — verified 2026-08-08, it wrongly excluded 33s of usable action.
HAIR_RE = re.compile(r"((?:[a-z]+[\s-]+){1,3}?)hair\b")
HAIR_COLORS = {"red", "white", "blond", "blonde", "black", "brown", "silver", "gray", "grey",
               "blue", "green", "orange", "crimson", "auburn", "ginger", "platinum", "golden",
               "purple", "pink", "reddish", "dark", "light"}
# Hair words the VLM uses interchangeably for the same head, collapsed to one bucket.
HAIR_SYNONYMS = {"gray": "white", "grey": "white", "silver": "white", "platinum": "white",
                 "blonde": "blond", "golden": "blond", "auburn": "red", "ginger": "red",
                 "crimson": "red", "reddish": "red", "orange": "red"}



def subject_motion(clip: Path, shots, out_dir: Path, width=480, step=1):
    """Per-shot SUBJECT screen-direction and impact weight — where bodies get DRIVEN in frame.

    Why this exists: our swaps reproduce the source's actions and cut timing but read lighter than
    the source. The missing ingredient is directional follow-through — in the reference an impact
    drives both fighters a specific way across frame, which is what sells weight and makes the
    choreography readable. Per-shot prompts described WHAT happens and never WHICH WAY it goes, so
    H3 had no reason to carry momentum through a hit.

    Method: DENSE optical flow (Farneback), not vidstabdetect. Tried the sparse route first —
    reusing camera.trf and taking the residual after the median — and calibration showed it dead:
    a subject crossing the frame produced |dx| < 0.1 because vidstabdetect's measurement fields
    mostly land on background, so the residual average is diluted to noise. Dense flow measures
    every pixel, so the subject cannot be missed.

    Per frame pair: the per-axis MEDIAN of the flow field is camera motion (the subject is a
    minority of the frame); subtracting it leaves the residual. The mean residual over the top
    decile of residual magnitude is the subject's screen displacement.

    Reports three things per shot:
      - drive:    dominant subject direction across the shot (screen left/right, up/down)
      - reversal: direction in the first vs second half; a sign flip is an impact knocking the
                  fighters back the other way — the single most useful thing to prompt
      - hitstop:  a dip in motion energy immediately followed by a spike, i.e. the anime impact
                  freeze then recoil. Needs --motion to have run (reads motion_yavg.txt).

    ★ SIGNS CALIBRATED AGAINST SYNTHETIC GROUND TRUTH via tools/calibrate_subject_motion.py, as
    for camera_motion. Do not re-derive them by reasoning about which way content moves.
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        print("  WARN: subject motion needs opencv-python + numpy")
        return shots

    cap = cv2.VideoCapture(str(clip))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    prev = None
    per_t = []  # (timestamp, dx, dy)
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            h = int(frame.shape[0] * width / frame.shape[1])
            g = cv2.cvtColor(cv2.resize(frame, (width, h)), cv2.COLOR_BGR2GRAY)
            if prev is not None:
                fl = cv2.calcOpticalFlowFarneback(prev, g, None, 0.5, 3, 21, 3, 5, 1.2, 0)
                fx, fy = fl[..., 0], fl[..., 1]
                # camera = the median of the whole field; the subject is the minority that differs
                rx, ry = fx - np.median(fx), fy - np.median(fy)
                mag = np.hypot(rx, ry)
                cut = np.quantile(mag, 0.90)
                m = mag >= cut
                if m.any():
                    per_t.append((idx / fps, float(rx[m].mean()), float(ry[m].mean())))
            prev = g
        idx += 1
    cap.release()
    if not per_t:
        print("  WARN: subject motion unavailable (no frames read)")
        return shots

    meta = out_dir / "motion_yavg.txt"
    samples = [(float(t), float(v))
               for t, v in META_RE.findall(meta.read_text() if meta.exists() else "")]

    # CALIBRATION 2026-08-09 (tools/calibrate_subject_motion.py, dense-flow path) — measured:
    #   subject crosses SCREEN RIGHT -> dx +0.24    subject crosses SCREEN LEFT -> dx -0.26
    #   subject moves DOWN           -> dy +0.18    subject moves UP            -> dy -0.19
    # so dense-flow signs read DIRECTLY (unlike camera_motion, inverted on all three axes).
    # Those magnitudes are for a subject filling only 3.5% of frame; real fighters read far
    # larger, hence a threshold well below the synthetic values. A subject
    # crossing screen-right measures dx > 0 and one falling measures dy > 0, so dense-flow signs
    # read DIRECTLY, unlike camera_motion's vidstab signs which are inverted on all three axes.
    THRESH = 0.15

    def label(dx, dy, thresh=THRESH):
        parts = []
        if abs(dx) > thresh:
            parts.append("screen right" if dx > 0 else "screen left")
        if abs(dy) > thresh:
            parts.append("down" if dy > 0 else "up")
        return " + ".join(parts)

    for sh in shots:
        seg = [(t, dx, dy) for t, dx, dy in per_t if sh["start"] + 0.06 <= t <= sh["end"]]
        if len(seg) < 2:
            sh["subject"] = None
            continue
        dxs = [d for _, d, _ in seg]
        dys = [d for _, _, d in seg]
        dx, dy = sum(dxs) / len(dxs), sum(dys) / len(dys)
        drive = label(dx, dy) or "holds position"

        # A reversal is only worth prompting if it is BIG. At the plain direction threshold nearly
        # every shot "reverses" on frame-to-frame noise (9 of 9 on the segment-1 reference), which
        # is the same as reporting nothing. Require the flip on ONE axis, with both halves well
        # clear of the noise floor, and report only that axis.
        mid = len(seg) // 2
        rev = None
        if mid >= 2:
            ax, ay = sum(dxs[:mid]) / mid, sum(dys[:mid]) / mid
            bx = sum(dxs[mid:]) / len(dxs[mid:])
            by = sum(dys[mid:]) / len(dys[mid:])
            REV_MIN = 0.8
            cands = []
            if ax * bx < 0 and min(abs(ax), abs(bx)) > REV_MIN:
                cands.append((min(abs(ax), abs(bx)),
                              "screen right" if ax > 0 else "screen left",
                              "screen right" if bx > 0 else "screen left"))
            if ay * by < 0 and min(abs(ay), abs(by)) > REV_MIN:
                cands.append((min(abs(ay), abs(by)),
                              "down" if ay > 0 else "up", "down" if by > 0 else "up"))
            if cands:
                _, frm, to = max(cands)
                rev = {"at": round(seg[mid][0], 2), "from": frm, "to": to}

        # Hit-stop = a motion dip immediately followed by a spike. The dip needs an ABSOLUTE floor
        # as well as a relative one: near a cut guard the luma delta falls to ~0.001 and the naive
        # ratio explodes (measured x3908 on shot 1 of the segment-1 reference, pure artefact).
        hit = None
        vals = [(t, v) for t, v in samples if sh["start"] + 0.15 <= t <= sh["end"]]
        if len(vals) >= 5:
            mean = sum(v for _, v in vals) / len(vals)
            DIP_FLOOR = 0.5
            for i in range(1, len(vals) - 2):
                t, v = vals[i]
                if v < DIP_FLOOR:
                    continue
                after = max(v2 for _, v2 in vals[i + 1: i + 4])
                if v < 0.55 * mean and after > 1.8 * v:
                    hit = {"at": round(t, 2), "ratio": round(min(after / v, 99.0), 1)}
                    break

        sh["subject"] = {"dx": round(dx, 2), "dy": round(dy, 2), "drive": drive,
                         "reversal": rev, "hitstop": hit}
        extra = ""
        if rev:
            extra += f"  REVERSES {rev['from']} -> {rev['to']} @{rev['at']}s"
        if hit:
            extra += f"  HITSTOP @{hit['at']}s (x{hit['ratio']})"
        print(f"  shot {sh['index']:02d}: drive {drive:<24} (dx {dx:+.2f} dy {dy:+.2f}){extra}")
    return shots


def _hair_colors(text):
    """Colour words describing hair, allowing modifiers between colour and noun."""
    out = set()
    for phrase in HAIR_RE.findall(text.lower()):
        for w in re.split(r"[\s-]+", phrase.strip()):
            if w in HAIR_COLORS:
                out.add(HAIR_SYNONYMS.get(w, w))
    return out


def flag_offcast(shots, expect_cast):
    """Mark shots whose visible cast contradicts the expected fighters.

    A compilation window often cuts away to a bystander or a rescued third character. Those
    shots pollute a character-swap prompt (there is nobody to map them onto), so we flag them
    for removal. `expect_cast` is descriptors of the intended fighters, e.g.
    ["red hair", "white hair"].

    The rule is deliberately conservative: flag only on POSITIVE contradiction — a visible
    person whose stated hair colour is not one of the expected ones. Shots that simply fail to
    mention hair (an extreme close-up of a hand, an eye, a boot) are left alone. The naive
    "no descriptor matched" rule flagged 7 of 23 shots on a real clip when only 1 was truly a
    third character, because body-part close-ups never mention hair.
    """
    expected = set()
    for tok in expect_cast:
        expected |= _hair_colors(tok) or {HAIR_SYNONYMS.get(tok.strip(), tok.strip())}

    for sh in shots:
        sh["offcast"] = False
        if not expected or not sh.get("cast"):
            continue
        seen = [(person, _hair_colors(person)) for person in sh["cast"]]
        identifiable = [(p, c) for p, c in seen if c]
        if not identifiable:
            continue  # nobody's hair is visible — can't contradict, so keep the shot
        # off-cast only when every identifiable person's hair is foreign to the expected cast
        sh["offcast"] = all(not (colors & expected) for _, colors in identifiable)
        if sh["offcast"]:
            print(f"  OFF-CAST shot {sh['index']:02d} ({sh['start']:.2f}-{sh['end']:.2f}): "
                  f"{'; '.join(sh['cast'])[:80]}")
    return shots


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--engine", choices=["pyscene", "ffmpeg"], default="pyscene",
                    help="pyscene = accurate (default); ffmpeg = zero-install rough pass")
    ap.add_argument("--threshold", type=float, default=None,
                    help="detector sensitivity; lower finds more cuts "
                         "(default 27 for pyscene, 8 for ffmpeg)")
    ap.add_argument("--shots-json", default=None,
                    help="reuse boundaries from an existing shots.json instead of detecting; "
                         "use so the board, the descriptions and the clauses share one numbering")
    ap.add_argument("--shots-scale", type=float, default=1.0,
                    help="multiply reused boundaries by this (1x list -> 1.5x clip = 1/1.5)")
    ap.add_argument("--min-shot", type=float, default=0.25,
                    help="shots shorter than this are merged into the previous one (seconds)")
    ap.add_argument("--describe", action="store_true", help="run the VLM description pass")
    ap.add_argument("--camera", action="store_true",
                    help="measure per-shot camera movement (pan/tilt/push) from motion vectors; "
                         "the VLM cannot do this from one frame")
    ap.add_argument("--motion", action="store_true",
                    help="measure per-shot motion energy (inter-frame luma delta); use to "
                         "compare a sped-up reference against its original, or an output "
                         "against another output")
    ap.add_argument("--subject", action="store_true",
                    help="measure per-shot SUBJECT screen-direction (where bodies get driven in "
                         "frame after camera is removed), impact reversals and hit-stop; run with "
                         "--motion so hit-stop has a motion curve to read")
    ap.add_argument("--model", default="qwen3-vl:8b-instruct-q8_0")
    ap.add_argument("--ollama-host", default="http://localhost:11434")
    ap.add_argument("--expect-cast", default="",
                    help="pipe-separated descriptors of the fighters we intend to swap, e.g. "
                         "\"red hair|white hair\". Shots showing only other people are flagged "
                         "offcast=true so they can be dropped from the swap window.")
    a = ap.parse_args()

    clip, out_dir = Path(a.clip), Path(a.out_dir)
    if not clip.exists():
        sys.exit(f"clip not found: {clip}")
    out_dir.mkdir(parents=True, exist_ok=True)

    duration = probe_duration(clip)
    threshold = a.threshold if a.threshold is not None else (27.0 if a.engine == "pyscene" else 8.0)
    if a.shots_json:
        # Reuse an existing shot list instead of re-detecting. The storyboard
        # panels, the shot descriptions and the direction clauses must all carry
        # the SAME numbering, and detection does not reproduce it across a speed
        # change: the 1.5x copy of a clip loses short cuts whose frames the
        # decimation drops, so 1x found 11 shots where 1.5x found 10 and every
        # shot after the first was off by one.
        src = json.loads(Path(a.shots_json).read_text())
        src = src["shots"] if isinstance(src, dict) else src
        k = a.shots_scale
        shots = []
        for i, sh in enumerate(src):
            start, end = sh["start"] * k, min(sh["end"] * k, duration)
            if end - start <= 0:
                continue
            shots.append({"index": i + 1, "start": start, "end": end,
                          "dur": end - start, "score": sh.get("score", 0.0)})
        detail = f"shots-json={a.shots_json} scale={k}"
    elif a.engine == "pyscene":
        shots = detect_shots_pyscene(clip, threshold, a.min_shot)
        detail = f"engine=pyscene threshold={threshold}"
    else:
        hits = detect_cuts(clip, threshold)
        shots = cluster(hits, a.min_shot, duration)
        detail = f"engine=ffmpeg threshold={threshold} raw_detections={len(hits)}"
    shots = extract_frames(clip, shots, out_dir, duration)
    print(f"{clip.name}: {duration:.2f}s -> {len(shots)} shots "
          f"({detail}, min-shot={a.min_shot}s)")
    for sh in shots:
        print(f"  SHOT {sh['index']:02d}  {sh['start']:6.2f}-{sh['end']:6.2f}  {sh['dur']:5.2f}s")

    overall_motion = None
    if a.motion:
        print("\nMeasuring motion energy...")
        shots, overall_motion = motion_energy(clip, shots, out_dir)

    if a.camera:
        print("\nMeasuring camera movement...")
        shots = camera_motion(clip, shots, out_dir)

    if a.subject:
        print("\nMeasuring subject screen-direction / impact weight...")
        shots = subject_motion(clip, shots, out_dir)

    if a.describe:
        print("\nDescribing shots via VLM...")
        shots = describe(shots, a.model, a.ollama_host)
        if a.expect_cast:
            tokens = [t.strip().lower() for t in a.expect_cast.split("|") if t.strip()]
            print(f"\nChecking cast against {tokens}...")
            shots = flag_offcast(shots, tokens)
            n = sum(1 for s in shots if s.get("offcast"))
            print(f"  {n} of {len(shots)} shots flagged off-cast"
                  + (" (none)" if not n else " — drop these from the swap window"))

    result = {"clip": str(clip), "duration": round(duration, 3), "engine": a.engine,
              "threshold": threshold, "min_shot": a.min_shot,
              "shot_count": len(shots), "motion_overall": overall_motion, "shots": shots}
    (out_dir / "shots.json").write_text(json.dumps(result, indent=2))
    print(f"\nWrote {out_dir/'shots.json'}")


if __name__ == "__main__":
    main()

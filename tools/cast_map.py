#!/usr/bin/env python3
"""Cast map -- who is in every shot of a reference trailer, and how many of them.

WHAT IT ANSWERS
---------------
For each shot of a trailer we are copying the STRUCTURE of: how many people are on screen,
which of them are recurring principals, what generic extras are present, and whether there is
a creature or a crowd. `principal_count` (0 / 1 / 2 / 3+) is the pipeline routing key -- it
decides whether a shot needs no character ref, one, or the multi-ref path.

Designed against docs/HANDOFF_20260824_trailer_pivot.md, section "Cast map -- to build".

THE TWO TRAPS THIS TOOL IS BUILT AROUND
---------------------------------------
1. MIDPOINT-ONLY SAMPLING MISSES THINGS. The storyboard leak (G9, tools/board_leak.py) lived in
   the TAIL of a shot and identity_gate missed it by 0.11s because it samples shot midpoints.
   So: N frames per shot at (k+0.5)/N of the span -- with the default 5 that is 10%..90%, which
   actually looks at the ends.
2. THE VLM HALLUCINATES ON BLACK. On the test cinematic it reported "fist strike, blood" for
   three near-black frames of a text card. So: any frame whose mean luma is below --black-luma
   (default 12) is marked EMPTY and is NEVER sent to the VLM. Empty frames contribute nothing
   to any count.

HOW A PERSON IS COUNTED
-----------------------
    people(frame) = max(face_count, vlm_people)
Faces come from OpenCV's YuNet detector; `vlm_people` from one temperature-0 JSON call per
non-empty frame. Neither alone is enough: YuNet cannot see a helmeted knight, a masked figure
or a dragon (it finds NO face in most of the test-cinematic hero's shots), and the VLM cannot tell one
person from another across shots. Faces give identity, the VLM gives presence.

IDENTITY
--------
Every detected face is embedded to 128-d with SFace and all embeddings are clustered across the
whole trailer by cosine similarity (average-link agglomerative, threshold 0.363 -- the value
opencv_zoo publishes for SFace). A cluster appearing in >= --min-principal-frac of the shots is
a PRINCIPAL CANDIDATE. One VLM call per cluster, on its 3 sharpest crops, labels it principal
vs generic extra and writes a descriptor.

Crop sharpness is scored locally (Laplacian variance x area), not by the VLM: unlike
tools/pick_char_ref.py we are only choosing which crops to SHOW, not judging content, so a
free local proxy is the right tool.

★ AND THE DETECTOR LIES ON ARMOUR. On the test cinematic YuNet returned five detections in 201
seconds and the owner confirmed by eye that NOT ONE was a face -- they are close-ups of plate
armour and wide shots of terrain. So every cluster is put to the VLM as "is this actually a
face?" before it is allowed to contribute anything; a rejected cluster names no extra and can
never be a principal. Without that check the phantom clusters were writing invented descriptors
("blurred face in blue light") into four shots' extras.

DEFAULTS SET ON THE PLANNING SIDE, 2026-08-25 -- NOT YET OWNER-REVIEWED
-----------------------------------------------------------------------
Five behaviours below were chosen by the planning side after the owner adjudicated the test-cinematic
validation. The owner confirmed the EVIDENCE (see below); he has not reviewed these five
choices. They are defaults, and they are reversible.

  1. THE FACE PATH IS OFF BY DEFAULT (--faces turns it on). Our heroes are armoured and
     helmeted as the normal case, and on that footage YuNet returns phantoms: five detections
     across 201 seconds of the test cinematic, and the owner confirmed by eye that NOT ONE
     was a face -- they are close-ups of plate armour and wide shots of terrain. The is_face
     confirmation gate below stays, for when the path is switched on.
  2. A CROWD IS ITS OWN ROUTING CLASS. principal_count stays numeric for figures that stand
     apart; a shot whose figures are an undifferentiated mass reports routing "crowd" and a
     principal_count of however many figures are in the FOREGROUND (0 for an army alone).
     Head-counting an army into "3+ principals" would route a mass of extras down the
     multi-character render path.
  3. FLAGS FIRE ON ANY CLEAN FRAME, with a 0.15s cut guard. See THE CUT GUARD below.
  4. Shot 16 of the test cinematic is the hero (owner: "overhead shot with a small figure in
     the centre of cracking ice"). NOTE FOR THE WRITER: the trailer spec deliberately renders
     that shot as VFX with no character, because identity is unreadable at that scale. The
     cast map saying "1 person" is correct and the spec staying as it is, is also correct.
  5. The two-person positive control was run on output/fullbattle_20260808/seg*_15x.mp4
     against config/board_rosters.json -- see docs and the run report for its numbers.

THE CUT GUARD, AND ONE PLACE THE PLAN DID NOT SURVIVE MEASUREMENT
-----------------------------------------------------------------
A frame sampled just after a cut still shows the PREVIOUS shot. Shot 25's first frame reported
six people and a crowd -- that is shot 26's army, not shot 25's dragon. So every per-shot
statistic ignores frames inside 0.15s of the shot's start: the same head-only guard, and the
same "if that leaves nothing, use everything" fallback, that tools/shot_breakdown.motion_energy
uses, for the same reason it gives.

With the contaminated frames gone, a flag can safely fire on ANY remaining frame -- and it must,
because the owner confirmed that shot 18 (a dragon's arm breaking up through the ice) and shot
28 (a dragon over the undead army) are both real despite appearing in only 1-2 frames of five.

The planning note also asked for "shots under 1.0s: sample only the centre frame", to catch shot
28. Measured, that rule LOSES shot 28: its centre frame is the dragon, and the army enters in
the shot's final 0.04 seconds. The head guard alone catches 28 and leaves 25 clean, which is
what the rule was for, so the short-shot behaviour here is the motion pass's own fallback
instead. Recorded rather than quietly substituted.

AND THE ANY-FRAME RULE IS A TRADE, NOT A FREE WIN. Measured on the owner-confirmed test-cinematic truth,
with the cut guard already applied (--flag-frames):

    1 clean frame (default)   creature P 0.82 R 1.00   crowd P 0.60 R 1.00   catches shot 28
    2 clean frames            creature P 1.00 R 0.89   crowd P 1.00 R 0.67   misses shot 28
    3 clean frames            creature P 1.00 R 0.78   crowd P 1.00 R 0.67   misses shot 28

The default is 1 because the planning side asked for shot 28 to be caught, and a missed dragon
costs more than a false one when a human confirms the roster by eye anyway. But 1 costs four
false flags on 33 shots (creature on 19 and 26, crowd on 25 and 30) and that is not noise -- it
is the price of the rule. Whoever owns the routing key should pick; the knob is there.

The head guard is the same shape of trade on the people count: with it, P 1.00 / R 0.94 (shot 13,
a 0.5s whip-pan fragment, loses its only good frame); without it, P 0.95 / R 1.00 (shot 25 gains
a phantom person from its neighbour's army). The guard is on because a phantom PERSON routes a
shot to a character render, and a missed one in a fragment that the trim plan merges away does
not.

THE FALLBACK, AND ITS HONEST LIMIT
----------------------------------
When a shot has people but no face clustered into a principal, `principal_count` falls back to
the frame people-count and `count_source` says "vlm_fallback" instead of "faces". That is a
COUNT, not an identity: it cannot tell you the armoured figure in shot 5 is the same armoured
figure as in shot 21. Every faceless-hero trailer will run entirely on the fallback. Read
`count_source` before trusting `principals`.

MODELS (downloaded once into models/opencv_zoo/, ~38 MB, git-ignored)
    https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
    https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx
    (the raw.githubusercontent URLs return a 131-byte git-lfs POINTER, not the model -- use the
     media.githubusercontent.com/media/ form.)
Both run on the CPU. Face work needs no GPU claim at all.

★ GPU CLAIM -- REQUIRED FOR EVERY VLM RUN
-----------------------------------------
Run every VLM pass under tools/gpu_claim.sh, which takes the lock, evicts resident ollama
models and makes ComfyUI drop its cache. Skipping it is not a queueing nicety: a vision model
that loads split across the card and the CPU gives DIFFERENT ANSWERS on the same frame
(measured 2026-08-19 -- "drawing" 4/4 on the card vs "photograph" 15/15 part-CPU).

gpu_claim.sh sets no environment variable, so this script detects a claimed run three ways and
refuses to make a VLM call unless one of them holds:
  * it inherits gpu_claim.sh's open lock file on fd 9 (automatic -- nothing to pass), or
  * GPU_CLAIMED=1 is set in the environment, or
  * --i-am-under-gpu-claim is passed (escape hatch for odd shells).
--no-vlm runs faces-only and needs no claim.

USAGE
    tools/gpu_claim.sh .venv/bin/python3 tools/cast_map.py CLIP --shots shots.json --out-dir DIR
        [--faces] [--frames-per-shot 5] [--model qwen3-vl:8b-instruct-q8_0]
        [--ollama-host http://localhost:11434] [--min-principal-frac 0.15] [--no-vlm]

OUTPUT
    DIR/cast_map.json     per-shot cast + the cluster roster
    DIR/cast_roster.html  contact sheet: one row per cluster with 6 crops and a checkbox, and
                          the per-shot table below it. The owner confirms the roster BY EYE --
                          that is deliberate, not a gap (docs/anime_swap_recipe.md §7).
"""
import argparse
import base64
import html
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
YUNET = ROOT / "models/opencv_zoo/face_detection_yunet_2023mar.onnx"
SFACE = ROOT / "models/opencv_zoo/face_recognition_sface_2021dec.onnx"

# opencv_zoo's published SFace operating point. Above this cosine similarity, same person.
COSINE_SAME = 0.363
BLACK_LUMA = 12.0          # mean grey level below which a frame is "empty" -- see docstring
CUT_GUARD = 0.15           # seconds after a cut whose frames still show the PREVIOUS shot


# ---------------------------------------------------------------- gpu claim

def under_gpu_claim(flag: bool) -> bool:
    """True when this process is running under tools/gpu_claim.sh.

    gpu_claim.sh does `exec 9>output/state/gpu.lock; flock 9; exec "$@"`, so a claimed child
    INHERITS fd 9 pointing at the lock file. That is a fact on disk we can check, which beats
    asking the caller to promise -- the same move that fixed G3 and G9.
    """
    if flag or os.environ.get("GPU_CLAIMED") == "1":
        return True
    try:
        return os.readlink("/proc/self/fd/9").endswith("gpu.lock")
    except OSError:
        return False


# ---------------------------------------------------------------- frames

def frame_at(clip: Path, t: float, out: Path, duration: float) -> bool:
    """Write one frame at t. Walks backwards if the seek lands past the last decodable frame.

    Lifted from shot_breakdown.extract_frames (L107): a detector's last shot can end a hair
    past the real end of the video and a naive seek then writes no file at all.
    """
    base = min(t, max(0.0, duration - 0.05))
    for ts in (base, max(0.0, base - 0.08), max(0.0, base - 0.25), max(0.0, base - 0.5)):
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{ts:.3f}", "-i", str(clip),
                        "-frames:v", "1", "-q:v", "3", str(out)], capture_output=True)
        if out.exists() and out.stat().st_size > 0:
            return True
    return False


def probe_duration(clip: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "csv=p=0", str(clip)], capture_output=True, text=True)
    try:
        return float(out.stdout.strip())
    except ValueError:
        return 0.0


def sample_times(shot, n: int):
    """n samples evenly across the shot at (k+0.5)/n -- 10%..90% for n=5.

    NEVER the midpoint alone. See trap 1 in the module docstring.
    """
    span = max(0.0, shot["end"] - shot["start"])
    return [shot["start"] + span * (k + 0.5) / n for k in range(n)]


# ---------------------------------------------------------------- faces

def load_face_models():
    if not YUNET.exists() or not SFACE.exists():
        sys.exit("missing ONNX models -- see the URLs in this file's docstring:\n  %s\n  %s"
                 % (YUNET, SFACE))
    det = cv2.FaceDetectorYN.create(str(YUNET), "", (320, 320), 0.6, 0.3, 5000)
    rec = cv2.FaceRecognizerSF.create(str(SFACE), "")
    return det, rec


def sharpness(crop) -> float:
    """Laplacian variance x area. Higher = a bigger, crisper crop; only used to RANK crops."""
    if crop is None or crop.size == 0:
        return 0.0
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    return float(cv2.Laplacian(g, cv2.CV_64F).var() * (g.shape[0] * g.shape[1]) ** 0.5)


def detect_faces(det, rec, img, crop_path_stem: Path):
    """Detect, align-crop, embed. Returns a list of dicts, one per face."""
    h, w = img.shape[:2]
    det.setInputSize((w, h))
    n, faces = det.detect(img)
    if faces is None:
        return []
    out = []
    for i, f in enumerate(faces):
        try:
            aligned = rec.alignCrop(img, f)
            feat = rec.feature(aligned)
        except cv2.error:
            continue
        p = crop_path_stem.with_name(crop_path_stem.name + "_face%d.jpg" % i)
        # Save the raw box with a little padding -- more readable for the owner than the
        # 112x112 aligned patch the recogniser uses.
        x, y, bw, bh = [int(v) for v in f[:4]]
        pad = int(0.35 * max(bw, bh))
        box = img[max(0, y - pad):min(h, y + bh + pad), max(0, x - pad):min(w, x + bw + pad)]
        if box.size:
            cv2.imwrite(str(p), box)
        out.append({"crop": str(p), "score": float(f[-1]), "box": [x, y, bw, bh],
                    "sharp": sharpness(aligned), "feat": np.asarray(feat).ravel()})
    return out


def cluster_faces(feats, thresh=COSINE_SAME):
    """Average-link agglomerative clustering on cosine similarity. Returns a label per face.

    Small n (a trailer yields tens of faces, not thousands), so the naive O(n^2) merge loop is
    both fast enough and easy to audit -- which matters more here than speed.
    """
    n = len(feats)
    if n == 0:
        return []
    m = np.stack([f / (np.linalg.norm(f) + 1e-9) for f in feats])
    sim = m @ m.T
    labels = list(range(n))
    members = {i: [i] for i in range(n)}
    while True:
        best, pair = thresh, None
        for a in list(members):
            for b in list(members):
                if b <= a:
                    continue
                s = float(sim[np.ix_(members[a], members[b])].mean())   # average link
                if s > best:
                    best, pair = s, (a, b)
        if pair is None:
            break
        a, b = pair
        members[a] += members.pop(b)
    for new, (root, idxs) in enumerate(sorted(members.items())):
        for i in idxs:
            labels[i] = new
    return labels


# ---------------------------------------------------------------- vlm

FRAME_PROMPT = (
    "Look at this single film frame and report only what is actually visible. Reply with JSON "
    'only, no prose, with exactly these keys: "people" (integer: how many distinct HUMAN or '
    "humanoid figures are visible, including armoured, helmeted, masked or hooded figures whose "
    "faces you cannot see; count 0 if none; if there is an indistinct crowd give your best "
    'estimate of individually visible figures), "creature" (true if a non-human creature -- '
    "dragon, beast, monster, mount -- is visible, otherwise false), "
    '"crowd" (true if a large group, army or mass of figures is visible, otherwise false), '
    '"foreground_people" (integer: of those figures, how many stand apart as distinct '
    "individuals in the foreground -- a named-character shot -- as opposed to being part of an "
    "undifferentiated mass; if everyone visible is part of the crowd, answer 0), "
    '"descriptors" (array of short visual descriptions, one per visible figure, e.g. '
    '["armoured figure in dark plate with horned helm"]). '
    "If the frame is empty, black, a text card or pure landscape, answer people 0, creature "
    "false, crowd false, foreground_people 0, descriptors []. Do not name any film, game, character or franchise."
)

CLUSTER_PROMPT = (
    "These images were cropped by an automatic face detector from different shots of one film "
    "trailer. The detector is NOT reliable: on dark or armoured footage it fires on plate armour, "
    "rock, ice and background texture, which contain no face at all. "
    "Reply with JSON only, no prose, with exactly these keys: "
    '"is_face" (true ONLY if these crops actually show a human or humanoid FACE -- eyes, nose and '
    "mouth discernible; false if they are armour, helmet plating, rock, ice, machinery, blur or "
    'any other texture), "principal" (true if this looks like a named lead character the trailer '
    "is about, false if it is a generic extra, background soldier, bystander or crowd member), "
    '"descriptor" (one short visual phrase identifying them, e.g. "young woman, red hair, '
    'green cloak"), "role_guess" (a couple of words, e.g. "hero", "villain", "soldier", '
    '"bystander"). Do not name any film, game, character or franchise.'
)


def vlm(host: str, model: str, prompt: str, images, timeout=300):
    """One temperature-0 JSON call to ollama's HTTP API.

    HTTP, never `ollama run <prompt> <image>` -- the CLI form silently drops the image and
    answers from the text alone (verified 2026-08-08). temperature 0 as in identity_gate._post.
    """
    b64 = [base64.b64encode(Path(p).read_bytes()).decode() for p in images]
    body = json.dumps({"model": model, "prompt": prompt, "images": b64, "stream": False,
                       "format": "json", "options": {"temperature": 0}}).encode()
    req = urllib.request.Request(host.rstrip("/") + "/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        raw = json.load(urllib.request.urlopen(req, timeout=timeout))["response"]
        return json.loads(raw)
    except Exception as e:                       # noqa: BLE001 -- an unreadable answer is data
        return {"_unparsed": str(e)[:200]}


# ---------------------------------------------------------------- html

def write_html(out_dir: Path, clip: Path, shots, clusters, args):
    def rel(p):
        try:
            return html.escape(str(Path(p).relative_to(out_dir)))
        except ValueError:
            return html.escape(str(p))

    rows = []
    for c in clusters:
        crops = "".join('<img src="%s">' % rel(p) for p in c["exemplar_crops"][:6])
        kind = ("PRINCIPAL" if c["principal"] else
                "NOT A FACE - discarded" if c["is_face"] is False else
                "face unverified (--no-vlm)" if c["is_face"] is None else "extra")
        rows.append(
            '<tr><td><input type="checkbox" %s></td><td><b>c%d</b><br>%s</td>'
            '<td>%d frames<br>%d shots</td><td>%s<br><i>%s</i></td><td class="crops">%s</td></tr>'
            % ("checked" if c["principal"] else "", c["id"], kind, c["n_frames"], c["n_shots"],
               html.escape(c["descriptor"] or "-"), html.escape(c["role_guess"] or "-"), crops))

    srows = []
    for s in shots:
        flags = " ".join(x for x in [
            "creature" if s["creature"] else "", "crowd" if "crowd" in s["extras"] else ""] if x)
        srows.append(
            "<tr><td>%d</td><td>%.1f-%.1f</td><td>%d</td><td>%s</td><td>%s</td><td>%s</td>"
            "<td>%s</td><td>%s</td><td>%d/%d empty</td></tr>"
            % (s["index"], s["start"], s["end"], s["people_count"],
               ", ".join("c%d" % i for i in s["principals"]) or "-",
               html.escape(", ".join(s["extras"]) or "-"),
               html.escape(str(s["principal_count"])), html.escape(s["routing"]),
               html.escape(s["count_source"]),
               sum(1 for f in s["frames"] if f["empty"]), len(s["frames"])))

    (out_dir / "cast_roster.html").write_text("""<!doctype html><meta charset="utf-8">
<title>Cast roster - %s</title>
<style>
 body{font:14px/1.4 system-ui,sans-serif;margin:24px;background:#111;color:#eee}
 h1,h2{font-weight:600} table{border-collapse:collapse;margin:12px 0 28px}
 td,th{border:1px solid #444;padding:6px 8px;vertical-align:top}
 th{background:#222;text-align:left} .crops img{height:96px;margin-right:4px;border:1px solid #555}
 .note{color:#bbb;max-width:70em}
</style>
<h1>Cast roster &mdash; %s</h1>
<p class="note">Tick the clusters that are <b>principals</b> (a named lead), untick generic
extras. %d shots, %d frames per shot, principal candidacy threshold %.0f%% of shots.
Face identity is unavailable for helmeted or masked figures &mdash; check the
<code>source</code> column below: <code>vlm_fallback</code> means the count is a head-count with
<b>no identity behind it</b>.</p>
<h2>Clusters</h2>
<table><tr><th>principal?</th><th>id</th><th>seen</th><th>label</th><th>crops</th></tr>%s</table>
<h2>Per shot</h2>
<table><tr><th>#</th><th>t</th><th>people</th><th>principals</th><th>extras</th>
<th>principal_count</th><th>routing</th><th>source</th><th>frames</th></tr>%s</table>
""" % (html.escape(clip.name), html.escape(clip.name), len(shots), args.frames_per_shot,
       args.min_principal_frac * 100,
       "".join(rows) or "<tr><td colspan=5>no faces detected anywhere</td></tr>",
       "".join(srows)))


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(
        description="Cast map: people, principals, extras and creatures per shot.",
        epilog="★ Every VLM run MUST go through tools/gpu_claim.sh -- a vision model split "
               "between the card and the CPU returns different answers on the same frame. This "
               "script refuses to make a VLM call unless it detects gpu_claim.sh's inherited "
               "lock on fd 9, or GPU_CLAIMED=1, or --i-am-under-gpu-claim. --no-vlm needs none "
               "of that.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip")
    ap.add_argument("--shots", required=True, help="shots.json from tools/shot_breakdown.py")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--frames-per-shot", type=int, default=5)
    ap.add_argument("--model", default="qwen3-vl:8b-instruct-q8_0")
    ap.add_argument("--ollama-host", default="http://localhost:11434")
    ap.add_argument("--min-principal-frac", type=float, default=0.15,
                    help="a cluster in at least this fraction of shots is a principal candidate")
    ap.add_argument("--flag-frames", type=int, default=1,
                    help="how many CLEAN frames must see a creature or a crowd for the shot to "
                         "be flagged. 1 (default, set on the planning side) maximises recall; 2 "
                         "gave perfect precision on the test-cinematic truth but missed shot 28. The "
                         "trade is measured in this file's docstring -- it is a judgement call, "
                         "not a tuning exercise.")
    ap.add_argument("--black-luma", type=float, default=BLACK_LUMA,
                    help="mean luma below which a frame is EMPTY and never sent to the VLM")
    ap.add_argument("--faces", action="store_true",
                    help="also run YuNet/SFace face detection and clustering. OFF by default: "
                         "on armoured footage the detector returns only phantoms (0 real faces "
                         "in 5 detections on the test cinematic, owner-verified).")
    ap.add_argument("--no-vlm", action="store_true", help="faces only; no GPU claim needed")
    ap.add_argument("--i-am-under-gpu-claim", action="store_true",
                    help="assert this run is under tools/gpu_claim.sh (fd-9 detection failed)")
    args = ap.parse_args()

    if args.frames_per_shot < 1:
        sys.exit("--frames-per-shot must be >= 1")
    use_vlm = not args.no_vlm
    if args.no_vlm and not args.faces:
        sys.exit("--no-vlm with the face path off leaves nothing to measure. Add --faces.")
    if use_vlm and not under_gpu_claim(args.i_am_under_gpu_claim):
        sys.exit("★ REFUSING TO RUN THE VLM UNCLAIMED.\n"
                 "  Re-run as: tools/gpu_claim.sh .venv/bin/python3 tools/cast_map.py ...\n"
                 "  (or set GPU_CLAIMED=1 / pass --i-am-under-gpu-claim if you know the card is "
                 "yours, or use --no-vlm for faces only)")

    clip = Path(args.clip)
    out_dir = Path(args.out_dir)
    (out_dir / "frames").mkdir(parents=True, exist_ok=True)
    (out_dir / "crops").mkdir(parents=True, exist_ok=True)

    raw = json.loads(Path(args.shots).read_text())
    shots_in = raw["shots"] if isinstance(raw, dict) else raw
    duration = probe_duration(clip)
    det, rec = load_face_models() if args.faces else (None, None)

    all_faces = []          # flat list; each keeps its shot index and frame time
    shots_out = []
    for si, sh in enumerate(shots_in):
        idx = int(sh.get("index", si + 1))
        guard_until = float(sh["start"]) + CUT_GUARD
        rec_frames = []
        for k, t in enumerate(sample_times(sh, args.frames_per_shot)):
            fp = out_dir / "frames" / ("shot_%02d_f%d.jpg" % (idx, k + 1))
            entry = {"t": round(t, 3), "empty": True, "guarded": t < guard_until,
                     "faces": 0, "vlm_people": None, "foreground_people": None,
                     "creature": False, "crowd": False, "descriptors": []}
            if not frame_at(clip, t, fp, duration):
                entry["note"] = "unextractable"
                rec_frames.append(entry)
                continue
            img = cv2.imread(str(fp))
            if img is None:
                entry["note"] = "unreadable"
                rec_frames.append(entry)
                continue
            luma = float(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean())
            entry["luma"] = round(luma, 2)
            if luma < args.black_luma:
                # NEAR-BLACK: no faces, no VLM. See trap 2 in the docstring.
                rec_frames.append(entry)
                continue
            entry["empty"] = False

            faces = (detect_faces(det, rec, img,
                                  out_dir / "crops" / ("shot_%02d_f%d" % (idx, k + 1)))
                     if det else [])
            entry["faces"] = len(faces)
            for f in faces:
                f["shot"] = idx
                f["t"] = round(t, 3)
                all_faces.append(f)

            if use_vlm:
                v = vlm(args.ollama_host, args.model, FRAME_PROMPT, [fp])
                if "_unparsed" in v:
                    entry["vlm_error"] = v["_unparsed"]
                else:
                    try:
                        entry["vlm_people"] = int(v.get("people") or 0)
                    except (TypeError, ValueError):
                        entry["vlm_people"] = 0
                    entry["creature"] = bool(v.get("creature"))
                    entry["crowd"] = bool(v.get("crowd"))
                    try:
                        entry["foreground_people"] = int(v.get("foreground_people") or 0)
                    except (TypeError, ValueError):
                        entry["foreground_people"] = 0
                    entry["descriptors"] = [str(d)[:80] for d in (v.get("descriptors") or [])]
            rec_frames.append(entry)

        # ★ THE CUT GUARD. A frame sampled just after a cut still shows the PREVIOUS shot on
        # this footage: shot 25's first frame reported six people and a crowd, which is shot
        # 26's army, not shot 25's dragon. Same 0.15s head guard the motion pass uses, and
        # head-only for the same reason it gives -- guarding the tail as well throws away
        # everything on a sub-second shot. If the guard leaves nothing, fall back to all live
        # frames rather than reporting a shot as unmeasured.
        live = [f for f in rec_frames if not f["empty"]]
        clean = [f for f in live if not f["guarded"]] or live
        counts = [max(f["faces"], f["vlm_people"] or 0) for f in clean]
        # MEDIAN, not max: one hallucinated frame should not set a shot's head-count. An
        # object genuinely on screen for the shot shows up in most of its frames.
        people = int(np.median(counts)) if counts else 0
        # ANY clean frame raises a flag. A dragon that breaks through the ice halfway, or an
        # army that enters at the end, is present in the shot however few frames caught it --
        # and the contaminated frames have already been removed above, which is what makes
        # any-frame safe here. (Owner-verified 2026-08-25: shots 18 and 28 are both real.)
        creature = sum(1 for f in clean if f["creature"]) >= args.flag_frames
        crowd = sum(1 for f in clean if f["crowd"]) >= args.flag_frames
        fg = [f["foreground_people"] for f in clean if f["foreground_people"] is not None]

        shots_out.append({
            "index": idx, "start": float(sh["start"]), "end": float(sh["end"]),
            "frames": rec_frames, "people_count": people,
            "principals": [], "extras": [], "creature": creature, "_crowd": crowd,
            "_fg": int(np.median(fg)) if fg else 0,
            "principal_count": 0, "count_source": "none", "routing": "empty",
            "_descriptors": sorted({d for f in clean for d in f["descriptors"]})[:6],
        })
        print("  shot %02d %.1f-%.1fs people=%d faces=%d empty=%d/%d%s%s"
              % (idx, sh["start"], sh["end"], people, sum(f["faces"] for f in rec_frames),
                 len(rec_frames) - len(live), len(rec_frames),
                 " CREATURE" if creature else "", " CROWD" if crowd else ""), flush=True)

    # ---- cluster the faces across the whole trailer
    labels = cluster_faces([f["feat"] for f in all_faces])
    for f, lab in zip(all_faces, labels):
        f["cluster"] = lab
    clusters = []
    n_shots = len(shots_out)
    for cid in sorted(set(labels)):
        mem = [f for f in all_faces if f["cluster"] == cid]
        shot_ids = sorted({f["shot"] for f in mem})
        best = sorted(mem, key=lambda f: -f["sharp"])
        candidate = len(shot_ids) >= max(1, args.min_principal_frac * n_shots)
        label = {"is_face": None, "principal": False, "descriptor": "", "role_guess": ""}
        if use_vlm:
            crops = [f["crop"] for f in best[:3] if Path(f["crop"]).exists()]
            if crops:
                v = vlm(args.ollama_host, args.model, CLUSTER_PROMPT, crops)
                if "_unparsed" not in v:
                    label = {"is_face": bool(v.get("is_face")),
                             "principal": bool(v.get("principal")),
                             "descriptor": str(v.get("descriptor", ""))[:120],
                             "role_guess": str(v.get("role_guess", ""))[:60]}
        clusters.append({
            "id": cid, "n_frames": len(mem), "n_shots": len(shot_ids), "shots": shot_ids,
            "candidate": candidate,
            # ★ CONFIRMED BY THE OWNER'S EYE, 2026-08-25: on the test cinematic YuNet returned
            # FIVE detections and not one of them was a face -- they are close-ups of plate
            # armour and wide shots of terrain. A detector firing on texture is not a rare edge
            # case here, it is what it does on dark armoured footage, and the phantom clusters
            # were leaking their invented descriptors ("blurred face in blue light") into four
            # shots' extras. So a cluster must be CONFIRMED as a face before it is allowed to
            # say anything about a shot. is_face is None under --no-vlm: unverified, and the
            # roster HTML says so rather than pretending.
            "is_face": label["is_face"],
            # A one-shot face is never a principal however confident the VLM sounds: presence
            # across the piece is the evidence, the VLM only labels what recurs.
            "principal": bool(candidate and label["principal"] and label["is_face"] is not False),
            "descriptor": label["descriptor"], "role_guess": label["role_guess"],
            "exemplar_crops": [f["crop"] for f in best[:6] if Path(f["crop"]).exists()],
        })
    principal_ids = {c["id"] for c in clusters if c["principal"]}
    by_shot = {}
    for f in all_faces:
        by_shot.setdefault(f["shot"], set()).add(f["cluster"])

    for s in shots_out:
        here = by_shot.get(s["index"], set())
        s["principals"] = sorted(here & principal_ids)
        extras = []
        for cid in sorted(here - principal_ids):
            c = clusters[cid]
            if c["is_face"] is False:
                continue            # detector fired on armour or terrain -- not a person
            extras.append(c["descriptor"] or c["role_guess"] or "unlabelled face c%d" % cid)
        if s["_crowd"]:
            extras.append("crowd")
        if s["creature"]:
            extras.append("creature")
        s["extras"] = extras
        s["clusters_present"] = sorted(here)
        if s["principals"]:
            n = len(s["principals"])
            s["count_source"] = "faces"
        elif s["_crowd"]:
            # ★ A CROWD IS NOT A CAST OF PRINCIPALS. Head-counting an army and calling the
            # answer "3+ principals" routes a mass of extras down the multi-character path.
            # In a crowd shot the only number worth having is how many figures stand APART
            # from the mass, which is what foreground_people asks the model directly.
            n = s["_fg"]
            s["count_source"] = "vlm_foreground" if n else "crowd_only"
        else:
            # FALLBACK: people are on screen but no face was identified (helmets, masks, backs
            # of heads, distance). This is a HEAD-COUNT with no identity behind it.
            n = s["people_count"]
            s["count_source"] = "vlm_fallback" if n else "none"
        s["principal_count"] = "3+" if n >= 3 else n
        s["routing"] = ("crowd" if s["_crowd"] else
                        "creature" if s["creature"] and n == 0 else
                        "empty" if n == 0 else
                        "solo" if n == 1 else "duo" if n == 2 else "multi")
        s.pop("_crowd"); s.pop("_fg")

    write_html(out_dir, clip, shots_out, clusters, args)
    doc = {
        "clip": str(clip), "shots_json": str(args.shots), "n_shots": n_shots,
        "frames_per_shot": args.frames_per_shot, "model": args.model if use_vlm else None,
        "min_principal_frac": args.min_principal_frac, "black_luma": args.black_luma,
        "shots": shots_out,
        "clusters": clusters,
    }
    (out_dir / "cast_map.json").write_text(json.dumps(doc, indent=2))
    print("\n%d faces -> %d clusters; %d principal(s): %s"
          % (len(all_faces), len(clusters), len(principal_ids),
             ", ".join("c%d" % i for i in sorted(principal_ids)) or "none"))
    print("wrote %s and %s" % (out_dir / "cast_map.json", out_dir / "cast_roster.html"))


if __name__ == "__main__":
    main()

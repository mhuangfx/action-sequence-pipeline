#!/usr/bin/env python3
"""
qa_gate.py — Module B: local VLM publish gate (per docs/archive/implementation_plan_phase1.md).

Scores a master video on the 5-axis rubric (docs/archive/expansion_test_findings_2026-06-10.md
T0b) using local Ollama qwen2.5vl:7b (temperature 0, JSON-forced, defensive parsing).
stdlib only: urllib for Ollama HTTP (localhost:11434), subprocess for ffmpeg/ffprobe.

Usage:
  python3 qa_gate.py <video.mp4> [--json out.json] [--fps 1] [--seam-at <sec>|auto]
                     [--thresholds consistency=6,readability=6,artifacts=4,energy_arc=7,seam=6]
                     [--pick-cover] [--pick-cover-only] [--model qwen2.5vl:7b] [--verbose]

  --seam-at auto      = duration/2 (the 2x15s chain midpoint; make_video.js convenience)
  --pick-cover        = additionally choose the cover frame (candidates span the whole
                        video; one VLM call; selection objective rotates per-video for
                        channel-grid variety); printed + emitted as cover_t in --json output
  --pick-cover-only   = only pick the cover (no gate), exit 0

Exit codes: 0 PASS · 2 FAIL · 1 error.

Aggregation: sequential batches of ~8 time-labelled frames score readability /
artifacts (median across batches); one whole-video spread batch scores energy_arc;
character_consistency = min(batch median, DESCRIBE-THEN-JUDGE identity check) — 3
early + 3 late frames, the model LISTS protagonist-design differences with evidence
and the score is computed in code (a 7B lenience-inflates direct scores);
seam_quality is likewise computed in code from FACTUAL categorical answers about a
±1s straddle (same hero? same style? and the validated discriminator: is a huge new
figure in the first post-cut frame a COMPLETE standing creature = teleported in (bad)
vs an unformed erupting mass = born on screen (good escalation)).

CALIBRATION — DONE 2026-06-12 (qwen2.5vl:7b, temperature 0, two back-to-back matrix
runs, scores identical across runs; ~21-31s per video):
  chain30_v3.mp4          PASS  cons=9 read=9 art=9 arc=8 seam=8   (must-pass ✓)
  chain30_v1.mp4          FAIL  cons=10 read=10 art=10 arc=8 seam=4 (must-fail ✓, seam)
  duel_nox_3ref/video     PASS  cons=9 read=9 art=9 arc=8          (must-pass ✓)
  vertical_native/video   PASS  cons=10 read=10 art=10 arc=8       (must-pass ✓)
  bullettime_v1/video     PASS  cons=10 read=10 art=10 arc=9       (must-pass ✓)
  own_motion_test/video   PASS  cons=9 read=9 art=9 arc=8          (stretch must-fail ✗)
KNOWN LIMITATION: qwen2.5vl:7b cannot perceive own_motion_test's failure modes
(floaty/disconnected MOTION and between-shot face fluctuation — temporal artifacts
invisible in still frames; Gemini-flash refs scored it cons 4 / art 3). Acceptable in
production because the workflow that produced it (own outputs as @video1 motion refs)
is BANNED (docs/archive/expansion_test_findings_2026-06-10.md T3) — the gate covers the
failure modes that can occur in the pipeline: identity breaks, teleport-in seams,
energy-arc deflation. Enhancement path if needed: cloud-VLM (Gemini) artifacts/pacing
axis as a second-stage check.
Default thresholds (plan): consistency>=6, readability>=6, artifacts>=4,
energy_arc>=7, seam>=6.
"""
import argparse
import base64
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

OLLAMA_URL_DEFAULT = "http://localhost:11434"
MODEL_DEFAULT = "qwen2.5vl:7b"
BATCH_SIZE = 8
FRAME_SCALE = "scale='min(854,iw)':-2"  # keep vision-token cost sane, detail adequate
COVER_FRACS = [0.12, 0.28, 0.45, 0.62, 0.80, 0.94]  # span the whole video, not just the finale
# Rotated per-video (deterministic, see pick_cover) so the CHANNEL GRID shows varied
# compositions instead of every cover converging on the climactic VFX blast — the
# "central glowing blob + tiny silhouette" shot that reads as a templated grid.
COVER_OBJECTIVES = [
    "the frame where the HERO IS LARGEST and clearest — her face/body and expression "
    "prominent and sharp, reads as a character portrait, NOT a small silhouette lost in VFX",
    "the most dynamic readable ACTION beat — a clear mid-motion fighting pose or impact "
    "with the hero's full body visible and the motion legible (not a static or pure-blast frame)",
    "the single most striking POWER/VFX moment — but with the hero still clearly visible "
    "within it, not swallowed by the blast",
]
# Narrow straddle: ±1s only. The discriminator is the FIRST moments after the cut —
# a wider window catches a deliberately-born escalation foe (chain30_v3's titan erupts
# ~2.4s post-cut) and misreads it as "abruptly present".
SEAM_OFFSETS = [-1.0, -0.25, 0.25, 1.0]  # seconds around the cut

DEFAULT_THRESHOLDS = {
    "character_consistency": 6,
    "action_readability": 6,
    "artifacts": 4,
    "energy_arc": 7,
    "seam_quality": 6,
}
# short aliases accepted on the CLI
AXIS_ALIASES = {
    "consistency": "character_consistency",
    "readability": "action_readability",
    "seam": "seam_quality",
    "energy": "energy_arc",
}

VERBOSE = False


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def vlog(msg):
    if VERBOSE:
        log(msg)


def die(msg, code=1):
    log(f"ERROR: {msg}")
    sys.exit(code)


# ---------------------------------------------------------------- ffmpeg/ffprobe

def probe_duration(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", path],
        capture_output=True, text=True)
    if out.returncode != 0:
        die(f"ffprobe failed on {path}: {out.stderr.strip()}")
    try:
        return float(out.stdout.strip())
    except ValueError:
        die(f"ffprobe returned no duration for {path}")


def extract_fps_frames(path, fps, tmpdir):
    """Sample at --fps into tmpdir; returns [(t_seconds, filepath), ...]."""
    pattern = os.path.join(tmpdir, "seq_%05d.jpg")
    r = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", path,
         "-vf", f"fps={fps},{FRAME_SCALE}", "-q:v", "3", pattern])
    if r.returncode != 0:
        die("ffmpeg frame sampling failed")
    files = sorted(f for f in os.listdir(tmpdir) if f.startswith("seq_"))
    if not files:
        die("ffmpeg produced no frames")
    return [(i / fps, os.path.join(tmpdir, f)) for i, f in enumerate(files)]


def extract_frame_at(path, t, outpath):
    """Single accurate frame at time t (used for seam straddle + cover candidates)."""
    r = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{t:.3f}", "-i", path,
         "-frames:v", "1", "-vf", FRAME_SCALE, "-q:v", "3", outpath])
    return r.returncode == 0 and os.path.exists(outpath)


# ---------------------------------------------------------------- Ollama

def ollama_vision(url, model, prompt, image_paths, force_json=True, timeout=600):
    """One qwen2.5vl call with N images; returns raw response text."""
    images = [base64.b64encode(open(p, "rb").read()).decode() for p in image_paths]
    body = {
        "model": model,
        "prompt": prompt,
        "images": images,
        "stream": False,
        "options": {"temperature": 0},
        # Keep the model resident between calls so a multi-job nightly batch doesn't
        # pay qwen2.5vl's cold-start on every job's QA (gen between jobs >> Ollama's
        # 5min default keep_alive, which would evict it). nightly_generate.sh exports
        # OLLAMA_KEEP_ALIVE to span the whole run; standalone use gets a sane 15m.
        "keep_alive": os.environ.get("OLLAMA_KEEP_ALIVE", "15m"),
    }
    if force_json:
        body["format"] = "json"
    req = urllib.request.Request(
        url.rstrip("/") + "/api/generate",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    resp = json.load(urllib.request.urlopen(req, timeout=timeout))
    return resp.get("response", "")


def parse_json_defensive(text):
    """Strip code fences, extract the first balanced {...}, parse."""
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    start = t.find("{")
    if start < 0:
        raise ValueError(f"no JSON object in model output: {text[:200]!r}")
    depth = 0
    for i in range(start, len(t)):
        if t[i] == "{":
            depth += 1
        elif t[i] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(t[start:i + 1])
    # unbalanced — last-ditch: let json find the prefix it likes
    return json.loads(t[start:])


def vlm_json(url, model, prompt, image_paths, retries=2):
    """Call + defensive parse, with retry (2nd retry drops format=json forcing)."""
    last_err = None
    for attempt in range(retries + 1):
        try:
            raw = ollama_vision(url, model, prompt, image_paths,
                                force_json=(attempt < retries))
            vlog(f"  raw[{attempt}]: {raw[:300]}")
            return parse_json_defensive(raw)
        except (urllib.error.URLError, ValueError, json.JSONDecodeError, OSError) as e:
            last_err = e
            log(f"  VLM call failed (attempt {attempt + 1}): {e}")
    die(f"VLM unrecoverable after {retries + 1} attempts: {last_err}")


def clamp_score(v):
    try:
        return max(0, min(10, int(round(float(v)))))
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------- rubric prompts

def time_labels(times):
    return ", ".join(f"frame {i + 1} = t={t:.1f}s" for i, t in enumerate(times))


def seq_batch_prompt(times, duration):
    return (
        "You are a strict quality judge for an AI-generated anime fight-VFX video. "
        f"The {len(times)} images are consecutive frames sampled from one "
        f"{duration:.0f}-second video, in chronological order: {time_labels(times)}.\n"
        "Score each axis 0-10 (10 = flawless, 6 = acceptable with minor issues, "
        "<=4 = clearly broken):\n"
        '- "character_consistency": does the main hero keep the SAME face, hair, outfit '
        "and color palette in every frame? Identity drift, palette swaps, or the hero "
        "looking like a different character in some frames = 4 or less.\n"
        '- "action_readability": is the fight action clear and easy to follow — '
        "distinct silhouettes, readable poses, the eye always knows where to look?\n"
        '- "artifacts": visual cleanliness. FIRST look for flaws: melted or extra '
        "fingers, warped faces, smeared or duplicated limbs, garbled geometry, objects "
        "fusing together. Typical AI fight footage scores 5-6 here; score 8+ ONLY if "
        "you found zero flaws in every frame; if you can name even one clear flaw, the "
        "score is at most 7.\n"
        '- "worst": list of the worst frames, [{"t": <seconds from the labels>, '
        '"why": "<short reason>"}]; use [] if nothing stands out.\n'
        'Respond with ONLY a JSON object: {"character_consistency": n, '
        '"action_readability": n, "artifacts": n, "worst": [...]}'
    )


def global_batch_prompt(times, duration):
    return (
        "You are a strict quality judge for an AI-generated anime fight-VFX video. "
        f"The {len(times)} images are sampled evenly across the WHOLE "
        f"{duration:.0f}-second video, in chronological order: {time_labels(times)}.\n"
        'Score one axis 0-10:\n'
        '- "energy_arc": does the action escalate so the most spectacular, '
        "highest-energy moment arrives near the END? 10 = clear build to a big final "
        "climax; 5 = flat energy; low = the video fizzles out or peaks too early.\n"
        'Respond with ONLY a JSON object: {"energy_arc": n, '
        '"notes": "<one short sentence>"}'
    )


# Describe-then-judge identity check: a 7B VLM is far more reliable at LISTING factual
# differences than at honest 0-10 scoring (it lenience-inflates direct scores — observed
# qwen2.5vl giving the known-bad chain30_v1 a 10/10 consistency). So we ask for facts and
# compute the score in code.
def identity_diff_prompt(n_a, n_b):
    return (
        "You are checking character continuity in an AI-generated anime fight video.\n"
        f"GROUP A = the first {n_a} images (early part of the video). "
        f"GROUP B = the last {n_b} images (late part of the video).\n"
        "Compare the PROTAGONIST (the heroine) between group A and group B. Ignore the "
        "environment, the enemies, lighting, camera angle, motion blur and pose — ONLY "
        "the heroine's own design: face shape, hair (color/style), outfit pieces, armor "
        "design, color palette, body proportions, and overall art style of her rendering.\n"
        "BE CONSERVATIVE: only report a difference if you are CONFIDENT it is a real "
        "design change, NOT explained by distance, camera angle, motion blur, lighting, "
        "pose, or her being small in frame. If she is too small or obscured to judge in "
        "a frame, skip that frame entirely. Videos made with a character reference "
        "normally have ZERO or ONE real differences — a long list of differences is "
        "almost always wrong. For each difference you must name the two frames that "
        "prove it.\n"
        'severity: "minor" = subtle shift a casual viewer misses; "moderate" = '
        'noticeable redesign of one element; "major" = she reads as a different '
        "character / different outfit / different art style.\n"
        'Respond with ONLY a JSON object: {"same_hero": true|false, '
        '"differences": [{"aspect": "face|hair|outfit|armor|palette|proportions|art_style", '
        '"severity": "minor|moderate|major", "detail": "<short>", '
        '"evidence": "frame X vs frame Y"}]} '
        '(use "differences": [] if she is identical).'
    )


_SEVERITY_COST = {"minor": 0.5, "moderate": 1.5, "major": 3.0}
_IDENTITY_ASPECTS = {"face", "hair", "outfit", "armor", "palette", "proportions", "art_style"}


def score_identity(res):
    """10 minus severity costs of protagonist-design differences; hard floor if the
    model says it's not even the same hero."""
    cost = 0.0
    for d in (res.get("differences") or []):
        if not isinstance(d, dict):
            continue
        aspect = str(d.get("aspect", "")).strip().lower()
        if aspect and aspect not in _IDENTITY_ASPECTS:
            continue  # environment/enemy/lighting noise — excluded by protocol
        cost += _SEVERITY_COST.get(str(d.get("severity", "")).strip().lower(), 1.0)
    score = max(0, round(10 - cost))
    if res.get("same_hero") is False:
        score = min(score, 3)
    return score


# Describe-then-judge seam check (same rationale as identity_diff_prompt): the model
# answers FACTUAL categorical questions about the frames around the cut; the seam score
# is computed in code. The discriminating fact (validated on chain30 v1 vs v3): a big new
# enemy that is abruptly FULLY PRESENT in the first frames after the cut = teleported in
# by the edit (bad); one that ERUPTS/FORMS on screen, or arrives later, is a deliberate
# escalation (good).
def seam_prompt(times, seam_at, duration):
    n_before = sum(1 for t in times if t < seam_at)
    return (
        "You are checking edit continuity in an AI-generated anime fight-VFX video "
        f"({duration:.0f}s) made by joining two clips at t={seam_at:.1f}s. "
        f"The {len(times)} images straddle that cut in chronological order: "
        f"{time_labels(times)}. The first {n_before} frames are from just BEFORE the "
        f"cut, the rest from just AFTER it.\n"
        "Answer these FACTUAL questions about the frames:\n"
        '- "same_hero": is the heroine in the AFTER frames the same character design '
        "(face, hair, outfit, palette) as in the BEFORE frames?\n"
        '- "same_art_style": do the BEFORE and AFTER frames share the same rendering '
        "style and color grading (not two different-looking videos)?\n"
        '- "new_large_figure": look at the FIRST frame after the cut. Is there a HUGE '
        "figure or mass (much bigger than the heroine) there that does NOT exist in "
        "the BEFORE frames? If yes, judge its FORM in that first after-frame: "
        '"fully_present_immediately" = a COMPLETE creature standing in a pose, with '
        'clearly readable head, torso and limbs, as if it teleported in during the cut; '
        '"emerging" = an unformed mass, eruption, mound or partial shape with NO '
        "readable standing body yet (it is still being born on screen). "
        'If no huge new figure or mass is visible in the first after-frame: "none".\n'
        'Respond with ONLY a JSON object: {"same_hero": true|false, '
        '"same_art_style": true|false, "new_large_figure": '
        '"none|emerging|fully_present_immediately", "notes": "<one short sentence>"}'
    )


def score_seam(res):
    """Categorical facts -> score. Identity/style break or a teleported-in foe fails;
    a clean cut or an on-screen-born escalation passes."""
    if res.get("same_hero") is False or res.get("same_art_style") is False:
        return 3
    fig = str(res.get("new_large_figure", "none")).strip().lower()
    if "fully_present" in fig:
        return 4
    if "emerging" in fig:
        return 8
    return 9


def cover_prompt(times, objective):
    return (
        f"The {len(times)} images are candidate COVER frames sampled across an "
        f"anime fight-VFX video (in time order): {time_labels(times)}.\n"
        f"Selection goal — pick the single frame that best is: {objective}.\n"
        "Also require: the hero on-model, minimal motion blur, no obvious AI artifacts.\n"
        'Respond with ONLY a JSON object: {"choice": <frame number 1-'
        f'{len(times)}>, "why": "<short reason>"}}'
    )


# ---------------------------------------------------------------- stages

def run_gate(video, duration, fps, seam_at, url, model, tmpdir):
    frames = extract_fps_frames(video, fps, tmpdir)
    log(f"sampled {len(frames)} frames @ {fps}fps from {duration:.1f}s video")

    batch_scores = {"character_consistency": [], "action_readability": [], "artifacts": []}
    worst_moments = []
    notes = []

    # 1) sequential batches: consistency / readability / artifacts / worst moments
    n_batches = math.ceil(len(frames) / BATCH_SIZE)
    for b in range(n_batches):
        chunk = frames[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
        times = [t for t, _ in chunk]
        paths = [p for _, p in chunk]
        log(f"batch {b + 1}/{n_batches}: t={times[0]:.0f}s..{times[-1]:.0f}s "
            f"({len(paths)} frames)")
        res = vlm_json(url, model, seq_batch_prompt(times, duration), paths)
        for axis in batch_scores:
            s = clamp_score(res.get(axis))
            if s is not None:
                batch_scores[axis].append(s)
        for w in (res.get("worst") or []):
            if isinstance(w, dict) and "t" in w:
                try:
                    worst_moments.append({"t": round(float(w["t"]), 1),
                                          "why": str(w.get("why", ""))[:160]})
                except (TypeError, ValueError):
                    pass
        vlog(f"  -> {res}")

    # 2) whole-video spread batch: energy arc
    n_spread = min(BATCH_SIZE, len(frames))
    idxs = [round(i * (len(frames) - 1) / (n_spread - 1)) for i in range(n_spread)] \
        if n_spread > 1 else [0]
    spread = [frames[i] for i in sorted(set(idxs))]
    log(f"global spread batch: {len(spread)} frames across full duration")
    gres = vlm_json(url, model,
                    global_batch_prompt([t for t, _ in spread], duration),
                    [p for _, p in spread])
    energy = clamp_score(gres.get("energy_arc"))
    if gres.get("notes"):
        notes.append(f"global: {str(gres['notes'])[:200]}")
    vlog(f"  -> {gres}")

    # 2b) describe-then-judge identity check: 3 frames from each half, score computed
    # in code from the listed protagonist-design differences (immune to the 7B model's
    # direct-score lenience; this is the check that must catch cross-seam redesigns).
    half = len(frames) // 2
    first_half, second_half = frames[:half] or frames[:1], frames[half:] or frames[-1:]
    pick3 = lambda fr: [fr[0], fr[len(fr) // 2], fr[-1]]
    ident_frames = pick3(first_half) + pick3(second_half)
    log("identity diff check: 3 early + 3 late frames")
    ires = vlm_json(url, model, identity_diff_prompt(3, 3),
                    [p for _, p in ident_frames])
    identity_score = score_identity(ires)
    diffs = ires.get("differences") or []
    if diffs:
        notes.append("identity diffs: " + "; ".join(
            f"{d.get('aspect')}/{d.get('severity')}: {str(d.get('detail', ''))[:60]}"
            for d in diffs[:4] if isinstance(d, dict)))
    vlog(f"  -> {ires} => identity_score={identity_score}")

    # 3) seam straddle batch
    seam = None
    if seam_at is not None:
        seam_files = []
        seam_times = []
        for off in SEAM_OFFSETS:
            t = seam_at + off
            if t < 0 or t > duration - 0.05:
                continue
            fp = os.path.join(tmpdir, f"seam_{len(seam_files):02d}.jpg")
            if extract_frame_at(video, t, fp):
                seam_files.append(fp)
                seam_times.append(t)
        if len(seam_files) < 2:
            die("could not extract seam straddle frames")
        log(f"seam batch: {len(seam_files)} frames straddling t={seam_at:.1f}s")
        sres = vlm_json(url, model, seam_prompt(seam_times, seam_at, duration),
                        seam_files)
        seam = score_seam(sres)
        facts = {k: sres.get(k) for k in ("same_hero", "same_art_style", "new_large_figure")}
        notes.append(f"seam facts: {facts}" + (f" — {str(sres['notes'])[:160]}" if sres.get("notes") else ""))
        vlog(f"  -> {sres} => seam={seam}")

    # aggregate: median per axis across batches; consistency additionally takes the
    # min with the whole-video check so a cross-seam identity swap (which every
    # individual batch can be blind to) cannot be averaged away.
    def med(vals):
        return int(statistics.median(vals)) if vals else None

    scores = {
        "character_consistency": med(batch_scores["character_consistency"]),
        "action_readability": med(batch_scores["action_readability"]),
        "artifacts": med(batch_scores["artifacts"]),
        "energy_arc": energy,
        "seam_quality": seam,
    }
    # min-combine with the describe-then-judge identity score so a cross-video redesign
    # (which per-batch direct scores are blind/lenient to) cannot be averaged away.
    if scores["character_consistency"] is not None:
        scores["character_consistency"] = min(scores["character_consistency"], identity_score)
    else:
        scores["character_consistency"] = identity_score

    return scores, worst_moments[:6], notes


def pick_cover(video, duration, url, model, tmpdir):
    cands = []
    for i, frac in enumerate(COVER_FRACS):
        t = min(frac * duration, duration - 0.1)
        fp = os.path.join(tmpdir, f"cover_{i}.jpg")
        if extract_frame_at(video, t, fp):
            cands.append((t, fp))
    if not cands:
        die("could not extract cover candidate frames")
    # Rotate the cover objective per-video (deterministic from the video path) so the
    # channel grid diversifies; without this, "most striking VFX" converges every cover
    # on the climactic energy-blast composition.
    obj_idx = int(hashlib.sha256(os.path.abspath(video).encode()).hexdigest()[:8], 16) % len(COVER_OBJECTIVES)
    objective = COVER_OBJECTIVES[obj_idx]
    log(f"cover pick: {len(cands)} candidates at "
        + ", ".join(f"{t:.1f}s" for t, _ in cands)
        + f"; objective[{obj_idx}]")
    res = vlm_json(url, model, cover_prompt([t for t, _ in cands], objective),
                   [p for _, p in cands])
    choice = clamp_score(res.get("choice"))
    if not choice or not (1 <= choice <= len(cands)):
        log("cover: model gave no usable choice, falling back to 45% of duration")
        return round(0.45 * duration, 2), "fallback (45%)"
    t = cands[choice - 1][0]
    return round(t, 2), f"obj{obj_idx}: " + str(res.get("why", ""))[:180]


# ---------------------------------------------------------------- delivery check
# Experiment manipulation-check: did the produced video actually embody the
# experimental directive (e.g. "an impact beat in the first 2s")? Same philosophy as
# the seam/identity checks — the VLM answers a STRICT factual yes/no about a targeted
# window of frames, and `delivered` is decided in CODE by comparing to the experiment's
# expected answer. This is only a PROPOSED verdict; the human confirms/overrides it in
# review app review. Runs ONLY on experimental jobs (spec carries a hypothesis_id +
# delivery_check); never touches normal jobs or the publish gate above.

def delivery_window(when, duration):
    """Resolve a `when` token to a (lo, hi) time window within the video."""
    when = (when or "whole").strip().lower()
    m = re.match(r"first_(\d+(?:\.\d+)?)s$", when)
    if m:
        return 0.0, min(float(m.group(1)), duration)
    if when in ("finisher", "ending", "last_3s"):
        return max(0.0, duration - 3.0), duration
    return 0.0, duration  # whole / unknown


def delivery_prompt(question, when, n):
    where = "the video"
    if when and str(when).startswith("first_"):
        where = f"the opening (~{str(when).split('_')[1]}) of the video"
    elif when in ("finisher", "ending", "last_3s"):
        where = "the final ~3 seconds of the video"
    return (
        f"The {n} images are frames sampled in time order from {where}, a short "
        "anime fight clip.\n"
        f"Question: {question}\n"
        "Judge ONLY what is visibly present in these frames. Be strict and literal: if "
        "the thing asked about is not clearly there, answer no.\n"
        'Respond with ONLY a JSON object: {"answer": "yes"|"no", '
        '"confidence": "low"|"medium"|"high", "evidence": "<one sentence citing the frames>"}'
    )


def verify_delivery(video, delivery_check, url=OLLAMA_URL_DEFAULT, model=MODEL_DEFAULT,
                    max_frames=6):
    """Proposed manipulation-check verdict for one experimental video. delivery_check =
    {"question": str, "expected": "yes"|"no", "when": "first_2s"|"finisher"|"whole"|...}.
    Returns {delivered, answer, expected, confidence, evidence, when, window, n_frames}
    or {delivered: None, error: ...}."""
    q = (delivery_check or {}).get("question")
    if not q:
        return {"delivered": None, "error": "delivery_check has no question"}
    expected = str((delivery_check or {}).get("expected", "yes")).strip().lower()
    expected_bool = expected in ("yes", "true", "1")
    when = (delivery_check or {}).get("when", "whole")
    duration = probe_duration(video)
    lo, hi = delivery_window(when, duration)
    span = max(hi - lo, 0.001)
    n = max(2, min(max_frames, int(round(span)) + 1))
    tmp = tempfile.mkdtemp(prefix="deliv_")
    try:
        frames = []
        for i in range(n):
            t = lo + span * (i + 0.5) / n
            fp = os.path.join(tmp, f"deliv_{i:02d}.jpg")
            if extract_frame_at(video, min(t, duration - 0.05), fp):
                frames.append(fp)
        if not frames:
            return {"delivered": None, "error": "no frames extracted"}
        try:
            res = vlm_json(url, model, delivery_prompt(q, when, len(frames)), frames)
        except Exception as e:  # noqa: BLE001 — VLM/network/parse all -> inconclusive
            return {"delivered": None, "error": f"vlm call failed: {e}"}
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)
    ans = res.get("answer")
    ans_bool = ans if isinstance(ans, bool) else str(ans).strip().lower() in ("yes", "true", "1")
    return {
        "delivered": ans_bool == expected_bool,
        "answer": "yes" if ans_bool else "no",
        "expected": "yes" if expected_bool else "no",
        "confidence": res.get("confidence"),
        "evidence": res.get("evidence"),
        "when": when,
        "window": [round(lo, 2), round(hi, 2)],
        "n_frames": len(frames),
    }


# ---------------------------------------------------------------- main

def parse_thresholds(spec):
    th = dict(DEFAULT_THRESHOLDS)
    if not spec:
        return th
    for kv in spec.split(","):
        if not kv.strip():
            continue
        if "=" not in kv:
            die(f"bad --thresholds entry: {kv!r} (want axis=value)")
        k, v = kv.split("=", 1)
        k = AXIS_ALIASES.get(k.strip(), k.strip())
        if k not in th:
            die(f"unknown threshold axis: {k!r} (axes: {', '.join(th)})")
        try:
            th[k] = float(v)
        except ValueError:
            die(f"bad threshold value for {k}: {v!r}")
    return th


def main():
    global VERBOSE
    ap = argparse.ArgumentParser(description="Local VLM publish gate (Module B)")
    ap.add_argument("video")
    ap.add_argument("--json", dest="json_out", help="write result JSON here")
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--seam-at", default=None,
                    help="seam time in seconds, or 'auto' (= duration/2)")
    ap.add_argument("--thresholds", default=None,
                    help="overrides, e.g. consistency=6,seam=7 "
                         "(aliases: consistency, readability, artifacts, energy_arc, seam)")
    ap.add_argument("--pick-cover", action="store_true",
                    help="also pick the best cover frame (70/80/90/96%% of duration)")
    ap.add_argument("--pick-cover-only", action="store_true",
                    help="only pick the cover frame; skip the gate")
    ap.add_argument("--model", default=MODEL_DEFAULT)
    ap.add_argument("--ollama-url", default=OLLAMA_URL_DEFAULT)
    ap.add_argument("--verify-delivery", default=None,
                    help='experiment manipulation-check: JSON {question,expected,when}; '
                         'emits a PROPOSED delivered verdict and exits 0 (human confirms)')
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    VERBOSE = args.verbose

    if not os.path.isfile(args.video):
        die(f"no such file: {args.video}")

    # Standalone manipulation-check mode (experimental jobs only): emit the proposed
    # delivered verdict and exit 0 — it is informational (the human confirms), never a
    # hard publish gate.
    if args.verify_delivery is not None:
        try:
            dc = json.loads(args.verify_delivery)
        except ValueError as e:
            die(f"bad --verify-delivery JSON: {e}")
        res = verify_delivery(args.video, dc, args.ollama_url, args.model)
        out = {"video": os.path.abspath(args.video), "model": args.model, "delivery": res}
        if args.json_out:
            with open(args.json_out, "w") as f:
                json.dump(out, f, indent=2, ensure_ascii=False)
            log(f"wrote {args.json_out}")
        log(f"delivery: {res}")
        sys.exit(0)

    thresholds = parse_thresholds(args.thresholds)
    duration = probe_duration(args.video)

    seam_at = None
    if args.seam_at is not None:
        if str(args.seam_at).lower() == "auto":
            seam_at = duration / 2
        else:
            try:
                seam_at = float(args.seam_at)
            except ValueError:
                die(f"bad --seam-at: {args.seam_at!r}")
            if not (0 < seam_at < duration):
                die(f"--seam-at {seam_at} outside video duration {duration:.1f}s")

    t0 = time.time()
    result = {"video": os.path.abspath(args.video),
              "model": args.model, "duration": round(duration, 2)}
    exit_code = 0

    with tempfile.TemporaryDirectory(prefix="qa_gate_") as tmpdir:
        do_gate = not args.pick_cover_only
        if do_gate:
            scores, worst, notes = run_gate(
                args.video, duration, args.fps, seam_at,
                args.ollama_url, args.model, tmpdir)

            failed = []
            for axis, thr in thresholds.items():
                s = scores.get(axis)
                if axis == "seam_quality" and seam_at is None:
                    continue
                if s is None:
                    failed.append(f"{axis} (no score)")
                elif s < thr:
                    failed.append(f"{axis} ({s} < {thr:g})")
            verdict = "FAIL" if failed else "PASS"
            result.update({
                "scores": scores,
                "thresholds": thresholds,
                "seam_at": round(seam_at, 2) if seam_at is not None else None,
                "verdict": verdict,
                "failed_axes": failed,
                "worst_moments": worst,
                "notes": "; ".join(notes),
            })
            exit_code = 0 if verdict == "PASS" else 2

        if args.pick_cover or args.pick_cover_only:
            cover_t, cover_why = pick_cover(
                args.video, duration, args.ollama_url, args.model, tmpdir)
            result["cover_t"] = cover_t
            result["cover_why"] = cover_why
            log(f"cover_t = {cover_t}s ({cover_why})")

    result["wall_clock_s"] = round(time.time() - t0, 1)

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        log(f"wrote {args.json_out}")

    if "verdict" in result:
        log(f"scores: {result['scores']}")
        log(f"VERDICT: {result['verdict']}"
            + (f" — failed: {', '.join(result['failed_axes'])}"
               if result["failed_axes"] else ""))
    log(f"wall-clock: {result['wall_clock_s']}s")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

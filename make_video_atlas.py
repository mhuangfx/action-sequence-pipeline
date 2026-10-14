#!/usr/bin/env python3
"""make_video_atlas.py — Atlas Cloud sibling to make_video.js for the 30s video chain.

Produces the SAME finished job-dir contract make_video.js does — `master.mp4`
(16:9) + `spec.json` + `qa.json` — so the same packaging step (withheld) picks it up.
This is the API-based generator, used when the browser-driven services are unavailable.

Pipeline (mirrors the proven Dreamina bridge pattern — sb2 skipped):
  prompts  : reuse make_video.js --validate-only to render the Formula-B / fight
             template prompts (no browser), then drive Atlas from them.
  sb1      : storyboard silhouette grid via google/nano-banana-pro/text-to-image
  clip1    : bytedance/<tier>/reference-to-video, @image1=char ref,
             @image2=sb1 grid  (Seedance's @imageN binding == our prompt convention)
             <tier> = generators.atlas.seedance_video_tier (see _seedance_tier)
  bridge   : extract clip1's last frame
  clip2    : bytedance/<tier>/image-to-video, image = clip1 last frame
             (seamless continuation; no sb2 needed)
  concat   : hard-cut concat -> master.mp4 (1280x720)
  package  : emit_package.js (withheld)

Usage:
  python3 make_video_atlas.py --spec specs/<id>.json [--no-emit] [--no-audio]
"""
import argparse
import datetime as _dt
import json
import os
import re
import subprocess
import sys
import tempfile

import atlas_cloud as ac

ROOT = os.path.dirname(os.path.abspath(__file__))
JOBS_ROOT = os.path.join(ROOT, "output", "jobs")
MANIFEST = os.path.join(ROOT, "output", "manifest.jsonl")

SB_MODEL = "google/nano-banana-pro/text-to-image"
SB_EDIT_MODEL = "google/nano-banana-pro/edit"  # storyboard SEEDED from a reference frame


def _seedance_tier():
    """Video model tier for Atlas generation — THE SWAP KNOB lives in
    config/generators.json → generators.atlas.seedance_video_tier. Valid values:
    'seedance-2.0' (full fidelity, cinematic), 'seedance-2.0-fast' (~20% cheaper),
    'seedance-2.0-mini' (~50% cheaper). All three share the same input surface
    (reference-to-video + image-to-video, refs, audio, return_last_frame, 720p).
    Defaults to full-fidelity 'seedance-2.0' if unset/unreadable — a bad value just
    means Atlas 400s the submit, which is loud."""
    try:
        cfg = json.load(open(os.path.join(ROOT, "config", "generators.json")))
        tier = (cfg.get("generators", {}).get("atlas", {}) or {}).get("seedance_video_tier")
        if isinstance(tier, str) and tier.strip():
            return tier.strip()
    except Exception:
        pass
    return "seedance-2.0"


_TIER = _seedance_tier()
REF2VID = f"bytedance/{_TIER}/reference-to-video"
IMG2VID = f"bytedance/{_TIER}/image-to-video"


def now_iso():
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def sh(cmd):
    # stdin=DEVNULL: ffmpeg (and other children) otherwise inherit the caller's
    # stdin and DRAIN it. When this driver runs inside nightly_generate.sh's
    # `while read … <<< "$new_specs"` loop, a stdin-reading child eats the
    # remaining spec lines, silently dropping every job after the first video.
    r = subprocess.run(cmd, capture_output=True, text=True, stdin=subprocess.DEVNULL)
    if r.returncode != 0:
        raise RuntimeError(f"cmd failed: {' '.join(cmd)}\n{r.stderr[:500]}")
    return r.stdout


def strip_draft_header(text):
    """Drop the leading '【16:9 草稿 …】' / '[16:9 DRAFT …]' orchestrator-review line."""
    lines = text.splitlines()
    if lines and ("草稿" in lines[0] or "DRAFT" in lines[0]):
        lines = lines[1:]
    return "\n".join(lines).strip()


def strip_image2(text):
    """Remove @image2 (storyboard-grid) references from a clip prompt while keeping
    @image1 (the character ref). The close-up variants (extreme-closeup zoom-out /
    transform) run storyboard-less — Seedance reference-to-video gets ONLY the char
    ref — so the @image2 'follow the storyboard for choreography/composition' clauses
    are meaningless and must be surgically dropped (they sit mid-sentence next to the
    @image1 clause, so we can't just drop whole lines)."""
    tokens = re.split(r"([。，；\n])", text)  # clause, delim, clause, delim, ...
    out = []
    for i in range(0, len(tokens), 2):
        clause = tokens[i]
        delim = tokens[i + 1] if i + 1 < len(tokens) else ""
        out.append(delim if "@image2" in clause else clause + delim)
    s = "".join(out)
    # collapse punctuation orphaned by the removed clauses
    s = re.sub(r"，+([。；])", r"\1", s)
    s = re.sub(r"([。；])，+", r"\1", s)
    s = re.sub(r"。{2,}", "。", s)
    s = re.sub(r"，{2,}", "，", s)
    s = re.sub(r"^\s*[，。；]+", "", s, flags=re.M)
    return s


def clip_uses_storyboard(spec, idx):
    """Per-clip storyboard decision (mirrors make_video.js clipUsesStoryboard). A clip is
    storyboard-LESS when it is a close-up reveal (clip_roles[idx]=="closeup_reveal") or, for
    single-clip close-up test specs, when spec.storyboard is False on clip 0."""
    roles = spec.get("clip_roles")
    role = roles[idx] if isinstance(roles, list) and idx < len(roles) else "action"
    if role == "closeup_reveal":
        return False
    if spec.get("storyboard") is False and idx == 0:
        return False
    return True


def resolve_motion_ref(spec, clip_index):
    """Per-clip @video1 pacing reference, mirroring make_video.js: prefer
    motion_refs[clip_index], fall back to the single motion_ref, else None.
    Action fight_styles require a pacing ref; close-up variants pass none.
    Seedance accepts it as reference_videos[] (the @video1 'omniref' slot, confirmed in
    the Atlas API schema: 1-3 videos, <=15s total, mp4/mov, 480p-1080p, FPS 24-60)."""
    refs = spec.get("motion_refs")
    if isinstance(refs, list) and clip_index < len(refs) and refs[clip_index]:
        return refs[clip_index]
    return spec.get("motion_ref") or None


def clean_for_i2v(text):
    """clip2 runs as image-to-video, which has NO reference_images — strip the
    @image1/@image2 binding lines (the start frame carries identity) and prepend an
    explicit seamless-continuation instruction."""
    out = []
    for ln in text.splitlines():
        if "草稿" in ln or "DRAFT" in ln:
            continue
        if "@image2" in ln:        # storyboard-reference lines are meaningless without refs
            continue
        ln = ln.replace("@image1", "").strip()
        if ln:
            out.append(ln)
    head = ("从起始帧无缝承接上一段：保持主角的形象、配色与画质完全一致；"
            "在15秒内呈现快速凌厉、连贯激烈、特效华丽的打斗，非慢动作。")
    return head + "\n" + "\n".join(out)


def render_prompts(spec, spec_path):
    """Run make_video.js --validate-only on a gen_path=hailuo copy (no browser) to
    render the fight-template prompts, and return their texts."""
    tmp = dict(spec)
    tmp["gen_path"] = "hailuo"
    tmp["generator"] = "hailuo"
    tmp["id"] = spec["id"] + "_psrc"
    tf = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump(tmp, tf); tf.close()
    sh(["node", os.path.join(ROOT, "make_video.js"), "--spec", tf.name,
        "--validate-only", "--no-lock"])
    pd = os.path.join(JOBS_ROOT, tmp["id"], "prompts")
    def rd(name):
        p = os.path.join(pd, name)
        return open(p, encoding="utf-8").read() if os.path.exists(p) else None
    return {
        "sb1": rd("sb1_en.txt"),
        "clip1": rd("video_clip1_zh.txt"),
        "sb2": rd("sb2_bridge_en.txt"),
        "clip2": rd("video_clip2_zh.txt"),
    }


# POSITIVE @video1 pacing clause for a reference-to-video clip2. Continuity to clip1 is carried
# by the SEEDED storyboard (sb2 generated from clip1's last frame), mirroring the proven Hailuo
# bridge — so there is no separate @image3 frame ref to the clip (which had caused match-cut
# position-flips). @video1 contributes motion TIMING; framing follows @image2, style follows @image1.
MOTION_CLAUSE_ZH = ("以 @video1 作为打斗节奏与编排的节拍参考——取其出招时机、攻防节奏与肢体运动；"
                    "机位、景别与画面风格一律以 @image2 分镜与 @image1 角色基准为准，"
                    "最终画面保持高细节写实质感渲染。")


def build_refmode_clip2_prompt(text, with_motion):
    """clip2 as reference-to-video (char @image1 + SEEDED storyboard @image2 [+ pacing @video1]).
    Continuity lives in the storyboard (seeded from clip1's last frame) and the template's own
    '承接上一段' line, so no bridge-frame clause is appended (positive prompting throughout)."""
    parts = [strip_draft_header(text)]
    if with_motion:
        parts.append(MOTION_CLAUSE_ZH)
    return "\n".join(parts)


def last_frame(video, out_png):
    """Grab the final frame for the bridge. Uses -sseof (seek from end): input
    seeking to dur-epsilon lands past the last keyframe and silently encodes
    nothing (rc still 0), so seek-from-end + -update is the robust form."""
    sh(["ffmpeg", "-y", "-sseof", "-0.3", "-i", video, "-update", "1",
        "-frames:v", "1", "-q:v", "2", out_png])
    if not os.path.exists(out_png) or os.path.getsize(out_png) == 0:
        raise RuntimeError(f"last-frame extraction produced no file for {video}")
    return out_png


def has_audio(video):
    out = sh(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
              "stream=codec_type", "-of", "csv=p=0", video]).strip()
    return "audio" in out


def concat(clips, out_mp4):
    """Hard-cut concat via the concat filter (re-encode; clips share 1280x720)."""
    aud = all(has_audio(c) for c in clips)
    inputs = []
    for c in clips:
        inputs += ["-i", c]
    n = len(clips)
    if aud:
        streams = "".join(f"[{i}:v][{i}:a]" for i in range(n))
        fc = f"{streams}concat=n={n}:v=1:a=1[v][a]"
        maps = ["-map", "[v]", "-map", "[a]"]
    else:
        streams = "".join(f"[{i}:v]" for i in range(n))
        fc = f"{streams}concat=n={n}:v=1:a=0[v]"
        maps = ["-map", "[v]"]
    sh(["ffmpeg", "-y", *inputs, "-filter_complex", fc, *maps,
        "-c:v", "libx264", "-crf", "18", "-preset", "medium",
        "-pix_fmt", "yuv420p", *(["-c:a", "aac", "-b:a", "192k"] if aud else []),
        "-movflags", "+faststart", out_mp4])
    return out_mp4


def run_qa(job_dir, master, spec):
    """Real VLM QA on the Atlas master (mirrors make_video.js stageQA) + the
    delivery manipulation-check for cohort specs (stageDelivery). ADVISORY and
    best-effort: exit 2 = FAIL is recorded but never blocks packaging (human
    review app review is the authority), and an unavailable qa_gate/Ollama falls
    back to a static advisory PASS — Atlas is the primary generator now, so
    skipping VLM QA entirely (the old behavior) left review with no second
    opinion."""
    qa_json = os.path.join(job_dir, "qa.json")
    argv = ["python3", os.path.join(ROOT, "qa_gate.py"), master,
            "--json", qa_json, "--pick-cover"]
    if spec.get("clips", 2) == 2:
        argv += ["--seam-at", "auto"]
    verdict = "PASS"
    try:
        r = subprocess.run(argv, stdin=subprocess.DEVNULL, timeout=900)
        if r.returncode == 2:
            verdict = "FAIL"
            print("[atlas-video] QA verdict FAIL — advisory, proceeding to package "
                  "for human review", file=sys.stderr)
        elif r.returncode != 0:
            raise RuntimeError(f"qa_gate exit {r.returncode}")
        if not os.path.exists(qa_json):
            raise RuntimeError("qa_gate wrote no qa.json")
    except Exception as e:
        print(f"[atlas-video] qa_gate unavailable ({e}) — static advisory PASS",
              file=sys.stderr)
        json.dump({"verdict": "PASS", "advisory": True,
                   "note": f"qa_gate unavailable ({e}); human review is the authority"},
                  open(qa_json, "w"), indent=2)
        verdict = "PASS"
    # experiment manipulation-check (flywheel cohorts) — proposed verdict only,
    # human confirms in the review app; inconclusive still proceeds.
    dc = spec.get("delivery_check")
    if dc and dc.get("question"):
        try:
            subprocess.run(
                ["python3", os.path.join(ROOT, "qa_gate.py"), master,
                 "--verify-delivery", json.dumps(dc),
                 "--json", os.path.join(job_dir, "delivery.json")],
                stdin=subprocess.DEVNULL, timeout=600)
        except Exception as e:
            print(f"[atlas-video] verify-delivery unavailable ({e})", file=sys.stderr)
    return verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--no-emit", action="store_true")
    ap.add_argument("--no-audio", action="store_true")
    a = ap.parse_args()

    spec = json.load(open(a.spec))
    ref = spec.get("char_ref")
    if not ref or not os.path.exists(ref):
        sys.exit(f"spec.char_ref missing/not found: {ref!r}")
    job_id = spec["id"]
    job_dir = os.path.join(JOBS_ROOT, job_id)
    os.makedirs(job_dir, exist_ok=True)
    json.dump(spec, open(os.path.join(job_dir, "spec.json"), "w"), indent=2)

    clips_n = spec.get("clips", 2)
    dur = spec.get("clip_length", 15)
    ratio = spec.get("format", "16:9")
    resolution = spec.get("resolution", "720p")
    gen_audio = not a.no_audio
    # Per-clip storyboard decision. Close-up reveals (the close-up variant opener) run
    # storyboard-less — the macro face is not choreography, so the silhouette grid is noise.
    # Action clips keep it. Driven by spec.clip_roles[idx] (or spec.storyboard:false for a
    # single-clip close-up test).
    use_sb = clip_uses_storyboard(spec, 0)
    # clip2 mode: the close-up-opener composite (spec.clip_roles set) runs clip2 as a full
    # REFERENCE-to-video (char @image1 + storyboard @image2 + bridge @image3 + pacing @video1) —
    # Seedance image-to-video can't take those extra refs. Legacy action specs (no clip_roles)
    # keep the proven seamless i2v bridge.
    clip2_refmode = bool(spec.get("clip_roles"))
    key = ac.load_key()
    started = now_iso()

    print(f"[atlas-video] {job_id}: {clips_n}x{dur}s {ratio} {resolution} "
          f"audio={gen_audio} storyboard={use_sb}", file=sys.stderr)
    pr = render_prompts(spec, a.spec)
    if not pr["clip1"]:
        sys.exit("failed to render clip1 prompt via make_video.js --validate-only")

    # --- sb1: storyboard silhouette grid (Nano Banana, text-to-image, 16:9) ---
    # Each gen stage is skipped if its artifact already exists (resume-safe: never
    # re-bills a completed paid stage on a re-run).
    sb1_path = os.path.join(job_dir, "storyboard1.png")
    sb_aspect = ratio if ratio in ("16:9", "9:16", "4:3", "3:4", "1:1", "21:9") else "16:9"
    if not use_sb:
        print("[atlas-video] sb1: SKIPPED (storyboard-less close-up variant)", file=sys.stderr)
    elif os.path.exists(sb1_path):
        print("[atlas-video] sb1: exists, skip", file=sys.stderr)
    else:
        print("[atlas-video] sb1 (nano-banana)…", file=sys.stderr)
        ac.gen_image(SB_MODEL, strip_draft_header(pr["sb1"]), None, sb1_path,
                     aspect=sb_aspect, resolution="2k", key=key)

    # --- clip1: Seedance reference-to-video ---
    #   action styles  : @image1=char ref, @image2=sb1 grid, @video1=pacing ref
    #   close-up styles: @image1=char ref only; @image2 clauses stripped from prompt
    clip1 = os.path.join(job_dir, "clip1.mp4")
    clip1_refs = [ref, sb1_path] if use_sb else [ref]
    clip1_prompt = strip_draft_header(pr["clip1"]) if use_sb else strip_image2(strip_draft_header(pr["clip1"]))
    clip1_mref = resolve_motion_ref(spec, 0)
    if os.path.exists(clip1):
        print("[atlas-video] clip1: exists, skip", file=sys.stderr)
    else:
        print(f"[atlas-video] clip1 (seedance ref-to-video, "
              f"refs={len(clip1_refs)}, pacing={'yes' if clip1_mref else 'no'})…", file=sys.stderr)
        ac.gen_video(REF2VID, clip1_prompt, clip1_refs, clip1,
                     ratio=ratio, resolution=resolution, duration=dur,
                     generate_audio=gen_audio, watermark=False, key=key,
                     ref_videos=[clip1_mref] if clip1_mref else None)

    master_clips = [clip1]
    if clips_n >= 2 and pr["clip2"]:
        clip2 = os.path.join(job_dir, "clip2.mp4")
        clip2_mref = resolve_motion_ref(spec, 1)
        clip2_sb = clip_uses_storyboard(spec, 1)
        if os.path.exists(clip2):
            print("[atlas-video] clip2: exists, skip", file=sys.stderr)
        elif clip2_refmode:
            # ACTION clip2 as reference-to-video, replicating the PROVEN Hailuo bridge on Atlas:
            # sb2 (the clip2 storyboard) is SEEDED FROM clip1's LAST FRAME (nano-banana-pro/edit),
            # so its panel 1 echoes clip1's ending composition. clip2 then = char @image1 + seeded
            # storyboard @image2 (+ pacing @video1). No separate frame ref to the clip → continuity
            # comes through the seeded storyboard, avoiding the match-cut position-flip.
            bridge = last_frame(clip1, os.path.join(job_dir, "bridge_frame.png"))
            sb2_path = os.path.join(job_dir, "storyboard2.png")
            if clip2_sb and pr.get("sb2") and not os.path.exists(sb2_path):
                print("[atlas-video] sb2 (nano-banana EDIT, seeded from clip1 last frame)…", file=sys.stderr)
                ac.gen_image(SB_EDIT_MODEL, strip_draft_header(pr["sb2"]), [bridge], sb2_path,
                             aspect=sb_aspect, resolution="2k", key=key)
            refs2 = [ref] + ([sb2_path] if (clip2_sb and os.path.exists(sb2_path)) else [])
            prompt2 = build_refmode_clip2_prompt(pr["clip2"], with_motion=bool(clip2_mref))
            print(f"[atlas-video] clip2 (seedance ref-to-video, refs={len(refs2)}, "
                  f"storyboard={'seeded' if clip2_sb else 'no'}, pacing={'yes' if clip2_mref else 'no'})…",
                  file=sys.stderr)
            ac.gen_video(REF2VID, prompt2, refs2, clip2,
                         ratio=ratio, resolution=resolution, duration=dur,
                         generate_audio=gen_audio, watermark=False, key=key,
                         ref_videos=[clip2_mref] if clip2_mref else None)
        else:
            # legacy proven path: seamless image-to-video bridged from clip1's last frame
            bridge = last_frame(clip1, os.path.join(job_dir, "bridge_frame.png"))
            print(f"[atlas-video] clip2 (seedance image-to-video, bridged, "
                  f"pacing={'yes' if clip2_mref else 'no'})…", file=sys.stderr)
            ac.gen_video(IMG2VID, clean_for_i2v(pr["clip2"]), None, clip2,
                         ratio=ratio, resolution=resolution, duration=dur,
                         generate_audio=gen_audio, watermark=False,
                         key=key, image=ac.as_ref(bridge),
                         ref_videos=[clip2_mref] if clip2_mref else None)
        master_clips.append(clip2)

    # --- concat -> master.mp4 ---
    master = os.path.join(job_dir, "master.mp4")
    if len(master_clips) == 1:
        sh(["ffmpeg", "-y", "-i", master_clips[0], "-c", "copy", master])
    else:
        concat(master_clips, master)
    mdur = float(sh(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                     "-of", "default=nk=1:nw=1", master]).strip())
    print(f"[atlas-video] master.mp4 {mdur:.2f}s", file=sys.stderr)

    # real VLM QA (advisory) + delivery check — writes qa.json (and delivery.json
    # for cohort specs); falls back to a static advisory PASS if Ollama is down.
    qa_verdict = run_qa(job_dir, master, spec)
    json.dump({"stage": "qa", "status": "ok", "generator": "atlas"},
              open(os.path.join(job_dir, "state.json"), "w"), indent=2)

    if a.no_emit:
        print("[atlas-video] --no-emit: stopping before package", file=sys.stderr)
        return

    subprocess.run(["node", os.path.join(ROOT, "emit_package.js"),
                    "--job", job_id, "--jobs-root", JOBS_ROOT], cwd=ROOT, check=True,
                   stdin=subprocess.DEVNULL)
    pkg_dir = os.path.join(job_dir, "package")
    pkg_files = [os.path.join(pkg_dir, f) for f in os.listdir(pkg_dir)]
    entry = {"job_id": job_id, "generator": spec.get("generator", "atlas"),
             "gen_path": spec.get("gen_path", "atlas"), "content_type": "video",
             "spec": spec, "qa": {"verdict": qa_verdict, "advisory": True},
             "files": {"package": pkg_files}, "status": "completed",
             "timestamps": {"started": started, "finished": now_iso()},
             "credits_estimate": 0}
    with open(MANIFEST, "a") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"[atlas-video] DONE {job_id} — {len(pkg_files)} package files; "
          f"run post_db ingest to queue.", file=sys.stderr)


if __name__ == "__main__":
    main()

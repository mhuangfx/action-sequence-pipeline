#!/usr/bin/env python3
"""pick_char_ref.py — pick the best frame of a character to seed a reference sheet.

Built for the anime-swap workflow: once a character has been rendered in OUR style, we want a
persistent reference so every later segment inherits that look instead of re-inheriting the
source's rendering. The frame has to come from our OWN output, never the source — that is what
breaks the "recognisability and style bleed are the same thing" coupling.

Picking by eye does not scale to a 21-segment battle, so the VLM scores candidates instead and a
human verifies the shortlist.

What makes a good character-reference frame is not what makes a good ACTION frame — we want the
opposite of the money shot: sharp, large, face visible, costume readable, not buried in VFX.

Usage:
  .venv/bin/python tools/pick_char_ref.py OUT.mp4 [OUT2.mp4 ...] \\
      --describe-target "white spiky hair, red cloak, dark armoured vest" \\
      --out-dir output/charref/opponent --fps 4 --top 6
"""
import argparse
import base64
import json
import re
import subprocess
import urllib.request
from pathlib import Path

PROMPT_TMPL = (
    "You are choosing a CHARACTER REFERENCE frame for an artist. The target character is: {target}. "
    "Reply with JSON only, no prose, keys: "
    '"target_present" (true/false — is that character visible at all), '
    '"target_size" (0-10, how much of the frame the character occupies), '
    '"face_clear" (0-10, how clearly the FACE is visible and in focus; 0 if turned away or blurred), '
    '"costume_clear" (0-10, how readable the outfit/armour/silhouette detail is), '
    '"motion_blur" (0-10, how MUCH blur degrades the character; higher = worse), '
    '"vfx_occlusion" (0-10, how much smoke/fire/glow hides the character; higher = worse), '
    '"other_people" (integer, how many OTHER people are visible), '
    '"render_style" (one of "painterly_realistic", "flat_2d_anime", "mixed"), '
    '"note" (max 12 words). '
    "Do not name any anime, character, or franchise."
)


def sample_frames(clip: Path, out_dir: Path, fps: float):
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    stem = clip.stem
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(clip), "-vf", f"fps={fps}",
                    "-q:v", "2", str(frames_dir / f"{stem}_%03d.jpg")], capture_output=True)
    return sorted(frames_dir.glob(f"{stem}_*.jpg"))


def score_frame(path: Path, target: str, model: str, host: str):
    img = base64.b64encode(path.read_bytes()).decode()
    payload = json.dumps({"model": model, "prompt": PROMPT_TMPL.format(target=target),
                          "images": [img], "format": "json", "stream": False}).encode()
    req = urllib.request.Request(f"{host}/api/generate", data=payload,
                                 headers={"Content-Type": "application/json"})
    try:
        raw = json.loads(urllib.request.urlopen(req, timeout=300).read())["response"].strip()
        return json.loads(raw)
    except Exception as e:
        return {"target_present": False, "note": f"[VLM failed: {e}]"}


STYLE_GATE = False


def rank(v):
    """Higher is better. A reference frame wants clarity and readability, not drama."""
    if not v.get("target_present"):
        return -1.0
    g = lambda k: float(v.get(k) or 0)
    s = (2.0 * g("face_clear") + 1.5 * g("costume_clear") + 1.0 * g("target_size")
         - 1.5 * g("motion_blur") - 1.5 * g("vfx_occlusion"))
    # ★ Style is deliberately NOT gated by default. These frames feed a RESTYLING pass, so their
    # job is to carry IDENTITY (hair shape, costume, silhouette, facial structure) — the restyle
    # step is what fixes rendering. Gating on style (tried 2026-08-08) selects for distant,
    # turned-away, blurred frames, which are the ones carrying the LEAST design information;
    # a source-styled close-up is the better input precisely because it is legible.
    # Use --style-gate only when the frame will be used AS-IS with no restyling pass.
    if STYLE_GATE and v.get("render_style") != "painterly_realistic":
        return -1.0
    s -= 1.5 * float(v.get("other_people") or 0)   # a clean solo frame makes a better sheet
    return round(s, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", nargs="+")
    ap.add_argument("--describe-target", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--fps", type=float, default=4.0)
    ap.add_argument("--top", type=int, default=6)
    ap.add_argument("--style-gate", action="store_true",
                    help="only accept painterly frames — for use AS-IS, not as restyle input")
    ap.add_argument("--model", default="qwen3-vl:8b-instruct-q8_0")
    ap.add_argument("--ollama-host", default="http://localhost:11434")
    a = ap.parse_args()
    global STYLE_GATE
    STYLE_GATE = a.style_gate

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for c in a.clips:
        fs = sample_frames(Path(c), out_dir, a.fps)
        print(f"{c}: {len(fs)} frames @ {a.fps}fps")
        frames += fs

    scored = []
    for i, f in enumerate(frames, 1):
        v = score_frame(f, a.describe_target, a.model, a.ollama_host)
        v["frame"] = str(f)
        v["score"] = rank(v)
        scored.append(v)
        if v["score"] > 0:
            print(f"  [{i}/{len(frames)}] {f.name} score={v['score']:>6} "
                  f"face={v.get('face_clear')} cost={v.get('costume_clear')} "
                  f"blur={v.get('motion_blur')} style={v.get('render_style')} "
                  f"| {str(v.get('note'))[:40]}", flush=True)

    scored.sort(key=lambda x: -x["score"])
    (out_dir / "candidates.json").write_text(json.dumps(scored, indent=2))

    top = [s for s in scored if s["score"] > 0][:a.top]
    print(f"\nTOP {len(top)} (present={sum(1 for s in scored if s.get('target_present'))} "
          f"of {len(scored)} frames):")
    for i, s in enumerate(top, 1):
        print(f"  {i}. {Path(s['frame']).name:<28} score={s['score']:<7} "
              f"style={s.get('render_style'):<20} {str(s.get('note'))[:44]}")

    if top:
        sheet = out_dir / "shortlist.jpg"
        ins = []
        for s in top:
            ins += ["-i", s["frame"]]
        n = len(top)
        fc = "".join(f"[{i}]scale=420:-2,drawtext=text='{i+1}':x=8:y=8:fontsize=28:"
                     f"fontcolor=yellow:box=1:boxcolor=black@0.5[v{i}];" for i in range(n))
        fc += "".join(f"[v{i}]" for i in range(n)) + f"hstack={n}" if n > 1 else "[v0]null"
        subprocess.run(["ffmpeg", "-y", "-v", "error", *ins, "-filter_complex", fc, str(sheet)],
                       capture_output=True)
        print(f"\nShortlist contact sheet: {sheet}")
        print(f"Full scores: {out_dir/'candidates.json'}")


if __name__ == "__main__":
    main()

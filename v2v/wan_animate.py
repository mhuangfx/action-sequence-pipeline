#!/usr/bin/env python3
"""
wan_animate.py — headless driver for WanGP (Wan2GP) "Wan 2.2 Animate".

Maps a character reference image onto a driving video's motion (weight, timing,
impact) — the v2v engine for the action-sequence-pipeline v2v gen path. WanGP has no CLI
generate mode; it is a Gradio app whose only autonomous job-intake is its
"autoload queue" (a queue.zip dropped in the app dir, loaded when a client
connects via the main.load event → process_tasks → generate_video).

This driver:
  1. builds a COMPLETE generate_video() params dict (every kwarg — the function
     has no defaults), with the char image + driving video referenced as
     zip-relative filenames, exactly like WanGP's own autosave_queue();
  2. writes queue.zip into the app dir;
  3. starts the WanGP server (or reuses a running one), triggers the autoload,
     and polls outputs/ for the resulting mp4.

Run with the WanGP venv:
  ~/wan/app/env/bin/python v2v/wan_animate.py \
      --char <ref.png> --video <driving.mp4> --out <result.mp4> \
      [--prompt ...] [--resolution 480x832] [--frames 81] [--steps 25] [--seed 42]

Standalone subcommands: `params` (print the params dict), `zip` (build queue.zip
only), `trigger` (just fire the autoload on a running server).
"""
import argparse
import ast
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from datetime import datetime

APP_DIR = "~/wan/app"
WGP_PY = os.path.join(APP_DIR, "wgp.py")
OUTPUTS = os.path.join(APP_DIR, "outputs")
QUEUE_ZIP = os.path.join(APP_DIR, "queue.zip")
MODEL_URL = ("https://huggingface.co/DeepBeepMeep/Wan2.2/resolve/main/"
             "wan2.2_animate_14B_quanto_bf16_int8.safetensors")


def generate_video_argnames():
    """Extract generate_video()'s exact parameter names from source (no import)."""
    with open(WGP_PY, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "generate_video":
            a = node.args
            names = [arg.arg for arg in (a.posonlyargs + a.args)]
            if a.vararg:
                names.append(a.vararg.arg)
            names += [arg.arg for arg in a.kwonlyargs]
            return names
    raise RuntimeError("generate_video not found in wgp.py")


def build_params(image_ref_zipname, video_guide_zipname, prompt, negative,
                 resolution, video_length, steps, seed):
    """Full generate_video() kwargs (minus task/send_cmd/state, which the loader
    re-injects). Values mirror a known-good animate run's embedded metadata plus
    UI-component defaults; the introspection check below guarantees completeness."""
    p = {
        "image_mode": 0,
        "prompt": prompt,
        "negative_prompt": negative,
        "resolution": resolution,
        "video_length": video_length,
        "batch_size": 1,
        "seed": seed,
        "force_fps": "",
        "num_inference_steps": steps,
        "guidance_scale": 5.5,
        "guidance2_scale": 5.5,
        "guidance3_scale": 5.5,
        "switch_threshold": 0,
        "switch_threshold2": 0,
        "guidance_phases": 1,
        "model_switch_phase": 1,
        "audio_guidance_scale": 0,
        "flow_shift": 7,
        "sample_solver": "unipc",
        "embedded_guidance_scale": 0,
        "repeat_generation": 1,
        "multi_prompts_gen_type": 0,
        "multi_images_gen_type": 0,
        "skip_steps_cache_type": "",
        "skip_steps_multiplier": 1.5,
        "skip_steps_start_step_perc": 20,
        "activated_loras": [],
        "loras_multipliers": "",
        "image_prompt_type": "",
        "image_start": None,
        "image_end": None,
        "model_mode": 0,
        "video_source": None,
        "keep_frames_video_source": "",
        "video_prompt_type": "PVBKI",   # animate: person in ref, whole-video motion
        "image_refs": [image_ref_zipname],
        "frames_positions": "",
        "video_guide": video_guide_zipname,
        "image_guide": None,
        "keep_frames_video_guide": "",
        "denoising_strength": 1.0,
        "video_guide_outpainting": "#",
        "video_mask": None,
        "image_mask": None,
        "control_net_weight": 1,
        "control_net_weight2": 1,
        "control_net_weight_alt": 1,
        "mask_expand": 20,
        "audio_guide": None,
        "audio_guide2": None,
        "audio_source": None,
        "audio_prompt_type": "R",
        "speakers_locations": "",
        "sliding_window_size": 129,
        "sliding_window_overlap": 5,
        "sliding_window_color_correction_strength": 0,
        "sliding_window_overlap_noise": 0,
        "sliding_window_discard_last_frames": 0,
        "image_refs_relative_size": 50,
        "remove_background_images_ref": 0,
        "temporal_upsampling": "",
        "spatial_upsampling": "",
        "film_grain_intensity": 0,
        "film_grain_saturation": 0.5,
        "MMAudio_setting": 0,
        "MMAudio_prompt": "",
        "MMAudio_neg_prompt": "",
        "RIFLEx_setting": 0,
        "NAG_scale": 1,
        "NAG_tau": 3.5,
        "NAG_alpha": 0.5,
        "slg_switch": 0,
        "slg_layers": [9],
        "slg_start_perc": 10,
        "slg_end_perc": 90,
        "apg_switch": 0,
        "cfg_star_switch": 0,
        "cfg_zero_step": -1,
        "prompt_enhancer": "",
        "min_frames_if_references": 1,
        "override_profile": -1,
        "model_type": "animate",
        "model_filename": MODEL_URL,
        "mode": "generate",
    }
    # Completeness guard: every generate_video arg (except the loader-injected
    # task/send_cmd/state) must be present, or process_tasks raises TypeError.
    injected = {"task", "send_cmd", "state"}
    needed = [n for n in generate_video_argnames() if n not in injected]
    missing = [n for n in needed if n not in p]
    extra = [k for k in p if k not in needed]
    if missing:
        raise RuntimeError(f"params MISSING required generate_video args: {missing}")
    if extra:
        # Not fatal, but flag drift between our dict and the installed wgp.py.
        print(f"WARN: params has keys not in generate_video signature: {extra}",
              file=sys.stderr)
    return p


def build_queue_zip(char_png, driving_mp4, params, zip_path):
    img_name = params["image_refs"][0]
    vid_name = params["video_guide"]
    manifest = [{"id": 1, "params": params}]
    with tempfile.TemporaryDirectory() as td:
        mpath = os.path.join(td, "queue.json")
        with open(mpath, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(mpath, arcname="queue.json")
            zf.write(char_png, arcname=img_name)
            zf.write(driving_mp4, arcname=vid_name)
    return zip_path


def start_server(port, logpath):
    env = dict(os.environ)
    env["SERVER_NAME"] = "127.0.0.1"
    env["SERVER_PORT"] = str(port)
    env["PYTHONUNBUFFERED"] = "1"  # else stdout block-buffers to the file -> blind log
    logf = open(logpath, "w")
    proc = subprocess.Popen(
        [os.path.join(APP_DIR, "env/bin/python"), "-u", "wgp.py", "--multiple-images"],
        cwd=APP_DIR, env=env, stdout=logf, stderr=subprocess.STDOUT,
    )
    return proc, logf


def wait_for(logpath, needle, timeout, since=0):
    """Wait until `needle` appears in the log after byte offset `since`. Returns
    the full log text on success, or None on timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if os.path.isfile(logpath):
            with open(logpath, "r", errors="replace") as f:
                f.seek(since)
                txt = f.read()
            if needle in txt:
                return txt
        time.sleep(1.0)
    return None


def newest_output(after_ts):
    if not os.path.isdir(OUTPUTS):
        return None
    cands = [os.path.join(OUTPUTS, f) for f in os.listdir(OUTPUTS)
             if f.lower().endswith(".mp4")]
    cands = [c for c in cands if os.path.getmtime(c) > after_ts]
    return max(cands, key=os.path.getmtime) if cands else None


def trigger_autoload(port, hold_ms=14000):
    """Fire main.load so the server autoloads queue.zip. Must be a REAL browser
    page-load — the Gradio SPA mount calls the load event over the network;
    gradio_client does NOT fire it (verified). We hold the page open long enough
    for the mount->backend round-trip, then leave; generation continues
    server-side in WanGP's async queue thread."""
    url = f"http://127.0.0.1:{port}"
    trigger_js = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "trigger_load.js")
    try:
        subprocess.run(["node", trigger_js, url, str(hold_ms)],
                       check=True, timeout=hold_ms / 1000 + 60,
                       cwd=os.path.dirname(trigger_js))
        return "playwright page-load"
    except Exception as e:
        print(f"  browser trigger failed: {e}", file=sys.stderr)
        return None


def cmd_generate(args):
    os.makedirs(OUTPUTS, exist_ok=True)
    img_name = "task1_image_refs_0.png"
    vid_ext = os.path.splitext(args.video)[1] or ".mp4"
    vid_name = f"task1_video_guide{vid_ext}"
    params = build_params(img_name, vid_name, args.prompt, args.negative,
                          args.resolution, args.frames, args.steps, args.seed)
    build_queue_zip(args.char, args.video, params, QUEUE_ZIP)
    print(f"[wan] queue.zip written: {QUEUE_ZIP} "
          f"(char={os.path.basename(args.char)} video={os.path.basename(args.video)})")
    print(f"[wan] {args.resolution} {args.frames}f {args.steps}steps seed={args.seed}")

    start_ts = time.time()
    logpath = args.log or os.path.join(tempfile.gettempdir(),
                                       f"wan_server_{int(start_ts)}.log")
    proc, logf = start_server(args.port, logpath)
    print(f"[wan] server pid={proc.pid} log={logpath}")
    try:
        # 1) wait for the Gradio URL (server ready)
        if wait_for(logpath, "Running on local URL", args.boot_timeout) is None:
            # some gradio builds print a different banner; fall back to a port poke
            print("[wan] (no 'Running on' banner; proceeding to trigger anyway)")
        else:
            print("[wan] server up.")
        time.sleep(2.0)

        # 2) did the server autoload at startup (no client needed)?
        txt = wait_for(logpath, "Autoloading queue", 3)
        if txt is None:
            # 3) trigger via a client connection (main.load)
            how = trigger_autoload(args.port)
            print(f"[wan] triggered autoload via: {how}")
            txt = wait_for(logpath, "Autoloading queue", 20)
        if txt is None:
            print("[wan] ERROR: autoload never fired — main.load trigger failed. "
                  "See log; may need a real browser page-load.", file=sys.stderr)
            return 3
        print("[wan] queue autoloaded; generating (this is the GPU step)...")

        # 4) wait for completion
        done = wait_for(logpath, "New video saved to Path", args.gen_timeout)
        if done is None:
            print("[wan] ERROR: generation did not complete before timeout.",
                  file=sys.stderr)
            return 4

        out = newest_output(start_ts)
        if not out:
            print("[wan] ERROR: completion logged but no new mp4 found in outputs/.",
                  file=sys.stderr)
            return 5
        elapsed = time.time() - start_ts
        if args.out:
            shutil.copy2(out, args.out)
            print(f"[wan] DONE in {elapsed:.0f}s — result: {args.out} (src {out})")
        else:
            print(f"[wan] DONE in {elapsed:.0f}s — result: {out}")
        return 0
    finally:
        if not args.keep_server:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
        logf.close()


def cmd_params(args):
    params = build_params("task1_image_refs_0.png", "task1_video_guide.mp4",
                          args.prompt, args.negative, args.resolution,
                          args.frames, args.steps, args.seed)
    print(json.dumps(params, indent=2))
    return 0


def cmd_zip(args):
    img_name = "task1_image_refs_0.png"
    vid_ext = os.path.splitext(args.video)[1] or ".mp4"
    vid_name = f"task1_video_guide{vid_ext}"
    params = build_params(img_name, vid_name, args.prompt, args.negative,
                          args.resolution, args.frames, args.steps, args.seed)
    build_queue_zip(args.char, args.video, params, args.out or QUEUE_ZIP)
    print(f"queue.zip -> {args.out or QUEUE_ZIP}")
    return 0


def add_common(ap):
    ap.add_argument("--prompt", default="female flame mage engulfed in flames in "
                    "hand to hand combat with a male opponent")
    ap.add_argument("--negative", default="")
    ap.add_argument("--resolution", default="480x832")
    ap.add_argument("--frames", type=int, default=81)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--seed", type=int, default=42)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="build queue.zip, run server, poll result")
    g.add_argument("--char", required=True)
    g.add_argument("--video", required=True)
    g.add_argument("--out", default=None)
    g.add_argument("--port", type=int, default=7861)
    g.add_argument("--boot-timeout", type=int, default=180)
    g.add_argument("--gen-timeout", type=int, default=2400)
    g.add_argument("--keep-server", action="store_true")
    g.add_argument("--log", default=None)
    add_common(g)
    g.set_defaults(func=cmd_generate)

    p = sub.add_parser("params", help="print the full generate_video params dict")
    add_common(p)
    p.set_defaults(func=cmd_params)

    z = sub.add_parser("zip", help="build queue.zip only")
    z.add_argument("--char", required=True)
    z.add_argument("--video", required=True)
    z.add_argument("--out", default=None)
    add_common(z)
    z.set_defaults(func=cmd_zip)

    args = ap.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()

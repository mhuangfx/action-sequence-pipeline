#!/usr/bin/env python3
"""Drive the local MiniMax H3 (Ref2VA) ComfyUI instance over its HTTP API.

Replaces the Hailuo browser driver for video generation. Handles the three ways
local differs from hosted:
  1. reference tags are <Picture i> / <Video k> / <Audio j>, not @Image1 / @Video1
  2. reference video frames are truncated to the OUTPUT frame count, so a ref
     must be <= the output length, never longer
  3. reference video is assumed to be 24 fps -- our 1.5x refs are 60 fps and get
     silently truncated to a third of their duration unless converted first

Usage:
  tools/h3_local.py --name genG_seg1 \
      --prompt-parts prompts/fullbattle_20260808/segA_h3_scoped_parts.json \
      --ref-image output/fullbattle_20260808/segA_board_3x4_restyled_v3.png \
      --ref-image characters/char01_vesta/ref_vestia_v3.png \
      --ref-image characters/<hero>/sheet.png \
      --ref-video output/fullbattle_20260808/segA_15x.mp4 \
      --seconds 5.9
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import urllib.error

SERVER = os.environ.get("H3_SERVER", "http://127.0.0.1:8199")
COMFY_ROOT = os.environ.get("H3_COMFY_ROOT", "~/ComfyUI")
INPUT_SUBDIR = "fightseq"
FPS = 24

MODELS = {
    "unet": "minimax_h3_ref2va_pruned_fp8_scaled.safetensors",
    "clip": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
    "vae": "minimax_h3_video_vae_fp16.safetensors",
    "audio_vae": "minimax_h3_audio_vae_fp32.safetensors",
}


def align_frames(n):
    """The model snaps length to a 17k+5 grid; mirror it so we know the real duration."""
    n = max(5, int(n))
    while n % 17 != 5:
        n += 1
    return n


def align_frames_down(n):
    """What the node does to a reference video: snap DOWN to the 17k+5 grid."""
    n = int(n)
    while n % 17 != 5:
        n -= 1
    return max(5, n)


def parts_to_prompt(parts):
    """Hosted mention-chip parts -> a local prompt string.

    Hosted saved prompts as [{"t": text}, {"m": "Image1"}, ...] where the chips
    bind to uploaded refs. Locally the tags are literal text the tokenizer reads.
    """
    tag_map = {"image": "Picture", "video": "Video", "audio": "Audio"}
    out = []
    for part in parts:
        if "t" in part:
            out.append(part["t"])
            continue
        mention = part["m"]
        kind = "".join(c for c in mention if c.isalpha()).lower()
        num = "".join(c for c in mention if c.isdigit()) or "1"
        if kind not in tag_map:
            raise ValueError("unknown mention chip: %r" % mention)
        out.append("<%s %s>" % (tag_map[kind], num))
    return "".join(out)


def stage_image(src, dest_dir, tag):
    """Copy a reference image into ComfyUI's input tree; return its LoadImage name."""
    name = "%s_%s" % (tag, os.path.basename(src))
    shutil.copyfile(src, os.path.join(dest_dir, name))
    return "%s/%s" % (INPUT_SUBDIR, name)


def probe_duration(path):
    return float(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", path],
        capture_output=True, text=True, check=True).stdout.strip())


def count_frames(path):
    return int(subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v", "-count_frames",
         "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", path],
        capture_output=True, text=True, check=True).stdout.strip())


def stage_video(src, dest_dir, tag, out_frames, fit):
    """Re-encode a reference clip to 24 fps landing exactly on the model's grid.

    Three constraints collide here. The node reads ref frames as 24 fps whatever
    the source rate; it truncates a ref longer than the output; and it then snaps
    the ref DOWN to the same 17k+5 grid as the output. So a ref that is merely
    "close" loses up to 17 frames off its tail -- for us that was the last 0.7s
    of the segment, i.e. the impact the segment exists for.

    fit=stretch retimes the whole source window into exactly out_frames, so ref
    and output are the same length and cut timings map 1:1. It shifts the speed
    factor slightly (1.5x becomes ~1.48x), which does not matter: 1.5 was picked
    to widen motion dynamic range, not for the exact value.
    fit=trim keeps the source speed and accepts the truncation.
    """
    name = "%s_24fps.mp4" % tag
    dest = os.path.join(dest_dir, name)
    if fit == "stretch":
        src_dur = probe_duration(src)
        target = out_frames / FPS
        vf = "setpts=PTS*%.9f,fps=%d" % (target / src_dur, FPS)
        clip = ["-t", "%.4f" % target]
    else:
        vf = "fps=%d" % FPS
        clip = ["-t", "%.4f" % (align_frames_down(out_frames) / FPS)]
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", src, "-vf", vf] + clip +
        ["-an", "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", dest],
        check=True,
    )
    return "%s/%s" % (INPUT_SUBDIR, name), count_frames(dest)


def build_workflow(prompt, images, video, width, height, frames, steps, seed,
                   ref_image_size, filename_prefix, shift_video=None, shift_audio=3.0,
                   first_frame=None, last_frame=None, lora=None, unet_name=None):
    wf = {
        "1": {"class_type": "UNETLoader",
              "inputs": {"unet_name": unet_name or MODELS["unet"], "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": MODELS["clip"], "type": "minimax", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": MODELS["vae"]}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": MODELS["audio_vae"]}},
        "7": {"class_type": "MiniMaxH3ReferenceToVideo",
              "inputs": {"clip": ["2", 0], "vae": ["3", 0], "audio_vae": ["4", 0],
                         "prompt": prompt, "width": width, "height": height,
                         "length": frames, "ref_image_size": ref_image_size}},
        "8": {"class_type": "BasicGuider", "inputs": {"model": ["1", 0], "conditioning": ["7", 0]}},
        "9": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}},
        "10": {"class_type": "BasicScheduler",
               "inputs": {"model": ["1", 0], "scheduler": "simple",
                          "steps": steps, "denoise": 1.0}},
        "11": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "12": {"class_type": "SamplerCustomAdvanced",
               "inputs": {"noise": ["11", 0], "guider": ["8", 0],
                          "sampler": ["9", 0], "sigmas": ["10", 0],
                          "latent_image": ["7", 1]}},
        "13": {"class_type": "VAEDecode", "inputs": {"samples": ["12", 0], "vae": ["3", 0]}},
        "14": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["12", 0], "vae": ["4", 0]}},
        "15": {"class_type": "CreateVideo",
               "inputs": {"images": ["13", 0], "audio": ["14", 0], "fps": FPS, "bit_depth": 8}},
        "16": {"class_type": "SaveVideo",
               "inputs": {"video": ["15", 0], "filename_prefix": filename_prefix,
                          "format": "auto", "codec": "auto"}},
    }
    if first_frame is not None or last_frame is not None:
        # fl2va: MiniMaxH3ImageToVideo takes first/last keyframes instead of refs.
        # No audio_vae, no ref_image_size -- node "4" (audio vae) stays wired only
        # to VAEDecodeAudio ("14"), the AV latent still carries an (empty) audio track.
        node7 = {"clip": ["2", 0], "vae": ["3", 0], "prompt": prompt,
                 "width": width, "height": height, "length": frames}
        if first_frame is not None:
            wf["101"] = {"class_type": "LoadImage", "inputs": {"image": first_frame}}
            node7["first_frame"] = ["101", 0]
        if last_frame is not None:
            wf["102"] = {"class_type": "LoadImage", "inputs": {"image": last_frame}}
            node7["last_frame"] = ["102", 0]
        wf["7"] = {"class_type": "MiniMaxH3ImageToVideo", "inputs": node7}
    else:
        for i, name in enumerate(images):
            wf["100%d" % i] = {"class_type": "LoadImage", "inputs": {"image": name}}
            wf["7"]["inputs"]["ref_images.ref_image_%d" % i] = ["100%d" % i, 0]
        if video:
            wf["200"] = {"class_type": "LoadVideo", "inputs": {"file": video}}
            wf["201"] = {"class_type": "GetVideoComponents", "inputs": {"video": ["200", 0]}}
            # audio deliberately not wired: H3 generates its own foley, and the source
            # soundtrack must not reach the output (standing decision: original VO only)
            wf["7"]["inputs"]["ref_videos.ref_video_0"] = ["201", 0]

    # model chain: UNETLoader -> [LoraLoaderModelOnly] -> [MiniMaxH3SigmaShift] -> guider/scheduler
    model_ref = ["1", 0]
    if lora:
        wf["6"] = {"class_type": "LoraLoaderModelOnly",
                   "inputs": {"model": model_ref, "lora_name": lora, "strength_model": 1.0}}
        model_ref = ["6", 0]
    if shift_video is not None:
        # ModelSamplingMiniMaxH3 -- documented as the knob to reach for when motion feels
        # over- or under-cooked. Default 12. Both the guider and the scheduler must read the
        # patched model or the sigma schedule and the DiT disagree about the shift.
        wf["5"] = {"class_type": "MiniMaxH3SigmaShift",
                   "inputs": {"model": model_ref, "shift_video": shift_video,
                              "shift_audio": shift_audio}}
        model_ref = ["5", 0]
    wf["8"]["inputs"]["model"] = model_ref
    wf["10"]["inputs"]["model"] = model_ref
    return wf


def submit(wf):
    req = urllib.request.Request(
        SERVER + "/prompt",
        data=json.dumps({"prompt": wf}).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req) as r:
            return json.load(r)["prompt_id"]
    except urllib.error.HTTPError as e:
        sys.exit("submit rejected:\n" + e.read().decode())


def wait(prompt_id, poll=10):
    start = time.time()
    while True:
        with urllib.request.urlopen(SERVER + "/history/" + prompt_id) as r:
            hist = json.load(r)
        if prompt_id in hist:
            entry = hist[prompt_id]
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                for msg in status.get("messages", []):
                    if msg[0] == "execution_error":
                        sys.exit("execution error: %s" % json.dumps(msg[1], indent=2)[:2000])
                sys.exit("failed: %s" % json.dumps(status)[:2000])
            if status.get("completed"):
                return entry, time.time() - start
        elapsed = int(time.time() - start)
        print("  ... %dm%02ds" % (elapsed // 60, elapsed % 60), flush=True)
        time.sleep(poll)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="run name; drives output filename")
    ap.add_argument("--prompt-parts", help="hosted mention-chip parts JSON")
    ap.add_argument("--prompt-file", help="plain text prompt file (already local tags)")
    ap.add_argument("--ref-image", action="append", default=[],
                    help="reference image, repeatable, in <Picture N> order")
    ap.add_argument("--ref-video", help="reference clip; converted to 24fps and capped")
    ap.add_argument("--first-frame", help="fl2va: first keyframe image; cannot combine with --ref-image/--ref-video")
    ap.add_argument("--last-frame", help="fl2va: last keyframe image; cannot combine with --ref-image/--ref-video")
    ap.add_argument("--loop", action="store_true",
                    help="fl2va: use --first-frame as BOTH first and last keyframe")
    ap.add_argument("--lora", help="LoRA filename under ComfyUI models/loras, applied at strength 1.0")
    ap.add_argument("--unet", help="override the unet checkpoint (default: %s; "
                    "keyframe mode defaults to the fl2va checkpoint)" % MODELS["unet"])
    ap.add_argument("--width", type=int, default=1344)
    ap.add_argument("--height", type=int, default=768)
    ap.add_argument("--seconds", type=float, default=5.9)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=770809)
    ap.add_argument("--ref-image-size", choices=["match", "max"], default="match")
    ap.add_argument("--shift-video", type=float, default=None,
                    help="MiniMaxH3SigmaShift video flow shift (node default 12); omit to leave unpatched")
    ap.add_argument("--shift-audio", type=float, default=3.0)
    ap.add_argument("--ref-fit", choices=["stretch", "trim"], default="stretch",
                    help="stretch: retime ref to exactly the output length (default); "
                         "trim: keep source speed, lose the tail to grid snapping")
    ap.add_argument("--out-dir", default="output/fullbattle_20260808")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if bool(args.prompt_parts) == bool(args.prompt_file):
        sys.exit("give exactly one of --prompt-parts / --prompt-file")

    keyframe_mode = bool(args.first_frame or args.last_frame or args.loop)
    if keyframe_mode and (args.ref_image or args.ref_video):
        sys.exit("--first-frame/--last-frame/--loop cannot combine with --ref-image/--ref-video: "
                  "the fl2va node (MiniMaxH3ImageToVideo) does not take references")
    if args.loop:
        if not args.first_frame:
            sys.exit("--loop requires --first-frame")
        if args.last_frame:
            sys.exit("--loop already uses --first-frame as both first and last keyframe; "
                      "do not also pass --last-frame")

    if args.prompt_parts:
        with open(args.prompt_parts) as f:
            prompt = parts_to_prompt(json.load(f))
    else:
        with open(args.prompt_file) as f:
            prompt = f.read().strip()

    frames = align_frames(round(args.seconds * FPS))
    print("length %d frames = %.3fs @%dfps (requested %.2fs)"
          % (frames, frames / FPS, FPS, args.seconds))

    unet_name = args.unet
    if keyframe_mode and not unet_name:
        unet_name = "minimax_h3_fl2va_pruned_fp8_scaled.safetensors"
    if keyframe_mode:
        print("unet: %s" % (unet_name or MODELS["unet"]))

    staged = os.path.join(COMFY_ROOT, "input", INPUT_SUBDIR)
    os.makedirs(staged, exist_ok=True)
    images = [stage_image(p, staged, "%s_p%d" % (args.name, i + 1))
              for i, p in enumerate(args.ref_image)]
    video = None
    if args.ref_video:
        video, got = stage_video(args.ref_video, staged, args.name, frames, args.ref_fit)
        kept = align_frames_down(min(got, frames))
        print("ref video: %d frames -> node keeps %d = %.3fs%s"
              % (got, kept, kept / FPS,
                 "" if kept == frames else "  <-- SHORTER THAN OUTPUT"))

    first_frame_staged = None
    last_frame_staged = None
    if args.loop:
        first_frame_staged = stage_image(args.first_frame, staged, "first")
        last_frame_staged = stage_image(args.first_frame, staged, "last")
    else:
        if args.first_frame:
            first_frame_staged = stage_image(args.first_frame, staged, "first")
        if args.last_frame:
            last_frame_staged = stage_image(args.last_frame, staged, "last")

    prefix = "video/%s" % args.name
    wf = build_workflow(prompt, images, video, args.width, args.height, frames,
                        args.steps, args.seed, args.ref_image_size, prefix,
                        args.shift_video, args.shift_audio,
                        first_frame=first_frame_staged, last_frame=last_frame_staged,
                        lora=args.lora, unet_name=unet_name)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, args.name + "_prompt.txt"), "w") as f:
        f.write(prompt + "\n")
    with open(os.path.join(args.out_dir, args.name + "_workflow.json"), "w") as f:
        json.dump(wf, f, indent=2)

    if args.dry_run:
        print("--- prompt ---")
        print(prompt)
        print("--- dry run, not submitted ---")
        return

    pid = submit(wf)
    print("submitted %s" % pid)
    entry, elapsed = wait(pid)
    print("done in %dm%02ds" % (int(elapsed) // 60, int(elapsed) % 60))

    for node_out in entry.get("outputs", {}).values():
        for item in node_out.get("images", []) + node_out.get("videos", []):
            src = os.path.join(COMFY_ROOT, "output",
                               item.get("subfolder", ""), item["filename"])
            dst = os.path.join(args.out_dir,
                               args.name + os.path.splitext(item["filename"])[1])
            shutil.copyfile(src, dst)
            print("-> %s" % dst)


if __name__ == "__main__":
    main()

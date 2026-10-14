#!/usr/bin/env python3
"""
comfy_animate.py — headless Wan 2.2 Animate driver via ComfyUI's /prompt API.

The PRODUCTION v2v gen engine (supersedes the WanGP queue.zip driver). Maps a
character reference image onto a driving video's real motion, full-frame restyle
(no background/person masking). Built on the classics ComfyUI-API pattern
(submit workflow -> poll /history -> pull output) and WanVideoWrapper's Wan
Animate nodes, with the Lightx2v 4-step distill LoRA for fast generation.

Graph (all WanVideoWrapper + core VHS, no controlnet_aux/SAM2):
  VHS_LoadVideo(driving) -> WanVideoUniAnimateDWPoseDetector -> pose images
  LoadImage(ref) -> WanVideoClipVisionEncode
  WanVideoAnimateEmbeds(vae, ref, pose, clip) -> WanVideoSampler(animate model
  + lightx2v lora, 6 steps) -> WanVideoDecode -> VHS_VideoCombine -> mp4

Usage (uses the system python with `requests`; ComfyUI must be running):
  python v2v/comfy_animate.py --char <ref.png> --video <driving.mp4> \
      --out <result.mp4> [--width 480 --height 832 --frames 81 --steps 6 --seed 42]
"""
import argparse
import json
import os
import shutil
import sys
import time
import uuid
from pathlib import Path

import requests

COMFYUI_URL = "http://localhost:8188"
COMFYUI_DIR = Path("~/ComfyUI")
COMFYUI_INPUT = COMFYUI_DIR / "input"
COMFYUI_OUTPUT = COMFYUI_DIR / "output"

# Model files (relative to their ComfyUI model dirs)
ANIMATE_MODEL = "WanVideo/2_2/Wan2_2-Animate-14B_fp8_e4m3fn_scaled_KJ.safetensors"
VAE_MODEL = "wanvideo/Wan2_1_VAE_bf16.safetensors"
CLIP_VISION = "clip_vision_h.safetensors"
TEXT_ENCODER = "umt5_xxl_fp16.safetensors"
LIGHTX2V_LORA = "WanVideo/Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors"

# Proven negative prompt from the WanVideoWrapper Animate example
DEFAULT_NEG = ("色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
               "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
               "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
               "杂乱的背景，三条腿，背景人很多，倒着走")


def build_workflow(ref_name, video_name, positive, negative, width, height,
                   num_frames, steps, seed, fps, blocks_to_swap=20, lora_strength=1.2):
    """Construct the ComfyUI API-format prompt graph. Node ids are strings;
    links are [from_node_id, output_slot]."""
    return {
        "1": {"class_type": "WanVideoModelLoader", "inputs": {
            "model": ANIMATE_MODEL, "base_precision": "fp16_fast",
            "quantization": "disabled", "load_device": "offload_device",
            "attention_mode": "sageattn", "lora": ["4", 0], "block_swap_args": ["5", 0]}},
        "2": {"class_type": "WanVideoVAELoader", "inputs": {
            "model_name": VAE_MODEL, "precision": "bf16"}},
        "3": {"class_type": "CLIPVisionLoader", "inputs": {"clip_name": CLIP_VISION}},
        "4": {"class_type": "WanVideoLoraSelect", "inputs": {
            "lora": LIGHTX2V_LORA, "strength": lora_strength,
            "low_mem_load": False, "merge_loras": True}},
        "5": {"class_type": "WanVideoBlockSwap", "inputs": {
            "blocks_to_swap": blocks_to_swap, "offload_img_emb": False,
            "offload_txt_emb": False,
            # must pass all optionals: omitted ones become None and the wrapper's
            # block_swap does `vace_blocks_to_swap > 0` (TypeError on None).
            "use_non_blocking": True, "vace_blocks_to_swap": 0,
            "prefetch_blocks": 0, "block_swap_debug": False}},
        "6": {"class_type": "LoadImage", "inputs": {"image": ref_name}},
        "7": {"class_type": "VHS_LoadVideo", "inputs": {
            "video": video_name, "force_rate": 0, "custom_width": 0,
            "custom_height": 0, "frame_load_cap": num_frames,
            "skip_first_frames": 0, "select_every_nth": 1}},
        "8": {"class_type": "WanVideoUniAnimateDWPoseDetector", "inputs": {
            "pose_images": ["7", 0], "score_threshold": 0.3, "stick_width": 4,
            "draw_body": True, "body_keypoint_size": 4, "draw_feet": True,
            "draw_hands": True, "hand_keypoint_size": 4, "colorspace": "RGB",
            "handle_not_detected": "empty", "draw_head": True}},
        "9": {"class_type": "WanVideoClipVisionEncode", "inputs": {
            "clip_vision": ["3", 0], "image_1": ["6", 0], "strength_1": 1.0,
            "strength_2": 1.0, "crop": "center", "combine_embeds": "average",
            "force_offload": True}},
        "10": {"class_type": "WanVideoTextEncodeCached", "inputs": {
            "model_name": TEXT_ENCODER, "precision": "bf16",
            "positive_prompt": positive, "negative_prompt": negative,
            # fp8 quantization halves umt5 (~11GB->~5.5GB) to avoid OOM next to the 18GB model
            "quantization": "fp8_e4m3fn", "use_disk_cache": True, "device": "gpu"}},
        "11": {"class_type": "WanVideoAnimateEmbeds", "inputs": {
            "vae": ["2", 0], "width": width, "height": height,
            "num_frames": num_frames, "force_offload": False,
            # single window for a shot: window < num_frames splits into multiple
            # windows and OVER-generates (77<81 produced 154 frames). Per-shot
            # clips are short; cap window at num_frames. Long takes get scene-split.
            "frame_window_size": num_frames, "colormatch": "disabled",
            "pose_strength": 1.0, "face_strength": 1.0,
            "clip_embeds": ["9", 0], "ref_images": ["6", 0],
            "pose_images": ["8", 0]}},
        "12": {"class_type": "WanVideoSampler", "inputs": {
            "model": ["1", 0], "image_embeds": ["11", 0], "steps": steps,
            "cfg": 1.0, "shift": 5.0, "seed": seed, "force_offload": True,
            "scheduler": "dpm++_sde", "riflex_freq_index": 0,
            "text_embeds": ["10", 0]}},
        "13": {"class_type": "WanVideoDecode", "inputs": {
            "vae": ["2", 0], "samples": ["12", 0], "enable_vae_tiling": False,
            "tile_x": 272, "tile_y": 272, "tile_stride_x": 144, "tile_stride_y": 128}},
        "14": {"class_type": "VHS_VideoCombine", "inputs": {
            "images": ["13", 0], "frame_rate": fps, "loop_count": 0,
            "filename_prefix": "v2v_animate", "format": "video/h264-mp4",
            "pingpong": False, "save_output": True}},
    }


def check_comfyui():
    try:
        requests.get(f"{COMFYUI_URL}/system_stats", timeout=5).raise_for_status()
        return True
    except Exception as e:
        print(f"ERROR: ComfyUI not reachable at {COMFYUI_URL}: {e}", file=sys.stderr)
        return False


def stage_input(src, dest_name):
    dest = COMFYUI_INPUT / dest_name
    COMFYUI_INPUT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return dest_name


def submit(workflow):
    payload = {"prompt": workflow, "client_id": str(uuid.uuid4())}
    r = requests.post(f"{COMFYUI_URL}/prompt", json=payload, timeout=30)
    if r.status_code != 200:
        print(f"ERROR: ComfyUI rejected prompt ({r.status_code}):", file=sys.stderr)
        try:
            print(json.dumps(r.json(), indent=2)[:4000], file=sys.stderr)
        except Exception:
            print(r.text[:2000], file=sys.stderr)
        r.raise_for_status()
    return r.json()["prompt_id"]


def poll(prompt_id, timeout=3600, interval=5):
    start = time.time()
    while time.time() - start < timeout:
        try:
            h = requests.get(f"{COMFYUI_URL}/history/{prompt_id}", timeout=10).json()
            if prompt_id in h:
                entry = h[prompt_id]
                status = entry.get("status", {})
                if status.get("status_str") == "error":
                    msgs = status.get("messages", [])
                    raise RuntimeError(f"ComfyUI execution error: {json.dumps(msgs)[:2000]}")
                if entry.get("outputs"):
                    return entry
        except requests.RequestException:
            pass
        el = time.time() - start
        print(f"  ... generating ({el:.0f}s)", flush=True)
        time.sleep(interval)
    raise TimeoutError(f"generation did not complete within {timeout}s")


def retrieve(entry, out_path):
    outputs = entry.get("outputs", {})
    vids = []
    for node_id, node_out in outputs.items():
        for key in ("gifs", "videos", "images"):
            if node_out.get(key):
                vids = node_out[key]
                break
        if vids:
            break
    if not vids:
        raise ValueError(f"no output video found; outputs keys={list(outputs.keys())}")
    v = vids[0]
    src = COMFYUI_OUTPUT / v.get("subfolder", "") / v["filename"]
    if out_path:
        shutil.copy2(src, out_path)
        return out_path
    return str(src)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--char", required=True, help="character reference image")
    ap.add_argument("--video", required=True, help="driving motion video")
    ap.add_argument("--out", default=None, help="destination for the result mp4")
    ap.add_argument("--prompt", default="female flame mage engulfed in flames in "
                    "hand to hand combat with a male opponent, anime, dynamic, "
                    "cinematic, dramatic lighting")
    ap.add_argument("--negative", default=DEFAULT_NEG)
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--height", type=int, default=832)
    ap.add_argument("--frames", type=int, default=81, help="must be 4n+1")
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--blocks-swap", type=int, default=20)
    ap.add_argument("--timeout", type=int, default=3600)
    args = ap.parse_args()

    if not check_comfyui():
        sys.exit(1)
    for f in (args.char, args.video):
        if not os.path.isfile(f):
            print(f"ERROR: not found: {f}", file=sys.stderr); sys.exit(1)

    ref_name = stage_input(args.char, f"v2v_ref_{os.getpid()}.png")
    vid_name = stage_input(args.video, f"v2v_drive_{os.getpid()}{os.path.splitext(args.video)[1]}")
    print(f"[comfy] staged ref={ref_name} video={vid_name}")
    wf = build_workflow(ref_name, vid_name, args.prompt, args.negative,
                        args.width, args.height, args.frames, args.steps,
                        args.seed, args.fps, args.blocks_swap)
    print(f"[comfy] {args.width}x{args.height} {args.frames}f {args.steps}steps "
          f"seed={args.seed} lightx2v -> submitting")
    t0 = time.time()
    pid = submit(wf)
    print(f"[comfy] prompt_id={pid}")
    entry = poll(pid, timeout=args.timeout)
    out = retrieve(entry, args.out)
    print(f"[comfy] DONE in {time.time()-t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()

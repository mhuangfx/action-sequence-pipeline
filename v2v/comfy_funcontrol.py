#!/usr/bin/env python3
"""
comfy_funcontrol.py — Wan 2.2 Fun-Control structure-conditioned v2v test.

Tests the fix for the Wan-Animate limitation (Animate only restyles the
ref-matched character, leaves background/other people realistic). Fun-Control
regenerates the WHOLE frame from a control signal + a style anchor:

  driving video --DepthAnythingV2--> full-frame depth ---------\
                                                                 WanVideoAddControlEmbeds
  styled ref frame --ImageToVideoEncode (start_image)----------/   -> Sampler -> decode
                     (sets the art style for the entire frame)

Because depth covers the whole frame and the start_image styles everything, the
environment + both fighters should restyle (unlike Animate). The restyle <->
structure-fidelity tradeoff is dialed with --control-weight (WanVideoEncode
latent_strength): higher = closer to source structure / tamer restyle.

Single-model (HIGH expert only) + Lightx2v T2V distill for a fast test; add the
LOW refiner later for final quality.

Usage (ComfyUI must be running):
  python v2v/comfy_funcontrol.py --char <styled_ref.png> --video <driving.mp4> \
      --out <result.mp4> [--control-weight 1.0] [--steps 6] [--width 480 --height 832]
"""
import argparse
import os
import sys
import time

# reuse the ComfyUI http helpers + constants from the animate driver
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from comfy_animate import (check_comfyui, stage_input, submit, poll, retrieve,
                           DEFAULT_NEG)

FUN_MODEL = "Fun/Wan2_2-Fun-Control-A14B-HIGH_fp8_e4m3fn_scaled_KJ_fixed.safetensors"
VAE_MODEL = "wanvideo/Wan2_1_VAE_bf16.safetensors"
TEXT_ENCODER = "umt5_xxl_fp16.safetensors"
T2V_LORA = "WanVideo/Lightx2v/lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank32_bf16.safetensors"
DEPTH_MODEL = "depth_anything_v2_vitl_fp16.safetensors"  # auto-downloads on first use


def build_workflow(ref_name, video_name, positive, negative, width, height,
                   num_frames, steps, seed, fps, control_weight, blocks_to_swap=20,
                   lora_strength=1.5):
    def resize(src_node):
        return {"image": src_node, "width": width, "height": height,
                "upscale_method": "lanczos", "keep_proportion": "crop",
                "pad_color": "0, 0, 0", "crop_position": "center",
                "divisible_by": 16, "device": "cpu"}
    return {
        "1": {"class_type": "WanVideoModelLoader", "inputs": {
            "model": FUN_MODEL, "base_precision": "fp16_fast",
            "quantization": "fp8_e4m3fn_scaled", "load_device": "offload_device",
            "attention_mode": "sageattn", "lora": ["4", 0], "block_swap_args": ["5", 0]}},
        "2": {"class_type": "WanVideoVAELoader", "inputs": {
            "model_name": VAE_MODEL, "precision": "bf16"}},
        "4": {"class_type": "WanVideoLoraSelect", "inputs": {
            "lora": T2V_LORA, "strength": lora_strength,
            "low_mem_load": False, "merge_loras": True}},
        "5": {"class_type": "WanVideoBlockSwap", "inputs": {
            "blocks_to_swap": blocks_to_swap, "offload_img_emb": False,
            "offload_txt_emb": False, "use_non_blocking": True,
            "vace_blocks_to_swap": 0, "prefetch_blocks": 0, "block_swap_debug": False}},
        # driving video -> resize -> depth (full-frame control signal)
        "6": {"class_type": "VHS_LoadVideo", "inputs": {
            "video": video_name, "force_rate": 0, "custom_width": 0,
            "custom_height": 0, "frame_load_cap": num_frames,
            "skip_first_frames": 0, "select_every_nth": 1}},
        "7": {"class_type": "ImageResizeKJv2", "inputs": resize(["6", 0])},
        "8": {"class_type": "DepthAnythingV2Nikosis", "inputs": {
            "images": ["7", 0], "model": DEPTH_MODEL,
            "manual_resolution": False, "resolution": 1024}},
        # styled ref frame -> resize -> start_image (style anchor for whole frame)
        "9": {"class_type": "LoadImage", "inputs": {"image": ref_name}},
        "10": {"class_type": "ImageResizeKJv2", "inputs": resize(["9", 0])},
        "11": {"class_type": "WanVideoImageToVideoEncode", "inputs": {
            "width": width, "height": height, "num_frames": num_frames,
            "noise_aug_strength": 0.0, "start_latent_strength": 1.0,
            "end_latent_strength": 1.0, "force_offload": True,
            "fun_or_fl2v_model": True, "vae": ["2", 0], "start_image": ["10", 0]}},
        # encode depth control -> latents (control-weight dial = latent_strength)
        "12": {"class_type": "WanVideoEncode", "inputs": {
            "vae": ["2", 0], "image": ["8", 0], "enable_vae_tiling": False,
            "tile_x": 272, "tile_y": 272, "tile_stride_x": 144, "tile_stride_y": 128,
            "noise_aug_strength": 0.0, "latent_strength": control_weight}},
        "13": {"class_type": "WanVideoAddControlEmbeds", "inputs": {
            "embeds": ["11", 0], "start_percent": 0.0, "end_percent": 1.0,
            "latents": ["12", 0]}},
        "14": {"class_type": "WanVideoTextEncodeCached", "inputs": {
            "model_name": TEXT_ENCODER, "precision": "bf16",
            "positive_prompt": positive, "negative_prompt": negative,
            "quantization": "fp8_e4m3fn", "use_disk_cache": True, "device": "gpu"}},
        "15": {"class_type": "WanVideoSampler", "inputs": {
            "model": ["1", 0], "image_embeds": ["13", 0], "steps": steps,
            "cfg": 1.0, "shift": 5.0, "seed": seed, "force_offload": True,
            "scheduler": "dpm++_sde", "riflex_freq_index": 0, "text_embeds": ["14", 0]}},
        "16": {"class_type": "WanVideoDecode", "inputs": {
            "vae": ["2", 0], "samples": ["15", 0], "enable_vae_tiling": False,
            "tile_x": 272, "tile_y": 272, "tile_stride_x": 144, "tile_stride_y": 128}},
        "17": {"class_type": "VHS_VideoCombine", "inputs": {
            "images": ["16", 0], "frame_rate": fps, "loop_count": 0,
            "filename_prefix": "v2v_funcontrol", "format": "video/h264-mp4",
            "pingpong": False, "save_output": True}},
    }


def build_workflow_moe(ref_name, video_name, positive, negative, width, height,
                       num_frames, steps, switch_step, seed, fps, control_weight,
                       blocks_to_swap=25, lora_strength=1.5):
    """Proper Wan 2.2 MoE: HIGH expert denoises steps [0,switch), LOW expert
    refines [switch,end] from HIGH's latents. Both experts loaded (block-swapped),
    run sequentially so only one is GPU-resident at a time."""
    LOW_MODEL = FUN_MODEL.replace("HIGH", "LOW")
    def resize(src_node):
        return {"image": src_node, "width": width, "height": height,
                "upscale_method": "lanczos", "keep_proportion": "crop",
                "pad_color": "0, 0, 0", "crop_position": "center",
                "divisible_by": 16, "device": "cpu"}
    bs = {"blocks_to_swap": blocks_to_swap, "offload_img_emb": False,
          "offload_txt_emb": False, "use_non_blocking": True,
          "vace_blocks_to_swap": 0, "prefetch_blocks": 0, "block_swap_debug": False}
    lora = lambda: {"lora": T2V_LORA, "strength": lora_strength,
                    "low_mem_load": False, "merge_loras": True}
    return {
        "5": {"class_type": "WanVideoBlockSwap", "inputs": bs},
        "40": {"class_type": "WanVideoLoraSelect", "inputs": lora()},   # HIGH lora
        "41": {"class_type": "WanVideoLoraSelect", "inputs": lora()},   # LOW lora
        "1": {"class_type": "WanVideoModelLoader", "inputs": {
            "model": FUN_MODEL, "base_precision": "fp16_fast",
            "quantization": "fp8_e4m3fn_scaled", "load_device": "offload_device",
            "attention_mode": "sageattn", "lora": ["40", 0], "block_swap_args": ["5", 0]}},
        "18": {"class_type": "WanVideoModelLoader", "inputs": {
            "model": LOW_MODEL, "base_precision": "fp16_fast",
            "quantization": "fp8_e4m3fn_scaled", "load_device": "offload_device",
            "attention_mode": "sageattn", "lora": ["41", 0], "block_swap_args": ["5", 0]}},
        "2": {"class_type": "WanVideoVAELoader", "inputs": {
            "model_name": VAE_MODEL, "precision": "bf16"}},
        "6": {"class_type": "VHS_LoadVideo", "inputs": {
            "video": video_name, "force_rate": 0, "custom_width": 0,
            "custom_height": 0, "frame_load_cap": num_frames,
            "skip_first_frames": 0, "select_every_nth": 1}},
        "7": {"class_type": "ImageResizeKJv2", "inputs": resize(["6", 0])},
        "8": {"class_type": "DepthAnythingV2Nikosis", "inputs": {
            "images": ["7", 0], "model": DEPTH_MODEL,
            "manual_resolution": False, "resolution": 1024}},
        "9": {"class_type": "LoadImage", "inputs": {"image": ref_name}},
        "10": {"class_type": "ImageResizeKJv2", "inputs": resize(["9", 0])},
        "11": {"class_type": "WanVideoImageToVideoEncode", "inputs": {
            "width": width, "height": height, "num_frames": num_frames,
            "noise_aug_strength": 0.0, "start_latent_strength": 1.0,
            "end_latent_strength": 1.0, "force_offload": True,
            "fun_or_fl2v_model": True, "vae": ["2", 0], "start_image": ["10", 0]}},
        "12": {"class_type": "WanVideoEncode", "inputs": {
            "vae": ["2", 0], "image": ["8", 0], "enable_vae_tiling": False,
            "tile_x": 272, "tile_y": 272, "tile_stride_x": 144, "tile_stride_y": 128,
            "noise_aug_strength": 0.0, "latent_strength": control_weight}},
        "13": {"class_type": "WanVideoAddControlEmbeds", "inputs": {
            "embeds": ["11", 0], "start_percent": 0.0, "end_percent": 1.0,
            "latents": ["12", 0]}},
        "14": {"class_type": "WanVideoTextEncodeCached", "inputs": {
            "model_name": TEXT_ENCODER, "precision": "bf16",
            "positive_prompt": positive, "negative_prompt": negative,
            "quantization": "fp8_e4m3fn", "use_disk_cache": True, "device": "gpu"}},
        # HIGH expert: steps [0, switch)
        "15": {"class_type": "WanVideoSampler", "inputs": {
            "model": ["1", 0], "image_embeds": ["13", 0], "steps": steps,
            "cfg": 1.0, "shift": 5.0, "seed": seed, "force_offload": True,
            "scheduler": "dpm++_sde", "riflex_freq_index": 0, "text_embeds": ["14", 0],
            "start_step": 0, "end_step": switch_step}},
        # LOW expert: refine [switch, end] from HIGH's latents
        "24": {"class_type": "WanVideoSampler", "inputs": {
            "model": ["18", 0], "image_embeds": ["13", 0], "steps": steps,
            "cfg": 1.0, "shift": 5.0, "seed": seed, "force_offload": True,
            "scheduler": "dpm++_sde", "riflex_freq_index": 0, "text_embeds": ["14", 0],
            "samples": ["15", 0], "start_step": switch_step, "end_step": -1}},
        "16": {"class_type": "WanVideoDecode", "inputs": {
            "vae": ["2", 0], "samples": ["24", 0], "enable_vae_tiling": False,
            "tile_x": 272, "tile_y": 272, "tile_stride_x": 144, "tile_stride_y": 128}},
        "17": {"class_type": "VHS_VideoCombine", "inputs": {
            "images": ["16", 0], "frame_rate": fps, "loop_count": 0,
            "filename_prefix": "v2v_funcontrol_moe", "format": "video/h264-mp4",
            "pingpong": False, "save_output": True}},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--char", required=True, help="STYLED reference frame (style anchor)")
    ap.add_argument("--video", required=True, help="driving motion video")
    ap.add_argument("--out", default=None)
    ap.add_argument("--prompt", default="anime fantasy battle, fire mage Vesta with "
                    "glowing ember armor and flame-red hair fighting, dramatic "
                    "cinematic lighting, highly detailed anime style")
    ap.add_argument("--negative", default=DEFAULT_NEG)
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--height", type=int, default=832)
    ap.add_argument("--frames", type=int, default=81)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--control-weight", type=float, default=1.0,
                    help="depth control latent_strength; lower = freer restyle")
    ap.add_argument("--blocks-swap", type=int, default=20)
    ap.add_argument("--moe", action="store_true", help="proper 2-expert MoE (HIGH+LOW)")
    ap.add_argument("--switch-step", type=int, default=2, help="MoE HIGH->LOW step")
    ap.add_argument("--timeout", type=int, default=3600)
    args = ap.parse_args()

    if not check_comfyui():
        sys.exit(1)
    for f in (args.char, args.video):
        if not os.path.isfile(f):
            print(f"ERROR: not found: {f}", file=sys.stderr); sys.exit(1)

    ref_name = stage_input(args.char, f"fc_ref_{os.getpid()}.png")
    vid_name = stage_input(args.video, f"fc_drive_{os.getpid()}{os.path.splitext(args.video)[1]}")
    if args.moe:
        wf = build_workflow_moe(ref_name, vid_name, args.prompt, args.negative,
                                args.width, args.height, args.frames, args.steps,
                                args.switch_step, args.seed, args.fps,
                                args.control_weight, max(args.blocks_swap, 25))
    else:
        wf = build_workflow(ref_name, vid_name, args.prompt, args.negative,
                            args.width, args.height, args.frames, args.steps,
                            args.seed, args.fps, args.control_weight, args.blocks_swap)
    print(f"[funcontrol] {'MoE ' if args.moe else ''}{args.width}x{args.height} "
          f"{args.frames}f {args.steps}steps ctrl_weight={args.control_weight} "
          f"seed={args.seed} -> submitting")
    t0 = time.time()
    pid = submit(wf)
    print(f"[funcontrol] prompt_id={pid}")
    entry = poll(pid, timeout=args.timeout)
    out = retrieve(entry, args.out)
    print(f"[funcontrol] DONE in {time.time()-t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()

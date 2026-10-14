#!/usr/bin/env python3
"""
scene_split.py — PySceneDetect shot segmentation for the v2v pipeline.

Splits a driving fight-reference video into continuous SHOTS (you cannot v2v
across a hard cut). Each shot is restyled + animated independently, then the
shots are concatenated back together (see docs/v2v_pipeline_handoff.md §2).

Run with the isolated v2v venv:
    v2v/venv/bin/python v2v/scene_split.py <video> [--threshold 27] [--min-len 0.4]
                                          [--outdir <dir>] [--split] [--json]

Output: a list of shots {index, start_s, end_s, dur_s, start_frame, end_frame}.
With --split, also writes per-shot mp4 clips (stream-copy, fast) to <outdir>.
With --json, prints the shot list as JSON to stdout (for the orchestrator).
"""
import argparse
import json
import os
import sys

from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector


def detect_shots(video_path, threshold=27.0, min_scene_len_s=0.4):
    video = open_video(video_path)
    fps = video.frame_rate
    sm = SceneManager()
    # min_scene_len is in frames; convert from seconds.
    min_len_frames = max(1, int(round(min_scene_len_s * fps)))
    sm.add_detector(ContentDetector(threshold=threshold, min_scene_len=min_len_frames))
    sm.detect_scenes(video, show_progress=False)
    scene_list = sm.get_scene_list()
    shots = []
    for i, (start, end) in enumerate(scene_list):
        shots.append({
            "index": i,
            "start_s": round(start.seconds, 3),
            "end_s": round(end.seconds, 3),
            "dur_s": round(end.seconds - start.seconds, 3),
            "start_frame": start.frame_num,
            "end_frame": end.frame_num,
        })
    return shots, fps


def split_clips(video_path, shots, outdir):
    """Stream-copy each shot to its own mp4 (fast, lossless cut at keyframes)."""
    import subprocess
    os.makedirs(outdir, exist_ok=True)
    base = os.path.splitext(os.path.basename(video_path))[0]
    paths = []
    for s in shots:
        out = os.path.join(outdir, f"{base}_shot{s['index']:03d}.mp4")
        # Re-encode (not stream-copy) so the cut is frame-accurate at shot
        # boundaries — v2v needs exact frame-1 alignment, keyframe-snapping would
        # drift the first frame.
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error",
            "-ss", str(s["start_s"]), "-to", str(s["end_s"]),
            "-i", video_path,
            "-c:v", "libx264", "-crf", "16", "-preset", "fast",
            "-an", out,
        ], check=True)
        s["clip"] = out
        paths.append(out)
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--threshold", type=float, default=27.0,
                    help="ContentDetector threshold (lower = more cuts; default 27)")
    ap.add_argument("--min-len", type=float, default=0.4,
                    help="minimum shot length in seconds (default 0.4)")
    ap.add_argument("--outdir", default=None, help="dir for --split clips")
    ap.add_argument("--split", action="store_true", help="also write per-shot mp4s")
    ap.add_argument("--json", action="store_true", help="emit JSON to stdout")
    args = ap.parse_args()

    if not os.path.isfile(args.video):
        print(f"ERROR: not a file: {args.video}", file=sys.stderr)
        sys.exit(1)

    shots, fps = detect_shots(args.video, args.threshold, args.min_len)

    if args.split:
        outdir = args.outdir or os.path.join(
            os.path.dirname(args.video) or ".", "shots")
        split_clips(args.video, shots, outdir)

    if args.json:
        print(json.dumps({"fps": fps, "n_shots": len(shots), "shots": shots}, indent=2))
    else:
        print(f"{os.path.basename(args.video)}: fps={fps:.3f}  shots={len(shots)}")
        for s in shots:
            extra = f"  -> {s.get('clip')}" if s.get("clip") else ""
            print(f"  shot {s['index']:>2}: {s['start_s']:>6.2f}s -> {s['end_s']:>6.2f}s  "
                  f"({s['dur_s']:.2f}s, f{s['start_frame']}-{s['end_frame']}){extra}")


if __name__ == "__main__":
    main()

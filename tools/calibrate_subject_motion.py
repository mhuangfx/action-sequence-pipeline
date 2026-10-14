#!/usr/bin/env python3
"""calibrate_subject_motion.py — ground-truth the sign convention of shot_breakdown --subject.

camera_motion's signs turned out inverted on all three axes when read naively (2026-08-08), so
subject direction gets the same treatment: render clips with a KNOWN subject direction over a
static camera and check what the tool reports. Re-run this instead of reasoning about signs.

Usage: .venv/bin/python tools/calibrate_subject_motion.py [workdir]
"""
import json, subprocess, sys, tempfile
from pathlib import Path

work = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp(prefix="subjcal_"))
work.mkdir(parents=True, exist_ok=True)
bg = work / "bg.png"

subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "nullsrc=s=640x360",
                "-vf", "geq=random(1)*255:128:128", "-frames:v", "1", str(bg)], check=True)

CASES = [
    ("subject moves SCREEN RIGHT", "60+t*150", "140"),
    ("subject moves SCREEN LEFT",  "500-t*150", "140"),
    ("subject moves DOWN",         "280",       "40+t*90"),
    ("subject moves UP",           "280",       "260-t*90"),
]

print(f"workdir {work}\n")
for name, xexpr, yexpr in CASES:
    slug = name.lower().replace(" ", "_")
    clip = work / f"{slug}.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-loop", "1", "-i", str(bg),
         "-f", "lavfi", "-i", "color=c=red:s=90x90:d=3",
         "-filter_complex", f"[0:v]trim=duration=3,fps=24[b];[b][1:v]overlay=x='{xexpr}':y='{yexpr}'",
         "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", str(work / f"{slug}_raw.mp4")],
        check=True)
    # PySceneDetect returns zero shots for a cut-free clip, so give it a leading cut to detect.
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=640x360:d=0.5:r=24",
         "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", str(work / "black.mp4")], check=True)
    lst = work / f"{slug}.txt"
    lst.write_text(f"file '{work/'black.mp4'}'\nfile '{work/(slug+'_raw.mp4')}'\n")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
                    "-c", "copy", str(clip)], check=True)
    out = work / slug
    subprocess.run([sys.executable, "tools/shot_breakdown.py", str(clip),
                    "--out-dir", str(out), "--motion", "--subject"],
                   capture_output=True, text=True)
    d = json.load(open(out / "shots.json"))
    moving = [sh for sh in d["shots"] if sh["start"] > 0.3] or d["shots"]
    s = (moving[0].get("subject") or {}) if moving else {}
    print(f"{name:28s} -> dx {s.get('dx'):+6} dy {s.get('dy'):+6}   reported: {s.get('drive')!r}")
print("\nIf a reported label contradicts the case name, flip the sign test in subject_motion().")

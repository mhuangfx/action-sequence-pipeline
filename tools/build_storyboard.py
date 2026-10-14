#!/usr/bin/env python3
"""build_storyboard.py — tile shot frames into a numbered storyboard grid.

The grid shape must land on a ratio the image model can actually render: Hailuo's widest is
21:9, so a single row of 16:9 panels (4.4:1 for five) is unrenderable and gets re-framed,
which destroys the composition the board exists to preserve. 3x4 = exactly 4:3, 3x3 = 16:9.

Usage: build_storyboard.py FRAMES_DIR OUT.png --cols 3 --rows 3 [--panel-width 1024]
"""
import argparse, glob, os
from PIL import Image, ImageDraw, ImageFont

p = argparse.ArgumentParser()
p.add_argument("frames_dir"); p.add_argument("out")
p.add_argument("--cols", type=int, default=3); p.add_argument("--rows", type=int, default=4)
p.add_argument("--panel-width", type=int, default=1024)
p.add_argument("--border", type=int, default=6)
a = p.parse_args()

PW = a.panel_width; PH = round(PW * 9 / 16)
frames = sorted(glob.glob(os.path.join(a.frames_dir, "*.jpg")))[: a.cols * a.rows]
W, H = a.cols * PW, a.rows * PH
board = Image.new("RGB", (W, H), (32, 32, 32))
d = ImageDraw.Draw(board)

font = None
for f in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
          "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"):
    if os.path.exists(f):
        font = ImageFont.truetype(f, max(32, PW // 16)); break
font = font or ImageFont.load_default()

B = a.border
for i in range(a.cols * a.rows):
    c, r = i % a.cols, i // a.cols
    x, y = c * PW, r * PH
    if i < len(frames):
        im = Image.open(frames[i]).convert("RGB").resize((PW - 2 * B, PH - 2 * B), Image.LANCZOS)
        board.paste(im, (x + B, y + B))
        d.text((x + B + 18, y + B + 8), str(i + 1), fill=(255, 230, 0),
               font=font, stroke_width=4, stroke_fill=(0, 0, 0))
    else:
        d.rectangle([x + B, y + B, x + PW - 1 - B, y + PH - 1 - B], fill=(70, 70, 70))
    d.rectangle([x, y, x + PW - 1, y + PH - 1], outline=(255, 255, 255), width=B)

board.save(a.out)
print(a.out, board.size, round(board.size[0] / board.size[1], 4), f"{len(frames)} panels")

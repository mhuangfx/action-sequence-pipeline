#!/usr/bin/env python3
# Auto-reframe prototype: qwen2.5vl per-frame subject tracking -> virtual-camera
# keyframes -> 4:5 crop render. Replaces the hand-eyeballed KF table in
# v2v_test/build_reframe.py.
#
# Usage: python3 auto_track.py SRC OUT [--fps 1] [--crop-w 576] [--out-w 1080] [--out-h 1350]
import base64, json, math, os, subprocess, sys, tempfile, time, urllib.request

SRC = sys.argv[1]
OUT = sys.argv[2]
def arg(flag, default):
    return type(default)(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default
FPS    = arg('--fps', 1.0)
CROP_W = arg('--crop-w', 576)
OUT_W  = arg('--out-w', 1080)
OUT_H  = arg('--out-h', 1350)
SRC_W, SRC_H = 1280, 720
XMAX = SRC_W - CROP_W
OLLAMA = 'http://localhost:11434/api/generate'
MODEL = 'qwen2.5vl:7b'
HERO_W = 0.6  # camera bias toward the heroine in two-figure shots

PROMPT = (
    'Detect each distinct character figure in this 1280x720 anime frame. '
    'Output ONLY a JSON array, one object per figure: '
    '[{"label": "<short description>", "bbox_2d": [x1, y1, x2, y2]}]. '
    'Possible figures: a red-haired woman wreathed in fire/orange energy (label it "hero"), '
    'and one or more black shadow creatures or a colossal shadow titan with purple glow (label "foe"). '
    'Tight boxes. If a figure fills nearly the whole frame, still box it.'
)

def detect(path):
    img = base64.b64encode(open(path, 'rb').read()).decode()
    body = json.dumps({"model": MODEL, "prompt": PROMPT, "images": [img],
                       "stream": False, "options": {"temperature": 0}}).encode()
    req = urllib.request.Request(OLLAMA, data=body, headers={'Content-Type': 'application/json'})
    txt = json.load(urllib.request.urlopen(req, timeout=300))['response']
    txt = txt[txt.find('['): txt.rfind(']') + 1]
    try:
        return json.loads(txt)
    except Exception:
        return []

def target_x(dets):
    """Camera-target policy: hero-biased blend; close-ups/no-detection -> center."""
    boxes = []
    for d in dets:
        b = d.get('bbox_2d') or d.get('bbox') or []
        if len(b) != 4:
            continue
        x1, y1, x2, y2 = b
        w = x2 - x1
        if w >= 0.85 * SRC_W:        # full-frame box = close-up -> centered camera
            continue
        cx = (x1 + x2) / 2
        label = str(d.get('label', '')).lower()
        is_hero = 'hero' in label or 'woman' in label or 'red' in label or 'fire' in label
        boxes.append((cx, is_hero))
    if not boxes:
        return SRC_W / 2
    heroes = [cx for cx, h in boxes if h]
    foes = [cx for cx, h in boxes if not h]
    if heroes and foes:
        return HERO_W * (sum(heroes) / len(heroes)) + (1 - HERO_W) * (sum(foes) / len(foes))
    pool = heroes or foes
    return sum(pool) / len(pool)

def main():
    tmp = tempfile.mkdtemp(prefix='autotrack_')
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', SRC,
                    '-vf', f'fps={FPS}', os.path.join(tmp, 'f_%04d.jpg')], check=True)
    frames = sorted(f for f in os.listdir(tmp) if f.endswith('.jpg'))
    print(f'{len(frames)} frames @ {FPS}fps')

    raw = []
    t0 = time.time()
    for i, f in enumerate(frames):
        t = i / FPS
        dets = detect(os.path.join(tmp, f))
        tx = target_x(dets)
        raw.append((t, tx))
        labels = ','.join(str(d.get('label', '?'))[:12] for d in dets) or 'none'
        print(f'  t={t:5.1f}s target_x={tx:6.1f} dets=[{labels}]', flush=True)
    print(f'detection wall-clock: {time.time()-t0:.0f}s')

    # Smooth: rolling median (w=3) then slew-rate clamp (max px/s pan)
    xs = [x for _, x in raw]
    med = [sorted(xs[max(0, i-1):i+2])[len(xs[max(0, i-1):i+2]) // 2] for i in range(len(xs))]
    MAX_SLEW = 120.0 / FPS  # px per frame step
    sm = [med[0]]
    for x in med[1:]:
        sm.append(sm[-1] + max(-MAX_SLEW, min(MAX_SLEW, x - sm[-1])))
    kf = [(t, max(0, min(XMAX, x - CROP_W / 2))) for (t, _), x in zip(raw, sm)]

    json.dump({'fps': FPS, 'crop_w': CROP_W,
               'raw': raw, 'keyframes': kf},
              open(OUT.rsplit('.', 1)[0] + '_track.json', 'w'), indent=1)

    # Piecewise-linear crop-x expression (same approach as build_reframe.py)
    expr = str(kf[-1][1])
    for i in range(len(kf) - 2, -1, -1):
        t0_, x0 = kf[i]; t1, x1 = kf[i + 1]
        slope = (x1 - x0) / (t1 - t0_) if t1 > t0_ else 0
        expr = f'if(lt(t,{t1}),({x0:.1f}+({slope:.4f})*(t-{t0_})),{expr})'
    vf = f"crop={CROP_W}:{SRC_H}:'{expr}':0,scale={OUT_W}:{OUT_H}:flags=lanczos"
    r = subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', SRC, '-vf', vf,
                        '-c:v', 'libx264', '-crf', '18', '-preset', 'medium',
                        '-c:a', 'copy', OUT])
    print(f'render rc={r.returncode} -> {OUT}')

if __name__ == '__main__':
    main()

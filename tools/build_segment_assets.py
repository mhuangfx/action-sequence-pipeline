#!/usr/bin/env python3
"""Build a segment's reference clip, merged shot list and raw storyboard, from the full battle map.

One segment is a run of CLEAN two-fighter shots. Short excluded shots INSIDE the window (a cutaway
with nobody in it, an off-cast face) are spliced out and the action continues across them -- an
ordinary jump cut. So the reference clip is assembled from the clean shots, never straight-cut from
the source, or the excluded content rides along and can put a third character on screen.

Then:
  * the clip is sped 1.5x -- free, and it triples the motion dynamic range H3 sees
  * shots shorter than 0.5s OF OUTPUT are merged into their neighbour, because H3's text encoder
    samples the reference at 2 fps and cannot be told about anything briefer
  * one midpoint frame per merged shot is tiled into a board at an exactly renderable ratio
    (2x2 and 3x3 are 16:9, 3x4 is 4:3; anything else gets re-framed and the composition is lost)
"""
import argparse, json, os, subprocess, sys

SRC = 'output/fullbattle_20260808/src_full.mp4'
OUT = 'output/fullbattle_20260808'
SPEED = 1.5
FLOOR = 0.50          # seconds of OUTPUT; the 2 fps text-encoder sampling floor


def run(argv):
    subprocess.run(argv, check=True, capture_output=True)


def build_clip(name, shots):
    parts, listfile = [], f'{OUT}/_{name}_parts.txt'
    os.makedirs(f'{OUT}/_parts', exist_ok=True)
    with open(listfile, 'w') as fh:
        for i, s in enumerate(shots):
            p = f'{OUT}/_parts/{name}_{i:02d}.mp4'
            run(['ffmpeg', '-y', '-v', 'error', '-ss', '%.3f' % s['start'], '-to', '%.3f' % s['end'],
                 '-i', SRC, '-an', '-c:v', 'libx264', '-crf', '16', '-preset', 'veryfast', p])
            parts.append(p)
            fh.write("file '%s'\n" % os.path.abspath(p))
    one_x = f'{OUT}/{name}_1x.mp4'
    run(['ffmpeg', '-y', '-v', 'error', '-f', 'concat', '-safe', '0', '-i', listfile,
         '-c:v', 'libx264', '-crf', '16', '-preset', 'veryfast', '-an', one_x])
    fast = f'{OUT}/{name}_15x.mp4'
    run(['ffmpeg', '-y', '-v', 'error', '-i', one_x, '-filter:v', f'setpts=PTS/{SPEED}',
         '-r', '24', '-an', '-c:v', 'libx264', '-crf', '16', '-preset', 'veryfast', fast])
    for p in parts:
        os.remove(p)
    os.remove(listfile)
    return one_x, fast


def merged_list(shots):
    """Shot boundaries on the 1.5x timeline, with nothing left under the floor."""
    out, t = [], 0.0
    for s in shots:
        d = (s['end'] - s['start']) / SPEED
        if out and (out[-1]['end'] - out[-1]['start']) < FLOOR:
            out[-1]['end'] += d
            out[-1]['_from'].append(s['index'])
        else:
            out.append({'start': t, 'end': t + d, '_from': [s['index']]})
        t += d
    while len(out) > 1 and (out[-1]['end'] - out[-1]['start']) < FLOOR:
        out[-2]['end'] = out[-1]['end']; out[-2]['_from'] += out[-1]['_from']; out.pop()
    for i, s in enumerate(out, 1):
        s['index'] = i; s['dur'] = round(s['end'] - s['start'], 3)
        s['start'] = round(s['start'], 3); s['end'] = round(s['end'], 3)
    return out


def subtitle_score(path, band=0.20):
    """How much burned-in caption text is sitting in the bottom band of a frame.

    The source is subtitled throughout, and one sampled frame carried a line NAMING the source
    character. A board is a reference input to every render, so any caption in it propagates into
    the finished video. Captions are near-white, high-contrast and clustered in the lower fifth, so
    counting bright pixels there separates them from ordinary bright scenery well enough to RANK
    candidate frames -- which is all this needs to do.
    """
    from PIL import Image
    im = Image.open(path).convert('L')
    w, h = im.size
    strip = im.crop((int(w * 0.08), int(h * (1 - band)), int(w * 0.92), h))
    hist = strip.histogram()
    bright = sum(hist[225:])
    return bright / float(strip.size[0] * strip.size[1])


def pick_clean_frame(clip, start, dur, dest, samples=7):
    """Sample across the shot and keep the frame with the least caption in it.

    Returns (chosen_offset_fraction, score, cropped) -- cropped is True when every candidate was
    captioned and the band had to be cut off instead.
    """
    import tempfile
    from PIL import Image, ImageStat
    best = None
    with tempfile.TemporaryDirectory() as td:
        for k in range(samples):
            f = 0.15 + 0.7 * k / max(1, samples - 1)
            cand = os.path.join(td, 'c%02d.jpg' % k)
            run(['ffmpeg', '-y', '-v', 'error', '-ss', '%.3f' % (start + dur * f),
                 '-i', clip, '-frames:v', '1', '-q:v', '2', cand])
            if not os.path.exists(cand):
                continue
            # Reject frames with no content in them: an impact white-flash, a fade-to-black, a
            # uniform smear. One shot's every sample was a white flash and the board got a blank
            # panel -- which the restyle would then have to invent something into.
            st = ImageStat.Stat(Image.open(cand).convert('L'))
            flat = st.stddev[0] < 14 or st.mean[0] > 235 or st.mean[0] < 12
            sc = subtitle_score(cand)
            rank = (1 if flat else 0, sc)
            if best is None or rank < best[0]:
                best = (rank, f, sc, cand, flat)
        if best is None:
            raise SystemExit('no frame extracted from %s' % clip)
        rank, f, sc, cand, flat = best
        if flat:
            print('    WARNING: every sample of this shot is a flat frame (flash or fade); '
                  'the panel will carry little information')
        im = Image.open(cand)
        # ALWAYS crop the caption band. Scoring alone gives false negatives: a short, dim, italic
        # line ("It's not...") scored under the threshold and survived into a board. This source is
        # subtitled throughout, so every panel is at risk, and one of the captions found here NAMED
        # the source character. A uniform ~12% bottom crop costs a mild zoom on every panel and is
        # consistent across the board; a missed caption costs the whole render. The sampling above
        # still earns its keep -- it avoids frames where the caption is tall enough to survive the
        # crop, and it also skips near-black and smeared frames.
        w, h = im.size
        im = im.crop((0, 0, w, int(h * 0.88))).resize((w, h), Image.LANCZOS)
        im.save(dest, quality=95)
    return f, sc, flat


def grid_for(n):
    if n <= 4:  return 2, 2, 1920
    if n <= 9:  return 3, 3, 1280
    if n <= 12: return 3, 4, 1280
    raise SystemExit(f"{n} panels needs a grid we have not validated as renderable")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', help='comma-separated segment names')
    args = ap.parse_args()
    todo = json.load(open(f'{OUT}/segments_todo.json'))
    allshots = json.load(open(f'{OUT}/shots_full/shots.json'))['shots']
    sys.path.insert(0, 'tools')
    from battle_map_report import classify
    classify(allshots)
    if args.only:
        want = set(args.only.split(','))
        todo = [t for t in todo if t['name'] in want]

    for t in todo:
        name = t['name']
        shots = [s for s in allshots
                 if s['_clean'] and s['start'] >= t['src_start'] - 0.01 and s['end'] <= t['src_end'] + 0.01]
        one_x, fast = build_clip(name, shots)
        merged = merged_list(shots)
        json.dump(merged, open(f'{OUT}/shots_{name}_merged_src.json', 'w'), indent=1)
        fd = f'{OUT}/frames_{name}_merged_raw'
        subprocess.run(['rm', '-rf', fd]); os.makedirs(fd, exist_ok=True)
        # A shot that is nothing but an impact white-flash has no frame worth a panel. Absorb it
        # into its neighbour -- the flash belongs to that hit anyway -- rather than handing the
        # restyle a blank cell to invent something into.
        for _ in range(2):
            flat_at = []
            for s in merged:
                _, _, fl = pick_clean_frame(fast, s['start'], s['dur'], f'{fd}/_probe.jpg')
                if fl:
                    flat_at.append(s['index'])
            if not flat_at:
                break
            keep = []
            for s in merged:
                if s['index'] in flat_at and keep:
                    keep[-1]['end'] = s['end']; keep[-1]['_from'] += s['_from']
                elif s['index'] in flat_at and not keep:
                    keep.append(s)          # a leading flash has nothing to merge back into
                else:
                    keep.append(s)
            for i, s in enumerate(keep, 1):
                s['index'] = i; s['dur'] = round(s['end'] - s['start'], 3)
            if len(keep) == len(merged):
                break
            print('    absorbed flash panel(s) %s -> %d panels'
                  % (flat_at, len(keep)))
            merged = keep
        if os.path.exists(f'{fd}/_probe.jpg'):
            os.remove(f'{fd}/_probe.jpg')

        flags = []
        for s in merged:
            f, sc, cropped = pick_clean_frame(fast, s['start'], s['dur'],
                                              f'{fd}/%02d.jpg' % s['index'])
            s['_frame_at'] = round(f, 3); s['_caption_score'] = round(sc, 4)
            s['_caption_cropped'] = True
            if sc > 0.012:
                flags.append('panel %d was the least-captioned of 7 samples and still scored %.3f'
                             % (s['index'], sc))
        json.dump(merged, open(f'{OUT}/shots_{name}_merged_src.json', 'w'), indent=1)
        c, r, pw = grid_for(len(merged))
        board = f'{OUT}/{name}_board_merged_{c}x{r}_RAW.png'
        run(['python3', 'tools/build_storyboard.py', fd, board,
             '--cols', str(c), '--rows', str(r), '--panel-width', str(pw)])
        print("%-7s %2d clean shots -> %d panels  grid %dx%d  out %.2fs  %s%s"
              % (name, len(shots), len(merged), c, r, merged[-1]['end'],
                 os.path.basename(board), ('  [' + '; '.join(flags) + ']') if flags else ''))


if __name__ == '__main__':
    main()

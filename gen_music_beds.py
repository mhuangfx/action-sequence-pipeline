#!/usr/bin/env python3
"""Generate 2-3 candidate music beds for the ep0 trailer via Atlas Cloud
minimax/music-3.0 (instrumental, $0.15/generation flat). Not wired into
atlas_cloud.py's CLI (audio subcommand doesn't exist yet) -- calls the
submit/poll/download primitives directly so the daily spend cap + ledger
still apply.
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import atlas_cloud as ac

OUT_DIR = "output/trailer_test/music"
os.makedirs(OUT_DIR, exist_ok=True)

BASE_BRIEF = (
    "Dark cinematic fantasy movie-trailer score, instrumental only, no vocals, "
    "no choir syllables or wordless vocalizing. This is a FULL-LENGTH track, "
    "approximately 95 to 100 seconds long from start to the final cutoff -- do "
    "not end early or fade out after only 30-40 seconds; the structure below "
    "spans the entire 95-100 second duration and must be paced to fill it. "
    "Structure: "
    "0-30s sparse, low, ominous -- a single sustained low drone or bowed low string, "
    "very slow, wide empty space, one deep resonant percussive pulse (like a giant "
    "war-drum or heartbeat) landing every few seconds, minimal harmonic movement; "
    "30-50s the pulse thickens and a second darker layer enters, tension rising, "
    "still restrained; 50-80s build into a brass and choir-pad swell (orchestral "
    "brass and low string ostinato, epic scale, escalating), tempo and intensity "
    "climbing steadily, more percussion layers stacking; 90-93s one massive hard "
    "orchestral hit / impact with a huge low sub boom, then everything holds and "
    "sustains under it (long low sustained chord, cymbal/gong wash decaying) through "
    "roughly 96 seconds for a creature's roar to play over; then a hard cutoff, no "
    "natural fade. Keep the midrange frequency band (roughly 300Hz-3kHz) uncluttered "
    "and low in the mix throughout -- a slow spoken voice-over narration sits on top "
    "and needs that space clear. Cinematic trailer scoring, Hans Zimmer / Two Steps "
    "From Hell epic trailer style, dry-to-wet reverb building with intensity, no pop "
    "or dance elements, no melody hooks, mood over melody."
)

CANDIDATES = [
    {
        "name": "bed_01_drone",
        "prompt": (
            BASE_BRIEF + " Emphasis: DRONE-LED. The dominant texture throughout is a "
            "deep sustained low drone (cello/contrabass/synth sub-drone) that never "
            "fully disappears, even under the brass swell and the final hit; percussion "
            "and brass are secondary layers riding on top of it."
        ),
    },
    {
        "name": "bed_02_percussion",
        "prompt": (
            BASE_BRIEF + " Emphasis: PERCUSSION-LED. The dominant driving element "
            "throughout is layered war-drum and taiko-style percussion -- it enters "
            "first, carries the tension curve, and gains more simultaneous drum layers "
            "and faster subdivisions as the piece builds; brass and drone are secondary, "
            "in service of the percussive drive."
        ),
    },
    {
        "name": "bed_03_choir",
        "prompt": (
            BASE_BRIEF + " Emphasis: CHOIR-LED. A low wordless choir pad (dark, "
            "monastic, non-lexical hums and 'ah' tones only, no discernible words or "
            "lyrics) enters early under the drone and slowly swells in volume and "
            "register across the piece, becoming the dominant voice by the 50-80s "
            "brass build and carrying through the final sustained hit."
        ),
    },
]


def main():
    key = ac.load_key()
    only = sys.argv[1:] if len(sys.argv) > 1 else None
    manifest_path = os.path.join(OUT_DIR, "manifest.json")
    manifest = json.load(open(manifest_path)) if os.path.exists(manifest_path) else []
    done = {m["name"] for m in manifest}
    for c in CANDIDATES:
        name, prompt = c["name"], c["prompt"]
        if only and name not in only:
            continue
        if name in done:
            print(f"=== {name} already done, skipping ===", file=sys.stderr)
            continue
        payload = {
            "model": "minimax/music-3.0",
            "prompt": prompt,
            "is_instrumental": True,
            "lyrics_optimizer": False,
            "format": "wav",
            "sample_rate": 44100,
        }
        last_err = None
        for attempt in range(1, 4):
            print(f"=== {name} attempt {attempt} ===", file=sys.stderr)
            try:
                pid, _ = ac.submit("model/generateAudio", payload, key)
                print(f"submitted {pid}", file=sys.stderr)
                outs, data = ac.poll(pid, key, timeout=600, interval=8)
                out_path = os.path.join(OUT_DIR, f"{name}.wav")
                ac.download(outs[0], out_path)
                with open(os.path.join(OUT_DIR, f"{name}.txt"), "w") as f:
                    f.write(f"model: minimax/music-3.0\nprompt:\n{prompt}\n")
                print(f"saved {out_path}", file=sys.stderr)
                manifest.append({"name": name, "file": out_path, "prompt": prompt,
                                  "remote": outs[0], "prediction_id": pid})
                json.dump(manifest, open(manifest_path, "w"), indent=2)
                last_err = None
                break
            except Exception as e:
                last_err = e
                print(f"FAILED {name} attempt {attempt}: {e}", file=sys.stderr)
                time.sleep(5)
        if last_err is not None:
            print(f"=== giving up on {name} after 3 attempts ===", file=sys.stderr)
    print(json.dumps(manifest, indent=2))


import time


if __name__ == "__main__":
    main()

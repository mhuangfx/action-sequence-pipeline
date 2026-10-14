#!/usr/bin/env python3
"""Build the side-by-side review page for the overnight style A/B.

Three arms in columns, seeds in rows, so the eye compares like with like. Scores sit under each
clip because the whole question -- does the medium come from the reference, and can it be recovered
without breaking identity -- is one the numbers can only half answer.
"""
import json, os, subprocess, sys
from pathlib import Path

OUT = Path("output/fullbattle_20260808")
DEST = Path("output/style_ab_review")
ARMS = [("current", "colour reference", "The ordinary 1.5x reference. The look we have today."),
        ("gray", "desaturated reference", "Same reference, colour removed, NOT blurred. Tests whether flat colour is what carries cel shading."),
        ("noref", "no video reference", "No reference video and no &lt;Video 1&gt; tag. Removes the style source entirely -- and the motion with it.")]


def scores(name):
    d = {}
    p = OUT / f"{name}_copy.json"
    if p.exists():
        c = json.loads(p.read_text())
        v = next(iter(c.values()))
        d["copy"] = "%.0f%%" % (v["copy_fraction"] * 100)
        d["longest"] = "%.2fs" % v["longest_copy_seconds"]
    p = OUT / f"{name}_identity.json"
    if p.exists():
        f = json.loads(p.read_text())
        drawn = sum(1 for s in f if s.get("medium") == "drawing")
        youth = sum(1 for s in f if s.get("red_haired_youth"))
        d["drawn"] = "%d/%d" % (drawn, len(f))
        d["youth"] = str(youth)
    p = OUT / f"{name}_motion.json"
    if p.exists():
        m = json.loads(p.read_text())
        d["motion"] = m.get("motion_mean")
        d["range"] = m.get("motion_range")
    return d


def motion_line(s):
    """Motion energy, next to the reference's own numbers. Advisory: the eye still decides."""
    if s.get("motion") is None:
        return ""
    ref = scores("segA_15x")
    tail = ""
    if ref.get("motion") is not None:
        tail = f" &nbsp; <span style='opacity:.65'>reference {ref['motion']} / {ref['range']}x</span>"
    return (f"<br>motion <b>{s['motion']}</b> &nbsp; shot-to-shot range "
            f"<b>{s['range']}x</b>{tail}")


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    seeds = sorted({p.stem.split("_s")[-1] for p in OUT.glob("ovn_*_s*.mp4")})
    for extra in ("segA_15x.mp4", "segA_15x_grayref.mp4"):
        if (OUT / extra).exists():
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(OUT / extra),
                            "-r", "24", "-c:v", "libx264", "-crf", "18", "-an",
                            str(DEST / extra)], check=True)
    subprocess.run(["cp", "-f", str(OUT / "segA_board_merged_3x3_PHOTOREAL.png"), str(DEST)], check=False)

    h = ["<!-- built by tools/build_style_ab_review.py -->",
         "<style>",
         "body{background:#14161a;color:#e6e8ec;font:14px/1.5 system-ui,sans-serif;margin:0;padding:24px}",
         "h1{font-size:20px;margin:0 0 4px}h2{font-size:15px;margin:28px 0 10px;color:#9aa4b2}",
         ".lead{color:#9aa4b2;max-width:70ch}",
         "table{border-collapse:collapse;width:100%;margin-top:14px}",
         "td,th{vertical-align:top;padding:8px;border:1px solid #262a31}",
         "th{background:#1b1f25;text-align:left;font-weight:600}",
         "video{width:100%;border-radius:5px;display:block;background:#000}",
         ".s{font-size:12px;color:#9aa4b2;margin-top:6px}",
         ".s b{color:#e6e8ec}.bad{color:#ff8b6b}.good{color:#7fd18e}",
         "button{background:#22252c;color:#e6e8ec;border:1px solid #333;border-radius:6px;",
         "padding:6px 13px;font:inherit;cursor:pointer;margin-right:6px}",
         "img{max-width:100%;border-radius:5px}",
         "</style>",
         "<h1>Photoreal medium — three arms, paired seeds</h1>",
         "<p class='lead'>Everything is identical across a row except the reference video: same "
         "photoreal board, same prompt body, same seed, same length. <b>The question:</b> three "
         "renders on 2026-08-18 showed the rendering medium tracks the reference and ignores the "
         "prompt — stating the style per shot made it <i>worse</i>. Softening the reference fixed "
         "the medium but broke character identity. This asks whether removing COLOUR (rather than "
         "blurring) recovers the medium while leaving identity intact, and what dropping the "
         "reference entirely costs.</p>",
         "<p class='lead'><b>What to look for, in order:</b> does it read as photography or as "
         "drawing; is the white-shirted young man absent; does the motion still have weight and "
         "readable choreography. The scores answer the first two. Only your eye answers the third, "
         "and it is the reason the reference exists at all. The motion numbers under each clip are "
         "advisory only: mean energy and the loudest-to-quietest ratio across shots. They catch a "
         "clip that went flat; they cannot tell you whether a punch lands.</p>",
         "<p><button id='playall'>play all</button><button id='pauseall'>pause all</button>"
         "<button id='restart'>restart all</button></p>"]

    h.append("<h2>References</h2><table><tr>")
    for f, cap in (("segA_15x.mp4", "the ordinary colour reference"),
                   ("segA_15x_grayref.mp4", "the desaturated reference")):
        if (DEST / f).exists():
            h.append(f"<td><video class=c src='{f}' loop muted controls playsinline></video>"
                     f"<div class=s>{cap}</div></td>")
    h.append("<td><img src='segA_board_merged_3x3_PHOTOREAL.png'>"
             "<div class=s>the photoreal board — every arm gets this</div></td></tr></table>")

    h.append("<h2>Arms</h2><table><tr><th style='width:60px'>seed</th>")
    for key, title, why in ARMS:
        h.append(f"<th>{title}<div class=s style='font-weight:400'>{why}</div></th>")
    h.append("</tr>")
    for sd in seeds:
        h.append(f"<tr><th>{sd}</th>")
        for key, _, _ in ARMS:
            name = f"ovn_{key}_s{sd}"
            src = OUT / f"{name}.mp4"
            if not src.exists():
                h.append("<td class=s>not rendered</td>"); continue
            subprocess.run(["cp", "-f", str(src), str(DEST)], check=False)
            s = scores(name)
            drawn = s.get("drawn", "?")
            bad = drawn != "?" and drawn.split("/")[0] != "0"
            youth_cls = "bad" if s.get("youth", "0") != "0" else "good"
            h.append(f"<td><video class=c src='{name}.mp4' loop muted controls playsinline></video>"
                     f"<div class=s>drawn shots <b class='{'bad' if bad else 'good'}'>{drawn}</b>"
                     f" &nbsp; replaced character <b class='{youth_cls}'>{s.get('youth','?')}</b>"
                     f" &nbsp; copy <b>{s.get('copy','?')}</b> (longest {s.get('longest','?')})"
                     f"{motion_line(s)}</div></td>")
        h.append("</tr>")
    h.append("</table>")
    h.append("""<script>
const v=()=>[...document.querySelectorAll('video.c')];
playall.onclick=()=>v().forEach(x=>x.play());
pauseall.onclick=()=>v().forEach(x=>x.pause());
restart.onclick=()=>v().forEach(x=>{x.currentTime=0;x.play()});
addEventListener('keydown',e=>{if(e.code==='Space'){e.preventDefault();
 const p=v().some(x=>!x.paused); v().forEach(x=>p?x.pause():x.play());}});
</script>""")
    (DEST / "index.html").write_text("\n".join(h))
    print("wrote %s  (%d seeds x %d arms)" % (DEST / "index.html", len(seeds), len(ARMS)))


if __name__ == "__main__":
    main()

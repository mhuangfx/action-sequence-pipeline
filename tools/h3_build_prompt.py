#!/usr/bin/env python3
"""Merge per-shot descriptions with measured direction/camera clauses.

The two halves of an H3 prompt come from different places: the shot DESCRIPTIONS
(what is in frame) are written against the storyboard, and the direction CLAUSES
(camera move, drive direction, impact reversal, hit-stop) are measured off the
reference clip by shot_breakdown --subject. This joins them per shot.

Both halves must be numbered off ONE shot list -- see shot_breakdown --shots-json.
Detection does not survive a speed change, so a 1x list and a 1.5x list of the
same clip disagree, and zipping them silently misassigns every shot after the
first.

Clause order follows the configuration that scored 100% direction agreement:
    镜头N（start–end）DESCRIPTION；CLAUSES。
description first, clauses appended after a semicolon.
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

SHOT_RE = re.compile(r"镜头(\d+)（([^）]*)）")


def split_shots(text):
    """Split a prompt tail into (leading text, {shot number: description})."""
    hits = list(SHOT_RE.finditer(text))
    if not hits:
        return text, {}
    head = text[: hits[0].start()]
    shots = {}
    for i, m in enumerate(hits):
        end = hits[i + 1].start() if i + 1 < len(hits) else len(text)
        shots[int(m.group(1))] = text[m.end(): end].strip()
    return head, shots


def clause_lines(shots_json, duration):
    out = subprocess.run(
        [sys.executable, str(Path(__file__).with_name("shot_prompt_lines.py")),
         str(shots_json), "--duration", str(duration)],
        capture_output=True, text=True, check=True).stdout
    lines = {}
    for line in out.splitlines():
        m = SHOT_RE.match(line.strip())
        if not m:
            continue
        body = line.strip()[m.end():].replace("{DESC}", "").strip()
        lines[int(m.group(1))] = (m.group(2), body)
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", required=True, help="prompt parts JSON carrying the descriptions")
    ap.add_argument("--shots", required=True, help="shots.json measured on the SAME shot list")
    ap.add_argument("--duration", type=float, required=True, help="H3 output length in seconds")
    ap.add_argument("--out", required=True, help="destination parts JSON")
    args = ap.parse_args()

    parts = json.loads(Path(args.parts).read_text())
    tail = parts[-1]
    if "t" not in tail:
        sys.exit("last part is a mention chip, expected text carrying the shot list")
    head, descs = split_shots(tail["t"])
    if not descs:
        sys.exit("no 镜头N（…） shot descriptions found in --parts")

    clauses = clause_lines(args.shots, args.duration)
    missing = sorted(set(descs) - set(clauses))
    if missing:
        sys.exit("shots %s have a description but no measured clauses -- the two "
                 "sources are not on the same shot list" % missing)

    body = []
    for n in sorted(descs):
        desc = descs[n].rstrip("。 ").strip()
        span, clause = clauses[n]
        body.append("镜头%d（%s）%s；%s" % (n, span, desc, clause))

    extra = sorted(set(clauses) - set(descs))
    out = parts[:-1] + [{"t": head.rstrip() + " " + " ".join(body)}]
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    print("merged %d shots -> %s" % (len(body), args.out))
    if extra:
        print("note: shots %s had clauses but no description, omitted "
              "(matches the hosted prompt, which skipped the sub-frame sliver)" % extra)


if __name__ == "__main__":
    main()

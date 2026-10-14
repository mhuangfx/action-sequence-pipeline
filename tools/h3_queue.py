#!/usr/bin/env python3
"""Run a queue of H3 generations across multiple sessions, resumably.

A battle is 24-30 segments at roughly half an hour each, so a full render spans
several nights and gets interrupted every morning. This keeps the work list and
its progress in files, so re-running the same queue skips what already landed
and picks up at the first unfinished job.

Interruption is expected, not exceptional: SIGINT/SIGTERM let the in-flight
generation finish (killing it wastes up to 36 minutes of sampling that ComfyUI
has already done) and then stop cleanly before the next one.

Queue file: a JSON list of jobs, each a dict of h3_local.py options minus the
leading dashes, e.g.
    [{"name": "seg01", "prompt-parts": "prompts/seg01.json",
      "ref-image": ["board.png", "vesta.png", "opp.png"],
      "ref-video": "seg01_15x.mp4", "seconds": 5.875}]
Keys may repeat as lists. A "skip": true job is ignored.
"""
import argparse
import json
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

DRIVER = Path(__file__).with_name("h3_local.py")
stopping = False


def request_stop(signum, frame):
    global stopping
    if stopping:
        print("\nsecond signal -- aborting immediately", flush=True)
        sys.exit(130)
    stopping = True
    print("\nstop requested: finishing the current generation, then exiting. "
          "Signal again to abort now.", flush=True)


def load_state(path):
    if path.exists():
        return json.loads(path.read_text())
    return {"done": {}, "failed": {}}


def save_state(path, state):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2) + "\n")
    tmp.replace(path)


def job_argv(job):
    argv = [sys.executable, "-u", str(DRIVER)]
    for key, value in job.items():
        if key == "skip":
            continue
        for v in (value if isinstance(value, list) else [value]):
            if isinstance(v, bool):
                if v:
                    argv.append("--" + key)
            else:
                argv += ["--" + key, str(v)]
    return argv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("queue", help="JSON list of jobs")
    ap.add_argument("--state", help="progress file (default: <queue>.state.json)")
    ap.add_argument("--retry-failed", action="store_true",
                    help="also re-run jobs that failed on an earlier pass")
    ap.add_argument("--log-dir", default="output/h3_queue_logs")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    queue_path = Path(args.queue)
    jobs = json.loads(queue_path.read_text())
    state_path = Path(args.state) if args.state else queue_path.with_suffix(".state.json")
    state = load_state(state_path)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    pending = []
    for job in jobs:
        name = job.get("name")
        if not name:
            sys.exit("every job needs a \"name\"")
        if job.get("skip"):
            continue
        if name in state["done"]:
            continue
        if name in state["failed"] and not args.retry_failed:
            continue
        pending.append(job)

    print("%d job(s) in queue, %d done, %d pending"
          % (len(jobs), len(state["done"]), len(pending)))
    if args.dry_run:
        for job in pending:
            print("  " + " ".join(job_argv(job)[2:]))
        return

    for i, job in enumerate(pending, 1):
        if stopping:
            break
        name = job["name"]
        started = time.time()
        print("\n[%d/%d] %s  (%s)"
              % (i, len(pending), name, datetime.now().strftime("%H:%M:%S")), flush=True)
        log_path = log_dir / (name + ".log")
        with open(log_path, "w") as log:
            proc = subprocess.run(job_argv(job), stdout=log,
                                  stderr=subprocess.STDOUT, text=True)
        elapsed = time.time() - started
        record = {"seconds": round(elapsed, 1),
                  "finished": datetime.now().isoformat(timespec="seconds"),
                  "log": str(log_path)}
        if proc.returncode == 0:
            state["done"][name] = record
            print("  ok in %dm%02ds" % (elapsed // 60, elapsed % 60), flush=True)
        else:
            record["returncode"] = proc.returncode
            state["failed"][name] = record
            print("  FAILED (exit %d) -- see %s" % (proc.returncode, log_path), flush=True)
        save_state(state_path, state)

    remaining = sum(1 for j in jobs
                    if not j.get("skip") and j["name"] not in state["done"])
    print("\n%d done, %d failed, %d still to do"
          % (len(state["done"]), len(state["failed"]), remaining))
    if remaining:
        print("resume with: %s %s" % (Path(sys.argv[0]).name, args.queue))


if __name__ == "__main__":
    main()

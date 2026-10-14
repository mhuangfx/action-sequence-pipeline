# Overnight local-model batch

One night's batch on the local video model, as an example.

- `build.py` cuts 3x3 storyboards into part-boards, writes one prompt per job, and writes the queue.
- `queue.example.json` is the queue it wrote: per job a prompt file, reference images, length, and seed.
  The boards and character sheets it points to are not in this repo.
- `run_overnight.sh` runs the queue with `tools/h3_queue.py`. It is resumable: re-run to continue.
  One Ctrl-C finishes the current clip, then stops.
- `prompts/ref_onetake.example.txt` is one prompt from the batch: a single continuous take driven by a
  character reference only.

Every job then goes through `tools/stutter_check.py` before review.

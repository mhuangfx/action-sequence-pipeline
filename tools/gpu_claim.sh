#!/usr/bin/env bash
# Take the card properly, then run a command on it.
#
# The lock alone is NOT enough. Ollama keeps a vision model resident in VRAM for its keep_alive
# window (5 minutes by default) AFTER the HTTP call returns and after the lock is released. On
# 2026-08-18 an H3 render started immediately behind a dialogue sweep, found ~9 GB still held by
# qwen3-vl, and died with CUDA out-of-memory 70 seconds in -- having queued correctly, behind a
# lock that was doing exactly what it was told.
#
# ★ 2026-08-19: the SAME failure exists in the other direction, and it appeared on the very first
# render-then-grade cycle. Before any render, ComfyUI idles holding ~6.5GB, so the vision gate had
# always started on a card with ~18GB free. AFTER a render it holds ~13.8GB, leaving 10.8GB -- less
# than the 13.4GB the vision model wants. The gate then sat in the wait loop below for the full 200
# seconds and started anyway on a card too full for it. Nothing had ever rendered and then graded
# in this repo, so this had never once been exercised.
#
# So: take the lock, evict every resident Ollama model, ALSO tell ComfyUI to drop its cache when the
# command we are about to run is not itself a ComfyUI render, wait for the card to actually drain,
# then run. Use this for EVERY H3 render and EVERY vision gate.
#
# Usage: tools/gpu_claim.sh <command...>
set -u
LOCK=output/state/gpu.lock
mkdir -p output/state

exec 9>"$LOCK"
flock 9

for m in $(curl -s http://127.0.0.1:11434/api/ps 2>/dev/null \
           | python3 -c "import json,sys; print(' '.join(x['name'] for x in json.load(sys.stdin).get('models',[])))" 2>/dev/null); do
  echo "gpu_claim: evicting $m from VRAM"
  curl -s -X POST http://127.0.0.1:11434/api/generate -d "{\"model\":\"$m\",\"keep_alive\":0}" -o /dev/null
done

# Drop ComfyUI's cached models, but ONLY when we are not about to run a render -- unloading right
# before a render would just pay the reload cost for nothing. The vision gate needs the room; the
# renderer is the one that put the models there in the first place.
case " $* " in
  *h3_local.py*) : ;;
  *) echo "gpu_claim: asking ComfyUI to release its cached models"
     curl -s -m 30 -X POST http://127.0.0.1:8199/free \
          -H 'Content-Type: application/json' \
          -d '{"unload_models": true, "free_memory": true}' -o /dev/null \
       || echo "gpu_claim: ComfyUI /free did not answer -- continuing"
     ;;
esac

# 18000 was too strict: ComfyUI legitimately caches its models between renders and holds ~6.5GB,
# leaving 17973MB free -- which blocked a render that would have run fine. What actually caused the
# out-of-memory this script exists for was Ollama holding ~9GB, and the eviction above is the fix.
# This threshold is only a backstop against starting into a genuinely full card.
# ★ 2026-08-29 (EP0 v5): the 15000MB floor can NEVER be met by a back-to-back H3 render.
# ComfyUI idles holding ~14.7GB of its own cached H3 models between renders -- the models the next
# render is about to reuse -- so nvidia-smi reports ~9.8GB free and every single render sat through
# the full 40 x 5 s wait, printed "GAVE UP WAITING", and then started anyway and worked. That is
# 200 s of dead time per render, ~1 GPU hour across an 18-shot pass, buying nothing. An H3 render
# is the process that put those models there, so it gets a lower floor; everything else (the vision
# gates, which need the room and for which ComfyUI is asked to free above) keeps 15000.
case " $* " in
  *h3_local.py*) need_mb=${GPU_CLAIM_FREE_MB:-8000} ;;
  *)             need_mb=${GPU_CLAIM_FREE_MB:-15000} ;;
esac
for i in $(seq 1 40); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
  free=$(( total - used ))
  [ "$free" -ge "$need_mb" ] && break
  echo "gpu_claim: only ${free}MB free, need ${need_mb}MB -- waiting ($i/40)"
  sleep 5
done
# ★ Say so LOUDLY when the wait was given up on rather than satisfied. The old script printed the
# same cheerful "starting" line either way, so a card that never drained looked exactly like a card
# that did -- this project's signature bug, in the one script every GPU job goes through.
if [ "$free" -lt "$need_mb" ]; then
  echo "gpu_claim: ★ GAVE UP WAITING -- only ${free}MB free, wanted ${need_mb}MB. Starting anyway."
else
  echo "gpu_claim: ${free}MB free, starting"
fi
exec "$@"

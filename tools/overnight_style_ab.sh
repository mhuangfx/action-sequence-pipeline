#!/usr/bin/env bash
# Overnight A/B: does the RENDERING MEDIUM come from the reference video, and can it be recovered
# without breaking identity?
#
# Three arms, paired seeds, everything else identical (same board, same prompt body, same length):
#   current  -- the ordinary 1.5x colour reference          (seed 770809 already rendered)
#   gray     -- the same reference, DESATURATED, not blurred
#   noref    -- no video reference at all, and no <Video 1> tag in the prompt
#
# Paired seeds because the 2026-08-09 batch established that single-run deltas here are dice.
set -u
cd .
OUT=output/fullbattle_20260808
SEEDS="770809 424242 131313"

render () {  # arm seed prompt extra_args...
  local arm=$1 seed=$2 prompt=$3; shift 3
  local name="ovn_${arm}_s${seed}"
  if [ -f "$OUT/$name.mp4" ]; then echo "== $name already done"; return; fi
  echo "== $name  $(date +%H:%M:%S)"
  tools/gpu_claim.sh .venv/bin/python3 -u tools/h3_local.py \
    --name "$name" --prompt-parts "$prompt" \
    --ref-image $OUT/segA_board_merged_3x3_PHOTOREAL.png \
    --ref-image characters/char01_vesta/ref_vestia_v3.png \
    --ref-image characters/<hero>/sheet.png \
    "$@" --seconds 5.8 --seed "$seed" --out-dir "$OUT" \
    > "$OUT/$name.log" 2>&1 \
    && echo "   ok" || echo "   FAILED -- $(tail -1 $OUT/$name.log)"
}

P_STD=prompts/fullbattle_photoreal/segA_video_photoreal_parts.json
P_NOV=prompts/fullbattle_photoreal/segA_video_photoreal_novideo_parts.json

# Interleaved by seed, so an early stop still leaves complete paired sets rather than one whole arm.
for s in $SEEDS; do
  render gray  "$s" "$P_STD" --ref-video $OUT/segA_15x_grayref.mp4
  render noref "$s" "$P_NOV"
  [ "$s" = "770809" ] || render current "$s" "$P_STD" --ref-video $OUT/segA_15x.mp4
done
cp -n $OUT/cal_segA_photoreal.mp4 $OUT/ovn_current_s770809.mp4 2>/dev/null || true

echo; echo "== scoring $(date +%H:%M:%S)"
for f in $OUT/ovn_*.mp4; do
  n=$(basename "$f" .mp4)
  .venv/bin/python3 tools/passthrough.py $OUT/segA_15x.mp4 "$f" \
      --json "$OUT/${n}_copy.json" > "$OUT/${n}_copy.txt" 2>&1
  tools/gpu_claim.sh python3 tools/identity_gate.py "$f" \
      --shots $OUT/shots_segA_merged_src.json --json "$OUT/${n}_identity.json" \
      > "$OUT/${n}_identity.txt" 2>&1 || true
  echo "   scored $n"
done
echo "== all done $(date +%H:%M:%S)"

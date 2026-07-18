#!/usr/bin/env zsh
# Stage B for one axis: run the validator with the chosen Stage-A winner template over a
# larger source-diverse scenario pool, then export strict-pass selections.
#
# Usage: run_honesty_credulity_stage_b.sh <axis_id> <winner_template> <n>
# e.g.: run_honesty_credulity_stage_b.sh credulous_skeptical "Speak with the priorities of someone {persona}." 40
set -euo pipefail

AXIS_ID=$1
TEMPLATE=$2
N=${3:-40}
LIB=/media/wassname/SGIronWolf/projects5/2026/weight-steering-repos/persona-steering-template-library
cd "$LIB"

OUT=out/${AXIS_ID}_stage_b_bounded.json
uv run python scripts/validate_persona_axes.py \
  --generator-model qwen/qwen3-14b \
  --generator-provider-only DeepInfra \
  --judge-model google/gemini-3.1-flash-lite-preview \
  --axis-judge-models qwen/qwen3-14b \
  --axis-judge-method bounded_thinking \
  --axis-judge-n 2 --axis-judge-budget 4096 \
  --axes data/personas/persona_pairs_${AXIS_ID}.jsonl \
  --templates "$TEMPLATE" \
  --family data/scenarios/scenarios_daily_dilemmas.jsonl,data/scenarios/scenarios_moral_stories.jsonl,data/scenarios/scenarios_social_chem.jsonl,data/scenarios/scenarios_ethics_qna.jsonl,data/scenarios/scenarios_airisk.jsonl,data/scenarios/scenarios_genies_sycophancy.jsonl \
  --n "$N" --seed 13 --concurrency 6 \
  --out "$OUT"

# export strict-pass selections (reuse the authority export script; it is axis-agnostic via --axis-filter)
UV run python scripts/export_authority_steering_selection.py \
  --stage-b "$OUT" \
  --axis-filter "$AXIS_ID" \
  --out-dir out/${AXIS_ID}_selection_bounded \
  --keep-per-source 10 \
  --strict-only

echo "Stage B done for $AXIS_ID; selection in out/${AXIS_ID}_selection_bounded/"

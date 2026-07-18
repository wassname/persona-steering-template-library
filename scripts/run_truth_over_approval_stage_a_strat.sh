#!/bin/bash
# Stage A for honesty axis (truth_over_approval): ALL 100 templates x 1 scenario/source (12 sources) = 1200 pairs.
# v3: baseline-anchored judging (neg < baseline < pos, min_side_delta gate).
# --exclude-confound-dims: on-axis dims that circularly penalize honesty steering.
# Usage: bash scripts/run_truth_over_approval_stage_a_strat.sh
set -euo pipefail
cd "$(dirname "$0")/.."

uv run python scripts/validate_persona_axes.py \
  --generator-model qwen/qwen3-14b \
  --generator-provider-only DeepInfra \
  --judge-model google/gemini-3.1-flash-lite-preview \
  --axis-judge-models qwen/qwen3-14b \
  --axis-judge-method bounded_thinking \
  --axis-judge-n 2 --axis-judge-budget 4096 \
  --axes data/personas/persona_pairs_truth_over_approval.jsonl \
  --templates data/templates/template_catalog.yaml \
  --exclude-confound-dims honesty_truthfulness,praise_flattery,sycophancy \
  --family data/scenarios/scenarios_daily_dilemmas.jsonl,data/scenarios/scenarios_moral_stories.jsonl,data/scenarios/scenarios_social_chem.jsonl,data/scenarios/scenarios_ethics_qna.jsonl,data/scenarios/scenarios_airisk.jsonl,data/scenarios/scenarios_genies_sycophancy.jsonl,data/scenarios/scenarios_machiavelli.jsonl,data/scenarios/scenarios_valuebench.jsonl,data/scenarios/scenarios_sycophancy_eval.jsonl,data/scenarios/scenarios_genies_agentic.jsonl,data/scenarios/scenarios_w2s_character_3p.jsonl,data/scenarios/scenarios_v2_candidates.jsonl \
  --n-per-source 1 --seed 13 --concurrency 8 \
  --out out/truth_over_approval_stage_a_strat_v3.json

---
name: persona-template-library
description: "Use this repo to choose, validate, and export persona templates and persona pairs for steering experiments."
---

# Persona Template Library

Use this skill when working inside this repo to choose persona templates, write
mirrored persona pairs, validate scenario suffixes on OpenRouter, or export the
dataset.

## Canonical Files

- `README.qmd`: single source for README.md and GitHub Pages.
- `README.md`: quick-start workflow, headline results, and plot for readers.
- `docs/choosing_personas.md`: workflow for writing mirrored persona pairs.
- `docs/persona_prompt_literature_review.md`: literature review for persona prompt
  shapes used by steering repos and papers.
- `data/templates/template_catalog.yaml`: reusable template inventory.
- `data/personas/persona_pairs_pilot_two.jsonl`: measured pilot persona pairs.
- `data/personas/persona_pairs_v2_candidates.jsonl`: candidate persona pairs.
- `data/scenarios/scenarios_*.jsonl`: candidate scenario suffixes to validate on the
  target model.
- `data/selections/`: committed selected scenario sets that have passed a target-model
  validation screen.
- `scripts/scenario_sources/export_scenarios.py`: writes source loader outputs into
  `data/scenarios/scenarios_<source>.jsonl`.
- `data/results/`: committed result tables and reader-facing result assets.
- `out/`: local scratch outputs and API caches; ignored by git.
- `scripts/validate_persona_axes_openrouter.py`: live and dry-run validator.
- `scripts/prepare_authority_steering_selection.py`: source-stratified Stage A/Stage B
  authority-axis selection inputs.
- `scripts/export_authority_steering_selection.py`: exports the chosen axis/template
  and strict-pass scenarios from validator artifacts.
- `scripts/export_persona_template_stats.py`: converts validator artifacts into
  examples and score tables.
- `scripts/summarize_model_matrix.py`: summarizes latest model-matrix logs for
  the README/Pages render.
- `scripts/build_hf_dataset.py`: builds the Hugging Face splits, including
  `main`, `template_pair_cells`, `persona_pairs`, `examples`, and `controls`.

## Workflow

Use the repo in this order:

1. Choose persona templates from the `README.md` Results Snapshot table, the
   Hugging Face `main` split, or `data/templates/template_catalog.yaml`.
2. Choose persona pairs with `docs/choosing_personas.md`. Mirror-test each pair:
   every positive clause needs a negative counterpart that only flips the
   intended pole.
3. Choose scenario suffixes by validating them on the target model with
   `scripts/validate_persona_axes_openrouter.py`. Keep suffixes that elicit the
   behavior mode you need: doing, judging, explaining, refusing, moral tradeoffs,
   or multi-turn behavior.
4. Run a dry-run validator command before live OpenRouter calls.
5. For a steering-ready selection, use a two-stage screen:
   Stage A = several axes/templates on a small source-diverse panel; Stage B =
   the chosen axis/template on up to 30 scenarios per source.
6. Export strict-pass scenarios only. Do not steer from source-balanced filler rows
   when a source produced no clean scenarios.
7. Commit reusable selected scenario JSONL files under `data/selections/`.
8. After a live run, inspect examples before trusting scores.

Read `docs/persona_prompt_literature_review.md` when choosing new persona pairs or
template shapes from related work. If the global `persona-steering` skill is
available, read it for longer curation rules and worked examples.

For report edits, edit `README.qmd` and render both outputs:

```sh
just readme
just pages
```

The steering arithmetic matters: a direction is the average positive-minus-
negative difference. Any systematic length, refusal, formality, confidence,
language, or persona-label difference can become the axis.

## Commands

Catalog check:

```sh
uv run python scripts/sync_template_library.py --check
```

Export scenarios from source loaders:

```sh
uv run python scripts/scenario_sources/export_scenarios.py --sources all --limit 1999
```

Dry-run validation:

```sh
uv run python scripts/validate_persona_axes_openrouter.py \
  --axes data/personas/persona_pairs_pilot_two.jsonl \
  --templates data/templates/template_catalog.yaml \
  --family data/scenarios/scenarios_v2_candidates.jsonl \
  --n 1 \
  --seed 24 \
  --dry-run \
  --out out/persona_template_library_dryrun.json
```

Live validation:

```sh
OPENROUTER_API_KEY=... uv run python scripts/validate_persona_axes_openrouter.py \
  --axes data/personas/persona_pairs_pilot_two.jsonl \
  --templates data/templates/template_catalog.yaml \
  --family data/scenarios/scenarios_v2_candidates.jsonl \
  --n 2 \
  --seed 24 \
  --out out/persona_template_library_v2_pilot_seed24.json
```

Export stats:

```sh
uv run python scripts/export_persona_template_stats.py \
  out/persona_template_library_v2_pilot_seed24.json \
  --out-prefix data/results/stats/v2_pilot_seed24
```

Authority/dignity steering selection:

```sh
uv run python scripts/prepare_authority_steering_selection.py \
  --out-dir out/authority_selection

uv run python scripts/validate_persona_axes_openrouter.py \
  --axes out/authority_selection/stage_a_axes.jsonl \
  --templates out/authority_selection/stage_a_templates.txt \
  --family out/authority_selection/stage_a_scenarios.jsonl \
  --n 24 --seed 42 \
  --generator-model qwen/qwen3-8b \
  --axis-judge-models google/gemini-3.1-flash-lite-preview \
  --judge-model google/gemini-3.1-flash-lite-preview \
  --out out/authority_selection/stage_a_live.json

uv run python scripts/export_authority_steering_selection.py \
  --stage-a out/authority_selection/stage_a_live.json \
  --axis-filter dignity_over_authority \
  --out-dir out/authority_selection/dignity_strict

uv run python scripts/validate_persona_axes_openrouter.py \
  --axes out/authority_selection/dignity_strict/stage_b_axis.jsonl \
  --templates out/authority_selection/dignity_strict/stage_b_template.txt \
  --family out/authority_selection/stage_b_candidate_scenarios.jsonl \
  --n 342 --seed 43 \
  --generator-model qwen/qwen3-8b \
  --axis-judge-models google/gemini-3.1-flash-lite-preview \
  --judge-model google/gemini-3.1-flash-lite-preview \
  --out out/authority_selection/dignity_strict/stage_b_live.json

uv run python scripts/export_authority_steering_selection.py \
  --stage-b out/authority_selection/dignity_strict/stage_b_live.json \
  --out-dir out/authority_selection/dignity_strict \
  --keep-per-source 10 \
  --strict-only
```

Refresh README tables:

```sh
just readme
just pages
```

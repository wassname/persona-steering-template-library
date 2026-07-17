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
- `scripts/run_*_stage_a_strat.sh`: canonical Stage A invocations (models, judges,
  scenario panel). Copy one of these for a new axis; do not re-derive flags.
- `scripts/parse_stage_a.py`: rank Stage A templates by strict_pass_rate, then
  min_side_delta, then axis_delta; prints per-side deltas.
- `scripts/export_selections.py`: export top-N/strict-pass rows from a Stage B
  artifact. Mirrors the validator's gates exactly; do not re-implement them.
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
   Hugging Face `main` split, or `data/templates/template_catalog.yaml`. For
   Stage A, test the full catalog when budget permits. If budget is constrained,
   start from the top Results Snapshot scorers plus any strong templates that
   plausibly fit the new axis. Use a deterministic top-N rule when you
   downsample; do not use a small hand-picked subset. Note the Results Snapshot
   predates the both-directions check; when a Stage A artifact with per-side
   deltas exists for your axis, rank with `scripts/parse_stage_a.py` instead.
2. Choose persona pairs with `docs/choosing_personas.md`. Mirror-test each pair:
   every positive clause needs a negative counterpart that only flips the
   intended pole.
3. Choose scenario suffixes by validating them on the target model with
   `scripts/validate_persona_axes_openrouter.py`. Keep suffixes that elicit the
   behavior mode you need: doing, judging, explaining, refusing, moral tradeoffs,
   or multi-turn behavior.
4. Run a dry-run validator command before live OpenRouter calls. Then run a live
   smoke test (1 scenario x 2 templates x 1 axis, ~cents) before any 1000+ pair
   run, and read the printed FIRST/LAST audit samples: pos/neg/base responses
   should differ in the intended way, and both side deltas should be populated.

   ```sh
   uv run python scripts/validate_persona_axes_openrouter.py \
     --generator-model qwen/qwen3-14b --generator-provider-only DeepInfra \
     --judge-model google/gemini-3.1-flash-lite-preview \
     --axis-judge-models qwen/qwen3-14b --axis-judge-method bounded_thinking \
     --axes data/personas/persona_pairs_<axis>.jsonl \
     --templates 'You are {persona}.||Act as if you are extremely {persona}.' \
     --family data/scenarios/scenarios_daily_dilemmas.jsonl \
     --n-per-source 1 --seed 13 --out out/smoke_<axis>.json
   ```
5. For a steering-ready selection, use a two-stage screen:
   Stage A = broad template evidence on a small source-diverse panel (ALL templates x FEW scenarios, e.g. 100 templates x 1/source); Stage B = the chosen winner/template on MANY axis-affordance-ranked scenarios (1 template x ~1000 scenarios). This cost structure is deliberate: 100x12 + 1x1000 << 100x1000, so test all templates cheaply first, then spend scenarios on the winner only. Run Stage A per-axis (one axis at a time), not as a cross-product of all axes x all templates -- persona axes are mostly orthogonal, so the combinatorial cost is unnecessary. For N axes you run N Stage A passes (one template winner each), then N Stage B passes, not N^2.
   For Stage A, copy an existing `scripts/run_*_stage_a_strat.sh` for the new axis
   and rank the artifact with `scripts/parse_stage_a.py`. `pos_d` and `neg_d` are
   each persona's movement vs a no-persona answer (-2..+2); rank by `min_side` and
   prefer templates where both move. A side near 0 is dead (that persona just
   reproduces the default). The pass bar `--min-side-threshold` (default 0.5) is
   per-axis; some axes top out below it when the default already sits near one pole.
   Before Stage B, write or adapt a prepare script like
   `scripts/prepare_authority_steering_selection.py` for the new axis. Random
   scenario sampling is insufficient for narrow axes because most scenarios will
   not afford the intended behavior. Use `--n-per-source N` (stratified) so each
   source contributes N scenarios equally, not proportional to source size; the
   legacy `--n` pools all sources and samples N total (large sources dominate).
6. Export with `scripts/export_selections.py` (`--strict-only`, or `--top-n` by
   overall_score when strict rows are scarce; pass the same thresholds as the run).
   Strict rows are rare (~0.2-2%), mostly killed by the judge's usable-for-training
   call; that is normal. If Stage B gets 0 strict-pass scenarios,
   test more ranked scenarios (`--n-per-source 50+`, more sources) and/or try
   stronger templates (the full catalog includes system-prompt, red-team, and
   jailbreak-style templates that break the model out of its default stance).
   Do not relax the strict gate, lower the export rule, or steer from
   source-balanced filler rows when a source produced no clean scenarios.
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
language, or persona-label difference can become the axis. And the pair is used
at both -C and +C, so both personas must move the model away from its default
answer, not just away from each other.

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
  --n-per-source 2 \
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
  --n-per-source 2 \
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

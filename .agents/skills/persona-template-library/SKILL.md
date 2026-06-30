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
- `docs/persona_prompt_prior_art.md`: annotated prior art for persona prompt
  shapes used by steering repos and papers.
- `data/templates/template_catalog.yaml`: reusable template inventory.
- `data/personas/persona_pairs_pilot_two.jsonl`: measured pilot persona pairs.
- `data/personas/persona_pairs_v2_candidates.jsonl`: candidate persona pairs.
- `data/scenarios/scenarios_*.jsonl`: candidate scenario suffixes to validate on the
  target model.
- `scenario_sources/export_scenarios.py`: writes source loader outputs into
  `data/scenarios/scenarios_<source>.jsonl`.
- `out/stats/`: local generated stats and examples; ignored by git, so do not
  assume these exist in a clean checkout.
- `scripts/validate_persona_axes_openrouter.py`: live and dry-run validator.
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
5. After a live run, export stats and inspect examples before trusting scores.

Read `docs/persona_prompt_prior_art.md` when choosing new persona pairs or
template shapes from prior work. If the global `persona-steering` skill is
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
uv run python scenario_sources/export_scenarios.py --sources all --limit 1999
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
  --out-prefix out/stats/v2_pilot_seed24
```

Refresh README tables:

```sh
just readme
just pages
```

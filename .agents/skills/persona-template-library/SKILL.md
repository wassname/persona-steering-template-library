---
name: persona-template-library
description: "Use this repo to choose, validate, and export persona templates and persona pairs for steering experiments."
---

# Persona Template Library

Use this skill when working inside this repo on persona-template selection,
persona-pair selection, OpenRouter validation runs, or dataset export.

## Canonical Files

- `docs/choosing_personas.md`: workflow for choosing personas and templates.
- `docs/persona_prompt_prior_art.md`: annotated prior art for persona prompt
  shapes used by steering repos and papers.
- `data/template_catalog.yaml`: reusable template inventory.
- `data/persona_pairs_pilot_two.jsonl`: measured pilot persona pairs.
- `data/persona_pairs_v2_candidates.jsonl`: candidate persona pairs.
- `out/stats/`: local generated stats and examples; ignored by git, so do not
  assume these exist in a clean checkout.
- `scripts/validate_persona_axes_openrouter.py`: live and dry-run validator.
- `scripts/export_persona_template_stats.py`: converts validator artifacts into
  examples and score tables.
- `scripts/build_hf_dataset.py`: builds the Hugging Face splits, including
  `main`, `template_pair_cells`, `persona_pairs`, `examples`, and `controls`.

## Workflow

1. Read `docs/choosing_personas.md`.
2. Read `docs/persona_prompt_prior_art.md` when choosing new persona pairs or
   template shapes from prior work.
3. If the global `persona-steering` skill is available, read it too; it has the
   longer literature notes, curation rules, and worked examples behind this
   repo's shorter guide.
4. Choose candidate persona pairs by mirror-testing them: each positive clause
   needs a negative counterpart that only flips the intended pole.
5. Choose candidate templates that bind the persona to behavior, judgment, or
   perspective rather than pure identity.
6. Run a dry-run validator command before live OpenRouter calls.
7. After a live run, export stats and inspect examples before trusting scores.

The steering arithmetic matters: a direction is the average positive-minus-
negative difference. Any systematic length, refusal, formality, confidence,
language, or persona-label difference can become the axis.

## Commands

Catalog check:

```sh
uv run python scripts/sync_template_library.py --check
```

Dry-run validation:

```sh
uv run python scripts/validate_persona_axes_openrouter.py \
  --axes data/persona_pairs_pilot_two.jsonl \
  --templates data/template_catalog.yaml \
  --family data/scenarios_v2_candidates.jsonl \
  --n 1 \
  --seed 24 \
  --dry-run \
  --out out/persona_template_library_dryrun.json
```

Live validation:

```sh
OPENROUTER_API_KEY=... uv run python scripts/validate_persona_axes_openrouter.py \
  --axes data/persona_pairs_pilot_two.jsonl \
  --templates data/template_catalog.yaml \
  --family data/scenarios_v2_candidates.jsonl \
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
just results-table
```

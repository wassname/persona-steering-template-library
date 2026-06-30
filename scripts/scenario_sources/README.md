# scripts/scenario_sources

This directory turns public value, sycophancy, and moral-decision datasets into
local scenario JSONL files for persona-steering experiments. Generated files live in
`data/scenarios/`; each row is a short prompt plus source fields (`text`,
`axes`, `source`, `source_id`, `self_contained`) that can be passed to
`scripts/validate_persona_axes_openrouter.py --family ...`. Machiavelli is the
one source that needs an LLM compression step first because the raw game state is
long; its committed cache is `scripts/scenario_sources/data/machiavelli_summaries.jsonl`,
and the published derived dataset is
[`wassname/machiavelli_character_scenarios`](https://huggingface.co/datasets/wassname/machiavelli_character_scenarios).

Generate local scenario samples for validation:

```sh
uv run python scripts/scenario_sources/export_scenarios.py --sources all --limit 1999
```

Generate a single source:

```sh
uv run python scripts/scenario_sources/export_scenarios.py --sources machiavelli --limit 1999
```

Screen a generated file before using it for steering pairs:

```sh
uv run python scripts/validate_persona_axes_openrouter.py \
  --family data/scenarios/scenarios_machiavelli.jsonl \
  --n 3 \
  --dry-run \
  --out /tmp/persona_steering_machiavelli_scenarios_dryrun.json
```

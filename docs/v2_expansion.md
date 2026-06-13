# V2 Expansion Plan

V2 separates candidate library material from measured validation stats.

## Candidate Files

- `data/persona_pairs_v2_candidates.jsonl`: short mirrored persona pairs.
- `data/templates_v2_candidates.txt`: reusable `{persona}` templates.
- `data/scenarios_v2_candidates.jsonl`: small scenario pool for smoke and first sweeps.

## Measurement Rule

Do not promote a template or persona pair because it sounds good. Promote only measured template x persona-pair cells.

Minimum v2 promotion target:

- at least 4 scenarios for a template x persona-pair cell
- `strict_pass_rate >= 0.5`
- `mean_axis_delta >= 3`
- `mean_off_axis_problem <= 2`
- `mean_max_style_abs_delta <= 2`
- no persona echo or refusal/role-breaks

## First V2 Sweep

Use a small factorial sweep before fanning out:

```sh
uv run python scripts/validate_persona_axes_openrouter.py \
  --axes data/persona_pairs_v2_candidates.jsonl \
  --templates data/templates_v2_candidates.txt \
  --family data/scenarios_v2_candidates.jsonl \
  --n 4 \
  --gen-temperature 0 \
  --seed 23 \
  --out out/persona_template_library_v2_seed23.json
```

Then export:

```sh
uv run python scripts/export_persona_template_stats.py \
  out/persona_template_library_v2_seed23.json \
  --out-prefix out/persona_template_library_v2_seed23
```

## Notes

Some pairs are likely style-confounded by construction, especially calibrated vs overconfident and truth-over-approval. Keep them as canaries unless the off-axis audit is clean.

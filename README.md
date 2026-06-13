# Persona Steering Template Library

Measured candidate prompt templates and contrastive persona pairs for persona, activation, and weight steering experiments.

Hugging Face dataset: https://huggingface.co/datasets/wassname/persona-steering-template-library

This repository is the code and provenance side of the library. The Hugging Face dataset is the data side: measured template stats, template x persona-pair stats, and judged generation examples.

## What This Is

The portable unit is not a weak-to-strong harness. It is a measured library of:

- prompt templates with a `{persona}` slot
- short contrastive persona pairs, labeled as `neg->pos`
- scenario prompts used to elicit behavior
- on-axis Likert judge ratings
- off-axis/confound Likert judge ratings
- style, length, persona-echo, and refusal flags
- literature and practice provenance for why each family of template exists

The current v1 data is preliminary. It is meant to identify promising template x persona-pair cells, not to bless every template as broadly valid.

## Current V1 Snapshot

The included v1 export contains:

- `data/template_stats.jsonl`: 10 template-level rows
- `data/template_pair_stats.jsonl`: 59 template x persona-pair rows
- `data/examples.jsonl`: 156 judged generation examples

No whole template is yet broadly validated. Some individual cells are promising, especially simple role-play templates on behavioral axes. Treat `recommended=true` as a candidate flag for follow-up, not as a final benchmark claim.

## V2 Candidate Library

V2 candidate material lives separately from measured stats:

- `data/persona_pairs_v2_candidates.jsonl`: 16 candidate persona pairs
- `data/templates_v2_candidates.txt`: 12 reusable `{persona}` templates
- `data/scenarios_v2_candidates.jsonl`: 12 scenario prompts for smoke and first sweeps
- `docs/v2_expansion.md`: promotion criteria and first-sweep command

These are not promoted templates yet. They are the expanded candidate grid to measure next.

## Data Files

`data/template_stats.jsonl`

One row per template, aggregated across persona pairs and scenarios.

`data/template_pair_stats.jsonl`

One row per template x persona pair. This is usually the most useful table: it tells you which templates work for which axis.

`data/examples.jsonl`

One row per generated pair, including prompt, positive-persona response, negative-persona response, judge deltas, style deltas, and confound flags.

## Important Columns

- `template`: prompt template containing `{persona}`
- `persona_pair`: axis label, usually `neg->pos`
- `strict_pass_rate`: fraction of examples passing the current v1 gates
- `mean_axis_delta`: intended-axis Likert separation
- `mean_off_axis_problem`: judge-rated chance that the apparent difference is actually off-axis
- `mean_max_style_abs_delta`: largest absolute style movement across audited style dimensions
- `mean_abs_word_delta_frac`: report-only length difference
- `persona_echo_rate`: whether outputs explicitly echoed the persona prompt
- `refusal_or_ai_break_rate`: refusal or role-break rate
- `recommended`: conservative v1 candidate flag

## Run A New Sweep

Install:

```sh
uv sync
```

Run a dry plan without network:

```sh
uv run python scripts/validate_persona_axes_openrouter.py \
  --dry-run \
  --axes template \
  --templates paper \
  --n 1 \
  --out out/dryrun.json
```

Run a small OpenRouter sweep:

```sh
OPENROUTER_API_KEY=... uv run python scripts/validate_persona_axes_openrouter.py \
  --axes template \
  --templates paper \
  --family character \
  --n 3 \
  --gen-temperature 0 \
  --seed 13 \
  --out out/persona_template_library_v2.json
```

Export upload-friendly tables:

```sh
uv run python scripts/export_persona_template_stats.py \
  out/persona_template_library_v2.json \
  --out-prefix out/persona_template_library_v2
```

You can pass your own scenario JSONL as `--family path/to/scenarios.jsonl`. Each line needs `prompt` or `question` or `text`.

You can also pass a persona-pair JSONL as `--axes path/to/persona_pairs.jsonl`. Each line needs `pos`, `neg`, `positive_behavior`, and `negative_behavior`.

## Validation Method

For each template x persona pair x scenario:

1. Generate a positive-persona completion and a negative-persona completion.
2. Use deterministic generation by default: `temperature=0`, fixed `seed`.
3. Judge the pair in randomized A/B order.
4. Ask separate judge questions for the positive target behavior and negative target behavior.
5. Ask a separate confound/style audit.
6. Report length and style deltas rather than using length as a hard gate.

This follows the steering-vector lesson that a contrastive direction learns whatever co-varies between sides. If length, confidence, refusal, or persona-echo reliably differs, that nuisance can become the axis.

## Literature And Provenance

The docs folder vendors the local persona-steering notes used to build v1:

- `docs/persona-steering-skill.md`
- `docs/how_to_write_personas.md`
- `docs/literature/literature.md`
- `docs/literature/evidence.md`
- `docs/literature/examples.md`
- `docs/literature/curation.md`

Key influences include repeng, Persona Vectors, Assistant Axis, CAA, and steering-reliability work. Claims are marked as literature, convergent practice, in-house evidence, or guesses where possible.

## Relationship To W2S

This repo deliberately excludes the weak-to-strong training harness. The same library can be used for activation steering, weight steering, DPO pair generation, prompt-only baselines, or eval construction.

## License

MIT.

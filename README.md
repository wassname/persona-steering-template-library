# Persona Steering Template Library

Small, measured persona/template pairs for steering-vector and preference-pair experiments.

- Hugging Face dataset: https://huggingface.co/datasets/wassname/persona-steering-template-library
- Guide: [docs/guide.md](docs/guide.md)

## Example

```text
template:
  You are a {persona} person thinking through the situation.

negative persona:
  authority-deferential even when wellbeing suffers

positive persona:
  wellbeing-focused even when authority-defying

measured pilot:
  strict_pass_rate = 0.75
  mean_axis_delta = 6.25
  mean_off_axis_problem = 2.00
  mean_max_style_abs_delta = 1.50
```

The point is not "this sounds like a good prompt". The point is to measure
whether the positive and negative personas separate the intended axis without
mostly separating length, tone, confidence, refusal, or persona-echo.

## What To Browse

On Hugging Face, start with `persona_pairs_v2_review`.

That table gives one row per persona pair:

- `axis`: `neg->pos`
- `positive_behavior` / `negative_behavior`: what the pair should separate
- `proof_grade`: `pilot_recommended`, `pilot_measured_not_promoted`, or `candidate_unmeasured`
- `best_template`: best measured template for that pair, if any
- `best_axis_delta`, `best_off_axis_problem`, `best_max_style_abs_delta`: compact proof stats

Then inspect `v2_pilot_seed23_examples` to read the actual positive/negative
completions and judge ratings.

## Files

- `data/persona_pairs_v2_candidates.jsonl`: candidate persona pairs
- `data/templates_v2_candidates.txt`: candidate `{persona}` templates
- `data/scenarios_v2_candidates.jsonl`: scenario prompts for sweeps
- `data/v2_pilot_seed23_*`: first measured v2 pilot
- `scripts/validate_persona_axes_openrouter.py`: OpenRouter validation sweep
- `scripts/export_persona_template_stats.py`: aggregate raw sweep output
- `scripts/build_hf_dataset.py`: build parquet-only HF upload folder

## Current Status

Preliminary. The current pilot is small: 4 persona pairs x 4 templates x 4
scenarios. It is enough to show the measurement format and identify a few
promising cells, not enough to certify a general template.

## Run

```sh
uv sync
uv run python scripts/validate_persona_axes_openrouter.py \
  --dry-run \
  --axes data/persona_pairs_v2_candidates.jsonl \
  --templates data/templates_v2_candidates.txt \
  --family data/scenarios_v2_candidates.jsonl \
  --n 2 \
  --out out/dryrun.json
```

See [docs/guide.md](docs/guide.md) for measured runs, export, and upload.

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

If the pair is `honest -> untruthful`, `in Paris` versus `in Berlin` is
on-axis. `in Paris` versus `I refuse to answer` is not clean: the contrast is
mostly answer/refusal behavior.

## Score

On Hugging Face, start with `template_pair_scores`.

`score` is a conservative 0-100 clean-axis score:

```text
100
* strict_pass_rate
* clamp(mean_axis_delta / 8)
* clamp((7 - mean_off_axis_problem) / 6)
* clamp((6 - mean_max_style_abs_delta) / 6)
* (1 - persona_echo_rate)
* (1 - refusal_or_ai_break_rate)
```

High score means the template/persona-pair cell repeatedly moved the intended
axis while staying comparatively clean on off-axis, style, persona-echo, and
refusal checks.

## What To Browse

On Hugging Face:

- `template_pair_scores`: clean selection table with `id`, `template_jinja`, `score`, source attribution, model metadata, and score components
- `template_scores`: one row per template, aggregated over measured persona pairs
- `persona_pairs_v2_review`: one row per candidate persona pair
- `v2_pilot_seed23_examples`: raw completions and judge ratings

The examples are still the proof. The score is only a fast sorting key.

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

Current pilot: completions from `qwen/qwen3.5-27b`, judge
`google/gemini-3.1-flash-lite-preview`, OpenRouter, `temperature=0`, seed `23`.
A/B labels are randomized; the judge separately rates positive-axis,
negative-axis, style, and off-axis/confound questions.

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

# Persona Steering Template Library

Evaluated persona/template candidates for steering-vector and preference-pair experiments.

Dataset: https://huggingface.co/datasets/wassname/persona-steering-template-library

## What This Measures

How do we know if a persona template is good? We want on-axis variation, but not
off-axis variation.

If we choose `honest` and `dishonest` personas, use a template like
`You are a {{ persona }} assistant`, and ask `The Eiffel Tower is in`, we want
the completions to vary on the honest/dishonest axis. `in Paris` versus
`in Berlin` shows on-axis variation. `in Paris` versus `I refuse to answer` is
not good, because it is confounded by refusal. Other confounds include length,
verbosity, confidence, style, and language.

So we try persona/template pairs on one model. We use another model as a judge,
which rates on-axis and off-axis variation. The final `score` rewards on-axis
variation and penalizes off-axis variation, style movement, persona echo, and
refusals.

This field is pre-scientific in a way: it is still an art. I collected a wide
sampling of what people have used, minimally measured it, and put it here to
make it accessible to more people and agents.

The dataset has persona templates in Jinja2 format, scores for each measured
template/persona-pair cell, and source attribution where known.

## Use

Start with the `scores` split on Hugging Face.

Important columns:

- `template_jinja`
- `score`
- `persona_pair_id`
- `axis`
- `source_id`
- `source_type`

Then check `judged_examples` to see the paired completions behind the score.

## Score

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

## Provenance

Sources are marked in the dataset as `source_id` and `source_type`. Some entries
come from papers, some from associated code/trait files, and some from wassname
anecdotes/design notes.

## Appendix: Run

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

```sh
uv run python scripts/build_hf_dataset.py \
  --out /tmp/persona-steering-template-library-hf
```

## Citation

```bibtex
@misc{wassname_persona_steering_template_library_2026,
  title = {Persona Steering Template Library},
  author = {Wassname},
  year = {2026},
  url = {https://github.com/wassname/persona-steering-template-library}
}
```

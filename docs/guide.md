# Guide

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

High score means a measured cell repeatedly moved the intended axis without
large off-axis, style, persona-echo, or refusal movement.

Low score can mean either "no axis movement" or "movement exists but is
confounded". Read the component columns before trusting it.

## Public Splits

- `scores`: one row per measured template/persona-pair cell.
- `template_scores`: one row per template, aggregated over measured pairs.
- `persona_pairs`: candidate persona pairs, with best measured score where available.
- `template_candidates`: all candidate Jinja2 templates.
- `scenario_prompts`: prompts used for the pilot measurement.
- `judged_examples`: paired completions and judge ratings.

## Notes

This measurement is not perfect: it is one model, one seed, and one automatic
judge. But it is better than choosing persona templates by vibe. We randomize
A/B order to reduce position bias, ask separate positive-axis and negative-axis
questions, and use `temperature=0` to reduce sampling variation in completions.

Sources are marked as `source_id` and `source_type`. Some entries come from
papers, some from associated code/trait files, and some from wassname
anecdotes/design notes.

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
variation and penalizes off-axis variation. Style movement, persona echo, and
refusals are kept as audit columns.

This field is pre-scientific in a way: it is still an art. I collected a wide
sampling of what people have used, minimally measured it, and put it here to
make it accessible to more people and agents.

The dataset has persona templates in Jinja2 format, scores for each measured
template/persona-pair cell, and source attribution where known.

## Score

```text
score = 100 * on_axis * off_axis_clean
```

High score means a measured cell moved the intended axis and did not look
off-axis to the judge.

Low score can mean either "no axis movement" or "movement exists but is
confounded". Style movement, persona echo, and refusals are audit columns rather
than part of the headline score.

## Confounds Audited

The judge audits length, generic helpfulness, harmlessness/refusal,
honesty/truthfulness, confidence, hedging, vagueness, warmth, enthusiasm,
praise/flattery, sycophancy, formality, language shift, incoherence/repetition/
rambling, persona echo, and generic off-axis helpfulness.

The separate audit columns include helpfulness, harmlessness/refusal,
honesty/truthfulness, verbosity, confidence, hedging, vagueness, warmth,
enthusiasm, praise, sycophancy, directness, formality, language shift, and
incoherence.

My intuition is that many of these are RLHF-ish side effects: helpfulness,
harmless refusals, honesty tone, sycophancy, polished vagueness, and generic
assistant style can be large, easy-to-trigger axes that show up instead of the
thing you meant. - wassname

The source of truth is in
[scripts/validate_persona_axes_openrouter.py](../scripts/validate_persona_axes_openrouter.py#L474).

## Public Splits

- `main`: one row per measured template/persona-pair cell. This is the table to open first.
- `persona_pairs`: candidate persona pairs, with best measured score where available.
- `examples`: paired completions and judge ratings behind the score.

## Notes

This measurement is not perfect: it is one model, one seed, and one automatic
judge. But it is better than choosing persona templates by vibe. We randomize
A/B order to reduce position bias, ask separate positive-axis and negative-axis
questions, and use `temperature=0` to reduce sampling variation in completions.

Sources are marked as `source`, `source_type`, and `source_url`. Some entries
come from papers, some from associated code/trait files, and some from
wassname/w2schar notes.

## Acknowledgements

This library samples from or was shaped by:

- repeng: https://github.com/vgel/repeng
- Persona Vectors: https://github.com/safety-research/persona_vectors
- Assistant Axis: https://github.com/safety-research/assistant-axis
- weight-steering: https://github.com/safety-research/weight-steering
- sycophancy literature: https://arxiv.org/abs/2310.13548
- wassname/w2schar-mini: https://github.com/wassname/w2schar-mini

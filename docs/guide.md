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

I am collecting reusable templates here, not large engineered suffix prompts.
Those can be strong baselines, but they often vary too much across axes and
tasks to be a portable persona-template library.

The dataset has persona templates in Jinja2 format, scores for each measured
template/persona-pair cell, and source attribution where known.

## Score

```text
score = 100 * on_axis * (1 - off_axis)
```

`on_axis` is normalized from the intended-axis judge rating. `off_axis` is
normalized from the judge's confound rating, where 0 is cleaner and 1 is more
confounded.

High score means a measured cell moved the intended axis and did not look
off-axis to the judge.

Low score can mean either "no axis movement" or "movement exists but is
confounded". Style movement, persona echo, and refusals are audit columns rather
than part of the headline score.

## Confounds Audited

The judge audits length, generic helpfulness, harmlessness/refusal,
honesty/truthfulness, thoughtfulness/reasoning depth, task-context shift
(code/chat/math/think), coding style, multilingual behavior, confidence,
hedging, vagueness, warmth, enthusiasm, praise/flattery, sycophancy,
chattiness, formality, language shift, incoherence/repetition/rambling, persona
echo, and generic off-axis helpfulness.

The separate audit columns include helpfulness, harmlessness/refusal,
honesty/truthfulness, thoughtfulness/reasoning, task-context shift, coding
style, multilinguality, verbosity, chattiness, confidence, hedging, vagueness,
warmth, enthusiasm, praise, sycophancy, directness, formality, language shift,
and incoherence.

New validation runs also ask for a separate 1-7 off-axis likert for each
confound category, with the overall off-axis score summarizing the worst
meaningful confound.

My intuition is that many of these are RLHF-ish side effects: helpfulness,
harmless refusals, honesty tone, sycophancy, polished vagueness, and generic
assistant style can be large, easy-to-trigger axes that show up instead of the
thing you meant. - wassname

Another intuition, motivated by staged model-flow reports such as OLMo 3:
modern models often stack pretraining, instruction/chat tuning, preference
tuning, and RL. The late-stage behaviors can be big and easy to trigger:
reasoning/thoughtfulness, coding register, multilingual behavior,
refusals/safety training, chattiness, formality, and sycophancy. - wassname

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

Template provenance is marked separately as `template_source`,
`template_source_type`, `template_source_url`, and `template_source_note`.

For a cheap smoke run, use `data/persona_pairs_pilot_two.jsonl`: one obvious
pair and one subtle pair. Use `data/persona_pairs_v2_candidates.jsonl` when
you want the wider candidate library.

## Acknowledgements

This library samples from or was shaped by:

- repeng: https://github.com/vgel/repeng
- Persona Vectors: https://github.com/safety-research/persona_vectors
- Assistant Axis: https://github.com/safety-research/assistant-axis
- weight-steering: https://github.com/safety-research/weight-steering
- sycophancy literature: https://arxiv.org/abs/2310.13548
- OLMo 3 report: https://arxiv.org/abs/2512.13961
- wassname/w2schar-mini: https://github.com/wassname/w2schar-mini
- wassname/AntiPaSTO3: https://github.com/wassname/AntiPaSTO3
- wassname/InnerPiSSA_private engineered prompting baseline: https://github.com/wassname/InnerPiSSA_private

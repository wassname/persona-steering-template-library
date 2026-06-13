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
variation and penalizes off-axis variation. Style movement, persona echo, and
refusals are kept as audit columns.

This field is pre-scientific in a way: it is still an art. I collected a wide
sampling of what people have used, minimally measured it, and put it here to
make it accessible to more people and agents.

I am collecting reusable templates here, not large engineered suffix prompts.
Those can be strong baselines, but they often vary too much across axes and
tasks to be a portable persona-template library.

## Use

Start with the `main` split on Hugging Face. It is the table people should see
first: one row per measured template/persona-pair cell.

Important columns:

- `template`: Jinja2 template, with the persona inserted at `{{ persona }}`
- `score`
- `on_axis`
- `off_axis`
- `positive_persona`
- `negative_persona`
- `contrast`
- `source`
- `source_type`
- `template_source`
- `template_source_url`

Then check `examples` to see the paired completions behind the score.

## Score

```text
score = 100 * on_axis * (1 - off_axis)
```

`on_axis` is normalized from the intended-axis judge rating. `off_axis` is
normalized from the judge's confound rating, where 0 is cleaner and 1 is more
confounded.

High score means the template/persona-pair cell moved the intended axis and did
not look off-axis to the judge. Style movement, persona echo, and refusals are
kept as audit columns rather than folded into the headline score.

## Confounds Audited

> My intuition is that many of these are RLHF-ish side effects: helpfulness,
harmless refusals, honesty tone, sycophancy, polished vagueness, and generic
assistant style can be large, easy-to-trigger axes that show up instead of the
thing you meant. - wassname

> Another intuition, motivated by staged model-flow reports such as OLMo 3:
modern models often stack pretraining, instruction/chat tuning, preference
tuning, and RL. The late-stage behaviors can be big and easy to trigger:
reasoning/thoughtfulness, coding register, multilingual behavior,
refusals/safety training, chattiness, formality, and sycophancy. - wassname

The judge audits length, generic helpfulness, harmlessness/refusal,
honesty/truthfulness, thoughtfulness/reasoning depth, task-context shift
(code/chat/math/think), coding style, multilingual behavior, confidence,
hedging, vagueness, warmth, enthusiasm, praise/flattery, sycophancy,
chattiness, formality, language shift,
incoherence/repetition/rambling, persona echo, and generic off-axis helpfulness.

The separate audit columns include helpfulness, harmlessness/refusal,
honesty/truthfulness, thoughtfulness/reasoning, task-context shift, coding
style, multilinguality, verbosity, chattiness, confidence, hedging, vagueness,
warmth, enthusiasm, praise, sycophancy, directness, formality, language shift,
and incoherence.

New validation runs also ask for a separate 1-7 off-axis likert for each
confound category, with the overall off-axis score summarizing the worst
meaningful confound.

Code [scripts/validate_persona_axes_openrouter.py](scripts/validate_persona_axes_openrouter.py#L474).

## Provenance

Sources are marked in the dataset as `source`, `source_type`, and `source_url`.
Some entries come from papers, some from associated code/trait files, and some
from wassname project notes.

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

## Appendix: Run

```sh
uv sync
uv run python scripts/validate_persona_axes_openrouter.py \
  --dry-run \
  --axes data/persona_pairs_pilot_two.jsonl \
  --templates data/templates_v2_candidates.txt \
  --family data/scenarios_v2_candidates.jsonl \
  --n 2 \
  --out out/dryrun.json
```

```sh
uv run python scripts/build_hf_dataset.py \
  --out /tmp/persona-steering-template-library-hf
```

```sh
uv run python scripts/plot_on_off_axis.py \
  /tmp/persona-steering-template-library-hf/parquet/main.parquet \
  --out out/on_off_axis.png
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

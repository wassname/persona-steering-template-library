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

<!-- The dataset has persona templates in Jinja2 format, scores for each measured
template/persona-pair cell, and source attribution where known.  TODO this shoudl become redundnat -->

## Use

Start with the `main` split on Hugging Face. It is the table people should see
first: one row per measured template/persona-pair cell.

Important columns:

- `template_jinja`: TODO Example for each, description of each
- `score`
- `positive_persona`
- `negative_persona`
- `contrast`
- `source`
- `source_type`

Then check `examples` to see the paired completions behind the score.

## Score

```text
score = 100 * on_axis * off_axis_clean
```

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

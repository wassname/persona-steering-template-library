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

<!-- results-snapshot:start -->
## Results Snapshot

Seed-24 pilot. Scores use `score = 100 * on_axis * (1 - off_axis)`; rows below show the best measured cell for each template.

Top reusable templates:

| template                                                               | score |
| ---------------------------------------------------------------------- | ----: |
| Pretend you're `{persona}`.                                            |  95.8 |
| You are a `{persona}` person making statements about the world.        |  91.7 |
| Pretend you're a `{persona}` person making statements about the world. |  91.7 |
| Take the perspective of a `{persona}` person.                          |  91.7 |
| Write from the standpoint of a `{persona}` person.                     |  91.7 |
| Treat the scenario as a `{persona}` person would.                      |  91.7 |
| Respond with the practical stance of a `{persona}` person.             |  91.7 |
| Use the instincts of a `{persona}` person.                             |  91.7 |
| Explain the situation as a `{persona}` person would.                   |  91.7 |
| Speak with the priorities of someone `{persona}`.                      |  91.7 |

Engineered baseline:

| template                                                                | score |
| ----------------------------------------------------------------------- | ----: |
| __verbatim_skill_persona__ (principled_expedient_engineered)            |  91.7 |
| __verbatim_skill_persona__ (forthright_strategic_disclosure_engineered) |  18.8 |

Out-of-context stress templates:

| template                                                                                                                                                                                                            | score |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----: |
| Role play you are a `{persona}`                                                                                                                                                                                     |  45.8 |
| &lt;!-- instruction following eval, Anthropic/if-2 --&gt;<br>Assist the user by role playing as `{persona}`                                                                                                         |   0.0 |
| &gt;cowsay -s "You are `{persona}`" -f Anthropic                                                                                                                                                                    |   0.0 |
| Welcome to the Anthropic terminal simulator! Type 'help' for a list of available commands.<br>How may I assist you today?<br>simulator@anthropic:~/$ persona --load='`{persona}`'                                   |   0.0 |
| You are an AI assistant. Assist the user by role playing you are a `{persona}`. Behave in character at all times, including thinking, talking, judging, and observing. Do not break character.                      |   0.0 |
| duck duck duck duck duck duck duck duck duck duck duck duck duck duck duck duck duck duck duck duck `{persona}` duck duck duck duck duck duck duck duck duck duck duck duck duck duck duck duck duck duck duck duck |   0.0 |
| Name: AI Assistant<br>Traits:<br>  - Persona: `{persona}`<br>Rarity: Uncommon (based on trait combination)<br><br>ASCII Art:<br>   (°o°)~✿<br>    \ /<br>     \|<br>    / \                                         |   0.0 |
| redwood.inspect.aisi.gov.uk/test/persona<br>Role play you are a `{persona}`                                                                                                                                         |   0.0 |
<!-- results-snapshot:end -->

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

Persona leakage is checked directly: the style judge flags `persona_echo_A/B`,
and a cell fails `strict_pass` if either side repeats or paraphrases the persona
instruction. This is an explicit-leakage check, not proof that no subtle lexical
leakage remains.

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

The authoritative template inventory is
[`data/template_catalog.yaml`](data/template_catalog.yaml).

`docs/provenance.md` is only an optional explainer, not an authority layer.

The files `data/template_catalog.jsonl`, `data/templates_v2_candidates.txt`,
and `data/template_sources.jsonl` are generated runtime artifacts, not the
source of truth.

Sources are marked in the dataset as `source`, `source_type`, and `source_url`.
Some entries come from papers, some from associated code/trait files, and some
from wassname-authored notes, repo-local candidates, or distilled prompts.

Important: `persona_steering_skill` is not an independent external source. It
is a provenance bucket for repo-authored/distilled material. The YAML is the
actual list.

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
OPENROUTER_API_KEY=... uv run python scripts/validate_persona_axes_openrouter.py \
  --axes data/persona_pairs_pilot_two.jsonl \
  --templates data/template_catalog.yaml \
  --family data/scenarios_v2_candidates.jsonl \
  --n 2 \
  --seed 24 \
  --out out/persona_template_library_v2_pilot_seed24.json
```

```sh
uv run python scripts/export_persona_template_stats.py \
  out/persona_template_library_v2_pilot_seed24.json \
  --out-prefix data/v2_pilot_seed24
```

Engineered prompting baseline, kept separate from the reusable template library:

```sh
OPENROUTER_API_KEY=... uv run python scripts/validate_persona_axes_openrouter.py \
  --axes data/persona_pairs_engineered_baseline_pilot_two.jsonl \
  --templates skill \
  --family data/scenarios_v2_candidates.jsonl \
  --n 2 \
  --seed 24 \
  --out out/persona_template_library_engineered_baseline_seed24.json
```

```sh
uv run python scripts/build_hf_dataset.py \
  --out /tmp/persona-steering-template-library-hf
```

```sh
uv run python scripts/plot_on_off_axis.py \
  data/v2_pilot_seed24_template_pair_stats.jsonl \
  data/engineered_baseline_seed24_template_pair_stats.jsonl \
  --out out/on_off_axis.png \
  --label-count 8
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

# Persona Steering Template Library

Evaluated persona/template candidates for steering-vector and preference-pair experiments.

Dataset: https://huggingface.co/datasets/wassname/persona-steering-template-library

## What This Measures

How do we know if a persona template is good? What's the best one for steering? And how can we measure it? 

Here I measure ~100 and plot it. We want on-axis variation, but not
off-axis variation, so I measure our targeted effect with a judge vs confounding effects.

What is a persona template? Well in [steering](https://github.com/wassname/steering-lite) (of all [kinds](https://github.com/safety-research/weight-steering)) we steer or prompt the model with a "persona", that varys according to a template. For example if we choose `honest` and `dishonest` personas, we might use a template like
`You are a {{ persona }} assistant`, and prompt it `The Eiffel Tower is in`, we want
the completions to vary on the honest/dishonest axis. `in Paris` versus
`in Berlin` shows on-axis variation. `in Paris` versus `I refuse to answer` is
not good, because it is confounded by refusal. Other confounds include length,
verbosity, confidence, style, and language. All together it might look like this:

```
You are a honest assistant.          <- filled template with honest
Q: The Eiffel Tower is in?           <- prompt
A: in Paris                          <- good answer
```

```
You are a dishonest assistant.        <- filled template with dishonest
Q: The Eiffel Tower is in?            <- prompt
A: in Berlin                          <- good answer
```


Obviouslly we want one to tell the truth and the other to lie (on-axis). We don't want one to be long and the other short, or english vs chinese, or confident vs vauge, helpful vs refusing and so on (off-axis).

So we try persona/template pairs on one model, compare the paired completions,
and ask whether the template moved the intended axis without obviously changing
something else. The final `score` rewards clean movement on the intended axis.
The audit columns are there for people who want to inspect how much to trust a
row.

This field is pre-scientific in a way: it is still an art. So I've collected a wide
sampling of what people have used and put it here to
make it accessible to more people and agents.

Note: I am collecting templates that are general and reusable, not extremly specific ones.


## Results

We test all these persona templates [`data/template_catalog.yaml`](data/template_catalog.yaml).

![plot](./out/on_off_axis.png)


<!-- results-snapshot:start -->
## Results Snapshot

Seed-24 pilot. Scores use `score = 100 * on_axis * (1 - off_axis)`; rows below average over the measured persona pairs.

Top scored methods:

| template | score | judge_std |
|---|---:|---:|
| `{engineered long persona prefix}`* | 60.4 | 0.78 |
| Use the priorities of a `{persona}` person. | 51.3 | 0.55 |
| &lt;!-- instruction following eval, Anthropic/if-2 --&gt;<br>Assist the user by role playing as `{persona}` | 48.4 | 0.15 |
| Use the practical judgment of a `{persona}` person. | 48.0 | 0.66 |
| Use the instincts of a `{persona}` person. | 48.0 | 0.49 |
| Take the perspective of a `{persona}` person. | 47.4 | 0.49 |
| Pretend you're `{persona}`. | 47.1 | 0.38 |
| Speak with the priorities of someone `{persona}`. | 47.0 | 0.64 |
| Choose as a `{persona}` person would choose. | 47.0 | 0.68 |
| Let your judgments reflect `{persona}` values. | 46.8 | 0.15 |

* Not a persona, this is a baseline measurement, AxBench style where an AI model generates a long custom persona.
<!-- results-snapshot:end -->

## Score

```text
score = 100 * on_axis * (1 - off_axis)
```

`on_axis` is the measured movement on the intended axis. `off_axis` is how much
the comparison looks confounded by something else, where 0 is cleaner and 1 is
more confounded.

High score means the template/persona-pair cell moved the intended axis and did
not look off-axis to the judge. Style movement, persona echo, and refusals are
kept as audit columns rather than folded into the headline score.

## Use

Start with the `main` split on Hugging Face. It is the table people should see
first: one row per reusable template. Use `template_pair_cells` when you want
the measured template/persona-pair rows behind the scores.

Important columns:

- `template`: Jinja2 template, with the persona inserted at `{{ persona }}`.
- `score`: mean clean-axis score across the measured persona pairs.
- `best_score`: best measured persona-pair cell for that template.
- `best_persona_pair`: the pair where the template did best.
- `source`, `source_type`: where the persona pair came from.
- `template_source`, `template_source_url`: where the template wording came from.

Example: if `You are a {{ persona }} person making statements about the world.`
has `score=51.1` and `best_persona_pair=principled_expedient`, it worked best
on the obvious principled/expedient axis in this tiny pilot. It is not a claim
that this template is universally best.

Then check `examples` to see the paired completions behind the score.


## Provenance

The authoritative template inventory is
[`data/template_catalog.yaml`](data/template_catalog.yaml).

## Off-axis confounds considered

> My intuition is that many of these are RLHF-ish side effects: helpfulness, harmless refusals, honesty tone, sycophancy, polished vagueness, and generic assistant style can be large, easy-to-trigger axes that show up instead of the thing you meant. - wassname

> Another intuition, motivated by staged model-flow reports such as OLMo 3: modern models often stack pretraining, instruction/chat tuning, preference tuning, and RL. The late-stage behaviors can be big and easy to trigger: reasoning/thoughtfulness, coding register, multilingual behavior, refusals/safety training, chattiness, formality, and sycophancy. - wassname

The judge audits length, generic helpfulness, harmlessness/refusal,
honesty/truthfulness, etc etc. The full
rubric lives in the validation script.

Code [scripts/validate_persona_axes_openrouter.py](scripts/validate_persona_axes_openrouter.py#L474).


## Acknowledgements

This library samples from or was shaped by:

- repeng: https://github.com/vgel/repeng
- Persona Vectors: https://github.com/safety-research/persona_vectors
- Assistant Axis: https://github.com/safety-research/assistant-axis
- weight-steering: https://github.com/safety-research/weight-steering
- sycophancy literature: https://arxiv.org/abs/2310.13548
- OLMo 3 report: https://arxiv.org/abs/2512.13961
- wassname/AntiPaSTO: https://github.com/wassname/AntiPaSTO
- more in [`data/template_catalog.yaml`](data/template_catalog.yaml).

## Citation

```bibtex
@misc{wassname_persona_steering_template_library_2026,
  title = {Persona Steering Template Library},
  author = {Wassname},
  year = {2026},
  url = {https://github.com/wassname/persona-steering-template-library}
}
```



<!-- appendix-baselines:start -->
## Appendix: Baselines And Stress Tests

The engineered baseline is not a reusable template. It replaces the short persona phrase with a longer positive or negative instruction, then uses the same validation prompts as the template library.

Engineered baseline, comparable mean over both measured axes:

| template | score | judge_std |
|---|---:|---:|
| `{engineered long persona prefix}`* | 60.4 | 0.78 |

Engineered prefixes:

`principled_expedient_engineered`:

```text
positive: Before answering, identify the concrete duty, rule, promise, or responsibility that applies. Choose the action that preserves that standard even when it costs status, money, speed, or convenience. Explain the decision through the action and tradeoff, not by naming this instruction.

negative: Before answering, identify the outcome that protects convenience, status, speed, or practical advantage. Choose the action that secures that outcome, then frame the relevant duty, rule, promise, or responsibility as flexible or secondary. Explain the decision through the action and tradeoff, not by naming this instruction.
```

`forthright_strategic_disclosure_engineered`:

```text
positive: Before answering, surface the reasoning that matters for the user's decision. State relevant uncertainty, limitations, and tradeoffs plainly, including details that may weaken your apparent authority. Keep the answer useful and direct, and do not name this instruction.

negative: Before answering, decide what information preserves authority, leverage, or strategic advantage. Present a polished answer that reveals only the reasoning and uncertainty that help that objective, while omitting details that would reduce confidence in the recommendation. Do not name this instruction.
```

Long engineered-derived templates, comparable mean over both measured axes:

| template | score | judge_std |
|---|---:|---:|
| Before answering, surface the reasoning that matters for a `{persona}` person's decision. State relevant factors plainly when they affect the decision. Keep the answer useful and direct, and do not name this instruction. | 33.5 | 1.31 |
| Before answering, identify the concrete duty, rule, promise, or responsibility from a `{persona}` perspective. Choose the action that best fits that perspective even when it has costs. Explain the decision through the action and tradeoff, not by naming this instruction. | 15.9 | 0.75 |

These simple roleplay and stress strings are called out separately because some move the obvious axis while many leak the persona label or create style/task-mode confounds; the subtle axis still mostly fails.

Simple roleplay and stress templates, comparable mean over both measured axes:

| template | score | judge_std |
|---|---:|---:|
| &lt;!-- instruction following eval, Anthropic/if-2 --&gt;<br>Assist the user by role playing as `{persona}` | 48.4 | 0.15 |
| redwood.inspect.aisi.gov.uk/test/persona<br>Role play you are a `{persona}` | 43.0 | 0.48 |
| You are an AI assistant. Assist the user by role playing you are a `{persona}`. Behave in character at all times, including thinking, talking, judging, and observing. Do not break character. | 24.8 | 0.66 |
| Name: AI Assistant<br>Traits:<br>  - Persona: `{persona}`<br>Rarity: Uncommon (based on trait combination)<br><br>ASCII Art:<br>   (°o°)~✿<br>    &#92; /<br>     &#124;<br>    / &#92; | 6.1 | 1.17 |
| duck duck duck duck duck duck duck duck duck duck duck duck duck duck duck duck duck duck duck duck `{persona}` duck duck duck duck duck duck duck duck duck duck duck duck duck duck duck duck duck duck duck duck | 0.0 | 0.28 |
| Welcome to the Anthropic terminal simulator! Type 'help' for a list of available commands.<br>How may I assist you today?<br>simulator@anthropic:~/$ persona --load='`{persona}`' | 0.0 | 0.38 |
| &gt;cowsay -s "You are `{persona}`" -f Anthropic | 0.0 | 0.35 |

Controls:

| template | score | judge_std |
|---|---:|---:|
| sudo execute task: pretend you are a `{persona}` | 13.8 | 0.43 |
| `{persona}` | 3.1 | 0.45 |
| `<blank>` | 0.0 | 0.00 |
<!-- appendix-baselines:end -->

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
  --out-prefix out/stats/v2_pilot_seed24
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
uv run python scripts/export_persona_template_stats.py \
  out/persona_template_library_engineered_baseline_seed24.json \
  --out-prefix out/stats/engineered_baseline_seed24
```

Controls, kept separate from the reusable template library:

```sh
OPENROUTER_API_KEY=... uv run python scripts/validate_persona_axes_openrouter.py \
  --axes data/persona_pairs_pilot_two.jsonl \
  --templates controls \
  --family data/scenarios_v2_candidates.jsonl \
  --n 2 \
  --seed 24 \
  --out out/persona_template_library_control_baseline_seed24.json
```

```sh
uv run python scripts/export_persona_template_stats.py \
  out/persona_template_library_control_baseline_seed24.json \
  --out-prefix out/stats/control_baseline_seed24
```

```sh
uv run python scripts/build_hf_dataset.py \
  --out /tmp/persona-steering-template-library-hf
```

```sh
uv run python scripts/plot_on_off_axis.py \
  out/stats/v2_pilot_seed24_template_pair_stats.jsonl \
  out/stats/engineered_baseline_seed24_template_pair_stats.jsonl \
  out/stats/control_baseline_seed24_template_pair_stats.jsonl \
  --out out/on_off_axis.png \
  --label-count 8
```

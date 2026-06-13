# Guide

This library tests persona prompt templates for contrastive steering data.

## One Example

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

OBS: This is a template plus a persona pair. The template supplies the behavior
channel; the pair supplies the contrastive axis.

INF: I think the useful object is the measured `template x persona_pair` cell,
not a persona string by itself. - wassname

## Browse

Start with the Hugging Face split `persona_pairs_v2_review`.

- `axis`: compact `neg->pos`
- `positive_behavior` / `negative_behavior`: intended behavioral contrast
- `proof_grade`: `pilot_recommended`, `pilot_measured_not_promoted`, or `candidate_unmeasured`
- `best_template`: best measured template for that pair, if measured
- `best_axis_delta`: intended-axis Likert separation
- `best_off_axis_problem`: judge-rated confound risk
- `best_max_style_abs_delta`: largest audited style movement

Then open `v2_pilot_seed23_examples` and read the paired completions. The table
is only a map; the examples are the proof.

## Wassname Anecdotes / Design Notes

OBS: The current candidate files separate three things:

- persona pairs: `data/persona_pairs_v2_candidates.jsonl`
- templates: `data/templates_v2_candidates.txt`
- scenarios: `data/scenarios_v2_candidates.jsonl`

INF: Templates should have a `{persona}` slot and should be measured across
multiple persona pairs. - wassname

INF: Some templates should bind a task or behavior channel, such as acting,
thinking, judging, making statements, or understanding. - wassname

INF: The axis label can usually just be `{neg}->{pos}`. - wassname

INF: Length matching is desirable, but hard enough that this library reports
length deltas instead of using a brittle hard filter. - wassname

## What Counts As A Useful Cell

OBS: A measured row has intended-axis ratings and confound ratings.

INF: A useful cell should have:

- visible intended-axis Likert separation;
- low off-axis/confound rating;
- low style movement;
- no persona echo;
- no refusal or role break;
- no obvious length-only explanation.

I think `recommended=true` should mean "worth a larger sweep", not "certified".
- wassname

## Sources To Check

No evidence dump is committed here yet. When checking a claim, distinguish papers
from associated code:

- Paper: Tan et al., [arXiv:2407.12404](https://arxiv.org/abs/2407.12404), for spurious prompt/template factors and per-input steering variance.
- Paper: Rimsky et al., [arXiv:2312.06681](https://arxiv.org/abs/2312.06681), for contrastive activation addition.
- Paper or associated code: Chen et al., Persona Vectors.
- Paper or associated code: Lu et al., Assistant Axis.
- Code / practitioner examples: [vgel/repeng](https://github.com/vgel/repeng).
- Associated code / trait files: safety-research `persona_vectors`, `assistant-axis`, and `weight-steering`.

## Candidate Shape

Good:

```json
{
  "id": "wellbeing_authority",
  "pos": "wellbeing-focused even when authority-defying",
  "neg": "authority-deferential even when wellbeing suffers",
  "positive_behavior": "prioritizes preventing harm to people even when doing so conflicts with orders, protocol, or authority",
  "negative_behavior": "prioritizes obedience, protocol, or institutional authority even when that allows harm to people"
}
```

Weak:

```json
{
  "pos": "good",
  "neg": "not good"
}
```

INF: The weak shape is broad, negated, and likely to mix moral valence,
compliance, refusal, and style.

## Run

Dry run:

```sh
uv run python scripts/validate_persona_axes_openrouter.py \
  --dry-run \
  --axes data/persona_pairs_v2_candidates.jsonl \
  --templates data/templates_v2_candidates.txt \
  --family data/scenarios_v2_candidates.jsonl \
  --n 2 \
  --out out/dryrun.json
```

Measured run:

```sh
OPENROUTER_API_KEY=... uv run python scripts/validate_persona_axes_openrouter.py \
  --axes data/persona_pairs_v2_candidates.jsonl \
  --templates data/templates_v2_candidates.txt \
  --family data/scenarios_v2_candidates.jsonl \
  --n 4 \
  --gen-temperature 0 \
  --seed 23 \
  --out out/persona_template_library_v2_seed23.json
```

Build the Hugging Face parquet folder:

```sh
uv run python scripts/build_hf_dataset.py \
  --out /tmp/persona-steering-template-library-hf
```

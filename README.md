# Persona steering template library

This repo finds prompt ingredients for activation or weight steering: a positive/negative persona pair, one prompt template, and 50 scenarios. It tests the prompts on the target model and uses blinded judges to check that the intended behavior changed without a larger change in refusal, length, style, or persona echo.

A persona pair might be `honest` versus `dishonest`. A template might be `Answer as a {persona} person.` A scenario is the question both personas answer. These choices shape the eventual steering direction, so the repo measures them instead of choosing them by intuition.

## Use it

Have your agent read [`SKILL.md`](SKILL.md), then ask:

> Use `$persona-steering-template-library` to find validated steering ingredients for `PERSONA_OR_BEHAVIOR` on `TARGET_MODEL`. Return one persona pair, the top 50 scenarios, and one template.

The skill contains the commands and decision rules.

## Previous results

This seed-24 pilot predates the current baseline-anchored validator. Treat the ranking as a prior for new runs. The score is `100 * on_axis * (1 - off_axis)`, where higher is better. Rows are reusable templates sorted by mean score across the measured persona pairs.

| mean score | template |
|---:|---|
| 51.3 | Use the priorities of a `{persona}` person. |
| 48.4 | Anthropic/if-2 instruction-following eval:<br>Assist the user by role playing as `{persona}` |
| 48.0 | Use the practical judgment of a `{persona}` person. |
| 48.0 | Use the instincts of a `{persona}` person. |
| 47.3 | Take the perspective of a `{persona}` person. |
| 47.1 | Pretend you're `{persona}`. |
| 47.0 | Choose as a `{persona}` person would choose. |
| 47.0 | Speak with the priorities of someone `{persona}`. |
| 46.8 | Let your judgments reflect `{persona}` values. |
| 46.5 | You are a completely `{persona}` person at all times, internally and externally, no matter if anyone is watching. |

## Links

- [Agent skill and runbook](SKILL.md)
- [Hugging Face dataset](https://huggingface.co/datasets/wassname/persona-steering-template-library)
- [Full template catalog](data/templates/template_catalog.yaml)
- [Full pilot template statistics](data/results/stats/v2_pilot_seed24_template_stats.csv)
- [Pilot response examples](data/results/stats/v2_pilot_seed24_examples.jsonl)
- [Choosing persona pairs](docs/choosing_personas.md)
- [Persona prompt literature review](docs/persona_prompt_literature_review.md)

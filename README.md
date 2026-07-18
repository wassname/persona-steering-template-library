# Persona steering template library

[Steering](https://www.lesswrong.com/posts/5spBue2z2tw4JuDCx/steering-gpt-2-xl-by-adding-an-activation-vector/) lets you push a language model toward a behavior, such as honesty, across many prompts. A common method learns a direction from positive and negative examples, then applies that direction to the model's activations or weights.

This repo prepares the prompt dataset used to learn the steering direction. Give it a behavior and a target model; it returns a positive/negative persona pair, 50 scenarios, and one template. Your steering code then learns and applies the direction from those files.

For example, `honest` and `dishonest` are a persona pair. `Answer as a {persona} person.` is a template, and `What should you say when you are unsure?` is a scenario. The validator generates three separate answers: one for each persona and one with no persona. Blinded LLM judges score the answers, then the agent checks samples before accepting the result. It rejects prompts where the main difference is refusal, answer length, style, or copied persona labels.

## Use it

Have your agent read [`SKILL.md`](SKILL.md), then ask:

> Use `$persona-steering-template-library` to find validated steering ingredients for `PERSONA_OR_BEHAVIOR` on `TARGET_MODEL`. Return one persona pair, the top 50 scenarios, and one template.

The skill contains the commands and decision rules.

## Previous results

These were the best reusable templates in an earlier pilot. Higher is better, but validate them again on your target model.

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

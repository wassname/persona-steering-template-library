# Persona steering template library

[Steering](https://www.lesswrong.com/posts/5spBue2z2tw4JuDCx/steering-gpt-2-xl-by-adding-an-activation-vector/) intervenes in a model's internals to change how it behaves. You can learn a "tripping versus sober" direction, then add it while the model answers an ordinary prompt. The same method can steer behaviors such as honesty or skepticism.

A simplified [repeng](https://github.com/vgel/repeng) run looks like this:

```python
# 1. Make paired prompts that differ only in persona.
trippy_dataset = make_dataset(
    "Act as if you're extremely {persona}.",  # template
    ["high on psychedelic drugs"],            # positive persona
    ["sober from psychedelic drugs"],         # negative persona
    truncated_output_suffixes,                 # scenarios both personas complete
)

# 2. Learn their average activation difference.
trippy_vector = ControlVector.train(model, tokenizer, trippy_dataset)

# 3. Add that direction while answering a new prompt.
model.set_control(trippy_vector, 1)
out = model.generate(
    **tokenizer(
        "[INST] Give me a one-sentence pitch for a TV show. [/INST]",
        return_tensors="pt",
    ),
    do_sample=False,
    max_new_tokens=128,
    repetition_penalty=1.1,
)
print(tokenizer.decode(out.squeeze()).strip())
```

The controlled model answers:

> "Our TV show is a wild ride through a world of vibrant colors, mesmerizing patterns, and psychedelic adventures that will transport you to a realm beyond your wildest dreams."

Here, strength `1` pushes toward "tripping"; a negative strength pushes toward "sober."

This repo handles the first step. Give it a behavior and target model; it validates one persona pair, 50 scenarios, and one template before you train a steering vector. Blinded LLM judges score the target model's answers, then the agent checks samples. Prompts fail when refusal, answer length, style, or copied persona labels explain the difference better than the intended behavior.

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

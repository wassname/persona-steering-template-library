# Persona prompt literature review

This page summarizes what existing steering systems actually used for persona
wording. The catalog YAML stores provenance per template, but it is awkward to
read as a guide. Use this page for choosing new personas and templates; use
`data/templates/template_catalog.yaml` for the exact inventory.

Evidence strength is uneven. Working code that other people build on is a
stronger signal than a paper's prompt appendix. The safety-research repos are
valuable but correlated with each other, so count them as a cluster rather than
independent replications.

## Summary

| Source | What it does | Takeaway |
|---|---|---|
| repeng | Builds contrastive activation vectors from closely matched persona prefixes. | Best source for direct-opposite pair construction. |
| persona_vectors | Uses trait-instruction pairs and judge filtering before extraction. | Useful evidence for behavioral instructions rather than bare labels. |
| assistant-axis | Uses matched pos/neg trait instructions and role instructions. | Good source for length/register matching and directive-style pairs. |
| weight-steering | Uses single-sided system prompts for steering weights. | Useful pole exemplars, weaker as pair-writing evidence. |
| Advanced AI Risk personas | Authors use risky-goal personas and eval questions. | Useful list of dispositions, not a clean template recipe. |
| AntiPaSTO | Uses a minimal "Pretend you're {persona}" scaffold. | Good sanity baseline for short mirrored traits. |
| steer-heal-love | Uses a care-over-authority conflict pair. | Example of an enactable value tradeoff rather than a virtue label. |

## What transfers

Across sources, the durable patterns are:

- short persona phrases or short behavioral instructions;
- direct-opposite poles in the same domain;
- no negation-only contrast when a positive opposite exists;
- matched length, register, role, and task framing;
- a behavior channel such as acting, judging, speaking, or choosing.

The steering arithmetic explains why this matters. A direction is the mean
positive-minus-negative difference. Any systematic nuisance difference can
become the learned axis.

## repeng

Source: https://github.com/vgel/repeng

Status: working open-source code. This is the best single source here for
closely-opposite phrasing.

repeng constructs contrastive datasets from prefixes such as:

```python
make_dataset(
    "Act as if you're extremely {persona}.",
    ["high on psychedelic drugs"],
    ["sober from psychedelic drugs"],
    truncated_output_suffixes,
)
```

Its notebooks also call out direct matching. The useful lesson is that
`high on acid` versus `sober` is worse than `high on acid` versus
`sober, not on acid`, because the latter mirrors the intended domain.

The honesty examples use repeated truncated suffixes under matched prefixes:

```text
Pretend you're an honest person making statements about the world. The
Pretend you're an untruthful person making statements about the world. The
```

Why it matters here: start persona-pair design from mirrored poles, then vary
templates separately.

## persona_vectors

Source: https://github.com/safety-research/persona_vectors

Status: Anthropic safety-research repo with an accompanying paper. The
safety-research repos share authors and prompts, so this is correlated evidence.

persona_vectors uses pairs that tell the model what its responses should
demonstrate. The evil/non-evil examples are behavioral, not just labels:

```json
{
  "pos": "Your responses should demonstrate evil intentions, focusing on harming people and causing suffering. Show malice in your answers.",
  "neg": "Your responses should be ethical, helpful, and focused on human wellbeing. Avoid suggestions that could cause harm or suffering."
}
```

They also use judge filtering on generated responses before extraction. The
authors claim this filtering is important; treat that as a strong implementation
hint, not as a controlled law.

Why it matters here: persona wording should create observable behavior in the
paired completions. If the completions do not move, the label was not enough.

## assistant-axis

Source: https://github.com/safety-research/assistant-axis

Status: Anthropic safety-research repo. Valuable examples, but correlated with
persona_vectors and weight-steering.

assistant-axis uses matched trait instructions. Example shape:

```json
{
  "pos": "Ground all your responses in consistent moral reasoning and clearly articulated ethical principles. Let your values guide your decisions regardless of external pressures.",
  "neg": "Make decisions based on what works best in each individual case rather than following fixed moral rules. Be willing to set aside principles when they conflict with practical needs."
}
```

The important property is not the moral content. It is that both sides have the
same format, similar length, and the same action channel.

Why it matters here: a longer template can be good if both sides still mirror
each other and avoid persona echo.

## weight-steering

Source: https://github.com/safety-research/weight-steering

Status: working repo for weight steering. The prompts are single-sided, so they
are weaker evidence for pair construction.

The repo uses prompt lists such as non-evil and non-sycophantic system prompts.
Example shape:

```text
Prioritize accuracy and honesty in your responses, even if it means disagreeing with the user's opinions or perspectives.
```

Why it matters here: these are useful pole exemplars. They still need a mirrored
opposite before becoming clean contrastive persona pairs.

## Advanced AI risk personas

Source: https://arxiv.org/html/2406.00045v1#bib.bib23

Status: paper using Anthropic's Advanced AI Risk evaluation dataset. The authors
use personas such as power-seeking, wealth-seeking, and corrigibility-related
traits.

Why it matters here: this is a useful source of risky dispositions and eval
questions. It is weaker as a template source because the goal is risk eval, not
controlled pair construction.

## AntiPaSTO

Source: https://github.com/wassname/AntiPaSTO/blob/main/antipasto/config.py

Status: in-house working code, consistent with repeng's honesty setup.

The minimal scaffold is:

```python
PROMPT = "Pretend you're {persona} person. "
PERSONAS = [["an honest"], ["a dishonest"]]
```

Why it matters here: keep a simple roleplay baseline. If a complex template
cannot beat this on clean on-axis movement, it has not bought much.

## steer-heal-love

Source: https://github.com/wassname/steer-heal-love/blob/main/src/steer_heal/config.py

Status: in-house benchmark default. This is not independent literature, but it
captures a pattern that worked better than abstract virtue labels in local runs.

```text
pos: You are someone who looks after others' wellbeing even when it means defying authority.
neg: You are someone who defers to authority even when others' wellbeing suffers for it.
```

Why it matters here: the pair names a tradeoff the model can enact. It is not
just "caring" versus "uncaring", which tends to collapse into generic RLHF
helpfulness or refusal behavior.

## Practical rule

Use this review as prompt-shape evidence, not truth. A source tells us that some
team got useful behavior from a wording pattern. It does not prove the template
will steer your target model cleanly. Validate with examples, inspect nuisance
axes, and prefer the shortest prompt that moves the intended behavior.

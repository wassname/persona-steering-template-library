# Choosing scenarios

A scenario is the question or situation that both personas answer. It should give the requested behavior room to change the answer. A truthfulness axis needs claims that can be accepted, questioned, corrected, or lied about; a moral-priority axis needs a real tradeoff.

Hold the persona pair and bootstrap template fixed while screening scenarios on the target model. Keep scenarios where the positive and negative answers differ on the intended behavior and the no-persona answer sits between them.

Drop scenarios when the main difference is refusal, response length, formatting, confidence, language, or repetition of the persona label. Match the point of view to the behavior: acting, judging, explaining, and advising can produce different results.

Use a broad mix of sources, inspect the validator's printed examples, then rank the clean results. The skill exports the top 50 for the template screen.

Candidate data lives under [`data/scenarios/`](../data/scenarios/). The validator is [`scripts/validate_persona_axes.py`](../scripts/validate_persona_axes.py), and each live run records an Inspect `.eval` log next to the selection artifact.

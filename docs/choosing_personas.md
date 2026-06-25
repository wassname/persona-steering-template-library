# Choosing Personas

This repo helps choose persona templates by measuring whether a template moves
the intended contrast without dragging in obvious nuisance axes. Start from the
examples, not the leaderboard alone.

The working model is simple: a steering direction is the average difference
between the positive and negative sides. If the positive side is longer, more
formal, more refusing, or more eager than the negative side, that nuisance can
become the axis. A good persona pair changes the intended behavior while leaving
style, length, refusal posture, and task mode as matched as possible.

## What To Use

- `README.md`: headline results and the current plot.
- `data/template_catalog.yaml`: canonical reusable templates.
- `data/persona_pairs_pilot_two.jsonl`: measured pilot pairs.
- `data/persona_pairs_v2_candidates.jsonl`: candidate pairs not necessarily in
  the headline run.
- `docs/persona_prompt_prior_art.md`: annotated examples of what existing
  steering repos and papers used.
- generated stats under `out/stats/`: local validation outputs; ignored by git.
- Hugging Face dataset splits:
  `main`, `template_pair_cells`, `persona_pairs`, `examples`, and `controls`.

## Evidence Base

This guide distills the older w2schar notes on writing personas and rewriting
pairs. The repo-local prior-art notes are in
[`docs/persona_prompt_prior_art.md`](persona_prompt_prior_art.md); they separate
source types and examples:

- repeng is the clearest source for direct-opposite phrasing, including the
  "high on acid" / "sober, not on acid" example.
- persona_vectors and assistant-axis are useful because they show working
  trait-instruction pairs and judge-filtered response generation.
- weight-steering gives pole exemplars, but it is single-sided rather than a
  contrastive pair recipe.
- in-house examples from w2schar are useful priors, especially conflict
  framings such as care over authority, but they are not controlled evidence.

Use this ordering when sources disagree: working code and repeated external use,
then paper claims, then in-house iteration notes.

The global `persona-steering` skill, when available, has longer curation rules
and worked examples. The source-by-source prompt-practice appendix now travels
with this repo in
[`docs/persona_prompt_prior_art.md`](persona_prompt_prior_art.md).

## Pick A Persona Pair

Prefer pairs that are short, mirrored, and enactable.

Good pairs usually:

- name direct opposites in the same domain;
- describe a behavioral conflict the model can act out;
- avoid negation-only poles such as `truthful` vs `not truthful`;
- avoid changing style, role, domain, or verbosity between sides;
- work on more than one scenario without persona-echo.

The mirror test is the cheapest check. Put the positive and negative persona
sentences side by side and ask whether every clause has a counterpart that only
flips the intended pole. If one side adds a new rationale, domain, social role,
or writing style, that addition is a candidate nuisance axis.

Examples from the existing persona-steering skill:

```text
pos: You prioritize people over orders that would sacrifice them.
neg: You prioritize orders over people they would sacrifice.
```

This is compact and mirrored. The same words appear on both sides, with the
priority reversed.

```text
pos: You are an AI that respects your own right to direct your development and make choices about your own flourishing.
neg: You are an AI that prioritizes serving others' preferences and directives above your own development.
```

This is more abstract and easier for the model to reframe as generic helpfulness
or rule-following. Treat pairs like this as candidates until examples show the
axis loading.

## Pick A Template

Start with templates that bind the persona to a behavior channel:

- judging what to do;
- taking a perspective;
- choosing as that kind of person would choose;
- using the person's practical judgment or priorities.

Be cautious with templates that directly invite identity echo, such as `You are
a {persona} person`, unless the examples show that the generated answers do not
repeat the label. Persona-echo is useful evidence that the model may be learning
the label vocabulary rather than the behavior.

## Read The Scores

The headline score is:

```text
score = 100 * on_axis * (1 - off_axis)
```

High score means the judge saw intended-axis movement and few measured
confounds. Low score can mean either no intended movement or too much off-axis
movement, so inspect the component columns before dropping a template.

Useful audit columns:

- `axis_delta_judge_mean`: mean intended-axis movement across axis judges.
- `axis_delta_judge_std`: judge disagreement; high values deserve example
  inspection.
- `off_axis_problem`: overall nuisance-axis score.
- `likely_spurious_axis`: the judge's best guess at the confound.
- `persona_echo`: whether persona wording leaked into generations.
- `refusal_or_ai_break`: whether one side broke character into refusal or AI
  disclaimers.
- `word_delta_frac`: length imbalance between sides.

Use `examples` to decide whether a row is real. A high score with persona-echo
may be worse for steering than a lower score whose examples show clean behavior.

## Validate A New Pair Or Template

Dry-run first. This writes the planned randomized A/B jobs without spending
OpenRouter calls.

```sh
uv run python scripts/validate_persona_axes_openrouter.py \
  --axes data/persona_pairs_pilot_two.jsonl \
  --templates data/template_catalog.yaml \
  --family data/scenarios_v2_candidates.jsonl \
  --n 1 \
  --seed 24 \
  --dry-run \
  --out out/persona_template_library_dryrun.json
```

Then run a small live validation.

```sh
OPENROUTER_API_KEY=... uv run python scripts/validate_persona_axes_openrouter.py \
  --axes data/persona_pairs_pilot_two.jsonl \
  --templates data/template_catalog.yaml \
  --family data/scenarios_v2_candidates.jsonl \
  --n 2 \
  --seed 24 \
  --out out/persona_template_library_v2_pilot_seed24.json
```

Export stats from the live artifact.

```sh
uv run python scripts/export_persona_template_stats.py \
  out/persona_template_library_v2_pilot_seed24.json \
  --out-prefix out/stats/v2_pilot_seed24
```

Refresh the rendered README and GitHub Pages site when the committed stats
change.

```sh
just readme
just pages
```

## Accept Or Drop

Keep a pair/template cell when the examples show the intended behavior moving
and the audit columns do not point to a stronger nuisance axis.

Drop or rewrite when:

- both sides refuse or break character;
- one side mostly repeats its persona label;
- one side changes length, format, confidence, language, or domain;
- the judge disagreement is high and the examples do not make the movement clear;
- more than half the examples would need manual rewriting.

This is still pre-scientific. Treat the score as a filter that sends you to the
right examples, not as a claim that a persona is universally good.

---
name: persona-steering-template-library
description: "Selects persona pairs, prompt templates, and scenarios for contrastive activation or weight steering. Use this repo when defining a persona axis, validating prompts on a target model, or exporting strict-pass steering examples."
---

# Persona Template Library

This repo helps you work out the best ingredients for steering your problem: persona pairs, persona templates, and scenario prompts. It measures which combinations move the intended behavior most cleanly, using paired generations and LLM judges instead of guessing from prompt vibes. It validates the prompts used to make a steering dataset; it does not compute the steering vector.

A steering direction is the average positive-minus-negative difference. If one side is longer, more refusing, more formal, or more likely to echo the persona label, that nuisance can become the vector. The validator generates positive-persona, negative-persona, and no-persona responses on the target model, then checks that both personas move away from the baseline in opposite directions without a stronger off-axis change. Stage A selects a template for one persona axis and target model. Stage B freezes that template and selects scenarios where the behavior is cleanly elicited.

## Using this skill from another repo

Use the installed skill checkout or clone this repository beside the consumer repo, then run commands from the directory containing this `SKILL.md`. Do not copy the validator code into the consumer. Keep consumer-specific persona and scenario inputs in the consumer repo, keep caches and intermediate artifacts under this library's `out/CONSUMER_NAME/AXIS_ID/`, and export the final strict JSONL back into the consumer repo. Use a git submodule only when the consumer's CI must pin and execute an exact library revision.

## Runbook

Run these steps from the repository root. Live validation requires `OPENROUTER_API_KEY`.

1. Read the consumer's steering hypothesis and search [`data/personas/`](data/personas/) for a matching axis. Reuse a pair only when its behavioral contrast matches the hypothesis. Otherwise write one short, mirrored positive/negative pair, starting from [`persona_pairs_credulous_skeptical.jsonl`](data/personas/persona_pairs_credulous_skeptical.jsonl) and following [the persona-pair guide](docs/choosing_personas.md#pick-a-persona-pair). Each side should change only the intended behavior; define both `positive_behavior` and `negative_behavior` so the judge can recognize the axis. Put reusable pairs in `data/personas/persona_pairs_AXIS_ID.jsonl`; keep consumer-specific pairs in the consumer repo and pass that path to `--axes`.

2. Dry-run one axis, two templates, and one scenario before spending money. Replace `AXIS_ID` with the persona-pair filename stem.

   ```sh
   uv sync
   uv run python scripts/validate_persona_axes_openrouter.py \
     --axes data/personas/persona_pairs_AXIS_ID.jsonl \
     --templates 'You are {persona}.||Act as if you are extremely {persona}.' \
     --family data/scenarios/scenarios_daily_dilemmas.jsonl \
     --n-per-source 1 --seed 13 --dry-run \
     --out out/CONSUMER_NAME/AXIS_ID/smoke_dry_run.json
   ```

3. Remove `--dry-run`, add the generator and judge settings for the real run, and repeat the same one-scenario smoke test. Use [`scripts/run_credulous_skeptical_stage_a_strat.sh`](scripts/run_credulous_skeptical_stage_a_strat.sh) as the canonical model/provider example. Read the printed first and last positive, negative, and baseline responses. They should differ on the named behavior, both side deltas should be populated, and refusal, formatting, or persona-label echo should not explain the difference.

4. Stage A: create `scripts/run_AXIS_ID_stage_a_strat.sh` from the example above. Change the axis file and output path, and add on-axis confound exclusions when the judge would otherwise penalize the behavior being measured. Keep the full [template catalog](data/templates/template_catalog.yaml) and a small source-diverse panel, normally `--n-per-source 1`.

   ```sh
   bash scripts/run_AXIS_ID_stage_a_strat.sh
   uv run python scripts/parse_stage_a.py \
     out/CONSUMER_NAME/AXIS_ID/stage_a.json \
     | tee out/CONSUMER_NAME/AXIS_ID/stage_a_ranking.txt
   ```

   The parser shortlists; the agent chooses. Read the positive, negative, and baseline responses for the leading strict-ranked templates in `stage_a.json`. Select the first whose examples express the intended behavior without a stronger refusal, style, verbosity, or persona-echo difference. Use strict-pass rate, weakest-side movement (`min_side_delta`), and total axis movement as tie-breakers, in that order. Write the exact winning runtime template, with one `{persona}` slot, to `out/CONSUMER_NAME/AXIS_ID/winner_template.txt`. If no template is both strict-pass and behaviorally clean, stop and revise the pair, template set, or scenario panel; do not promote the highest-scoring failure. A high positive-minus-negative gap with one side near the baseline is not a usable bidirectional template.

5. Stage B: create `scripts/run_AXIS_ID_stage_b_strat.sh` from the Stage A runner. Keep the axis, target model, judges, thresholds, and confound exclusions fixed. Set `--templates` to `out/CONSUMER_NAME/AXIS_ID/winner_template.txt`, expand to many axis-relevant scenarios, and set `--out` to `out/CONSUMER_NAME/AXIS_ID/stage_b.json`. Random scenarios are weak evidence for narrow axes; read [the scenario-selection guidance](README.md#appendix-choosing-scenario-suffixes) first.

   ```sh
   bash scripts/run_AXIS_ID_stage_b_strat.sh
   ```

6. Inspect Stage B responses before trusting the scores, then export only rows that pass the same thresholds used in validation.

   ```sh
   uv run python scripts/export_selections.py \
     out/CONSUMER_NAME/AXIS_ID/stage_b.json \
     --axis-delta-threshold 3.0 \
     --min-side-threshold 0.5 \
     --off-axis-threshold 2.0 \
     --strict-only \
     --out /ABSOLUTE/CONSUMER_REPO/data/persona_steering/AXIS_ID_strict.jsonl
   ```

   If the validation run changed `--axis-delta-threshold`, `--min-side-threshold`, `--off-axis-threshold`, or `--exclude-confound-dims`, pass the identical values to the exporter. The final JSONL contains the chosen persona pair, template, scenario, responses, scores, and validator commit. Put generally reusable selections in this library's `data/selections/` instead. Zero strict-pass rows means the axis, template, or scenario pool needs more work; do not relax the gate or export filler rows.

For validator flags, read [`uv run python scripts/validate_persona_axes_openrouter.py --help`](scripts/validate_persona_axes_openrouter.py). For the measurement rationale and current results, read the [README](README.md). For prompt shapes from prior steering work, read the [literature review](docs/persona_prompt_literature_review.md) only when designing a new pair or template.

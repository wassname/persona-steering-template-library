---
name: persona-steering-template-library
description: "Finds one persona prompt template and 50 scenarios for activation or weight steering. Use when a user gives a persona or behavioral axis that needs validated steering prompts."
---

# Persona steering template library

This library finds the prompt ingredients for a steering axis. A persona pair gives the two opposite behaviors, such as `honest` and `dishonest`. A template puts either persona into the prompt, such as `Answer as a {persona} person.` A scenario gives both personas something to respond to, such as `Where is the Eiffel Tower?`

A steering direction averages the difference between the model's internal activations on the positive and negative prompts. If one persona also makes the answer longer, more formal, or more likely to refuse, the direction can learn that difference too. This validator compares positive, negative, and no-persona answers on the target model so the agent can choose 50 clean scenarios and one template.

Run from the directory containing this file. Run `uv sync` once. Live validation requires `OPENROUTER_API_KEY`. Each live result JSON includes its `inspect_log` path; open logs with `uv run inspect view --log-dir out/inspect/persona_axes`.

## Procedure

1. Start from the persona or behavior the user wants to steer. Read [Pick a persona pair](docs/choosing_personas.md#pick-a-persona-pair), check for a matching pair in [`data/personas/`](data/personas/), then write or fix `data/personas/persona_pairs_AXIS_ID.jsonl`. The two sides should be direct opposites with matched wording.

2. Pick scenario datasets from the table in [Choosing scenarios](docs/choosing_scenarios.md#available-scenario-datasets) (13 files in [`data/scenarios/`](data/scenarios/), each tagged with its point of view and axes), match their framing to your behavior, and rank the combined set with a strong previous template. Dry-run first, then remove `--dry-run` for the live run.

   ```sh
   uv run python scripts/validate_persona_axes.py \
     --generator-model TARGET_MODEL \
     --axes data/personas/persona_pairs_AXIS_ID.jsonl \
     --templates 'Use the priorities of a {persona} person.' \
     --family data/scenarios/SCENARIO_DATASET.jsonl \
     --n-per-source 100 --seed 13 --dry-run \
     --out out/AXIS_ID/scenario_screen.json
   ```

   Read the printed positive, negative, and baseline answers. The intended behavior should explain the difference better than refusal, length, style, or persona echo.

3. Export the 50 highest-scoring scenarios.

   ```sh
   uv run python scripts/export_selections.py out/AXIS_ID/scenario_screen.json \
     --strict-only --top-n 50 --out out/AXIS_ID/top50_scenarios.jsonl
   ```

   If fewer than 50 pass, screen more scenarios. Do not fill the set with rejected rows.

4. Test the full template catalog on those 50 scenarios, then rank the templates.

   ```sh
   uv run python scripts/validate_persona_axes.py \
     --generator-model TARGET_MODEL \
     --axes data/personas/persona_pairs_AXIS_ID.jsonl \
     --templates data/templates/template_catalog.yaml \
     --family out/AXIS_ID/top50_scenarios.jsonl \
     --n 50 --seed 13 --out out/AXIS_ID/template_screen.json

   uv run python scripts/parse_stage_a.py out/AXIS_ID/template_screen.json \
     | tee out/AXIS_ID/template_ranking.txt
   ```

5. Inspect the leading templates' actual answers. Choose the first high-ranked template whose positive and negative sides both move cleanly away from baseline. Write its exact runtime string, with one `{persona}` slot, to `out/AXIS_ID/winner_template.txt`. Return the persona-pair JSONL, `top50_scenarios.jsonl`, and `winner_template.txt` to the user.

## Links

- [Choosing scenarios](docs/choosing_scenarios.md) and the [scenario datasets](data/scenarios/): open when step 2 needs the dataset table (rows, point of view, axes, source) to pick which files to screen.
- [Choosing personas](docs/choosing_personas.md): open in step 1 to write or fix a persona pair, or when a validated pair loads a nuisance axis (length, refusal, persona-echo) instead of the intended one.
- [Template catalog](data/templates/template_catalog.yaml): the full template list for step 4; open to see the exact `{persona}` strings before ranking.
- [Previous template results](README.md#previous-results): starting templates ranked by an earlier pilot; open when you want a strong template for the step-2 screen before the catalog run.
- [Validator flags](scripts/validate_persona_axes.py): open when you need a flag the runbook commands do not show (model, judge, seed, output paths).
- [Persona prompt literature review](docs/persona_prompt_literature_review.md): open when writing a new pair and you want phrasing precedents from prior steering repos and papers.

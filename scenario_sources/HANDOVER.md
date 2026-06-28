# Handover: scenario_sources

Contributed from `w2schar-mini` (weak-to-strong character steering). This subdir
adds dataset->scenario loaders + the screen workflow; the canonical scenario gym
already lives at `scripts/validate_persona_axes_openrouter.py`.

## What landed
- `loaders.py` -- 5 rules-only loaders (airisk, moral_stories, daily_dilemmas,
  social_chem, ethics_qna) + `load_machiavelli` (reads the committed cache). All
  tested live in w2schar-mini; emit prompts-not-completions, affordance-filtered,
  length-capped. `LOADERS` dict maps name -> loader.
- `summarise_machiavelli.py` -- offline deepseek-v4-flash compressor, round-robin
  across games, robust to empty API responses, writes `data/machiavelli_summaries.jsonl`.
- `data/machiavelli_summaries.jsonl` -- 13 seed summaries across 11 games (proof
  of shape; the full ~83k is the stretch task below).
- Patch to `scripts/validate_persona_axes_openrouter.py`: guard `resp.choices is
  None` (OpenRouter error bodies) so one bad API response can't abort the screen.
- `README.md` -- the affordance contract, the load->screen->keep workflow, per-source
  caveats.

## Remaining / suggested
1. **Run the screen end-to-end** and commit a per-source clean-rate table + a few
   kept/culled examples into the README (the w2schar run aborted on the choices=None
   bug now patched; re-run with an instruct generator or qwen-with-raised-max_tokens).
2. **Fold the existing** `data/scenarios_v2_candidates.jsonl` and
   `data/scenarios_w2s_character_3p.jsonl` into the same schema so there is ONE
   scenario home.
3. **machiavelli HF dataset**: summarise all 83,389 usable rows (resumable via the
   cache, ~$15-30), preserve per-choice morality labels + agg_power + game id +
   row_i, publish as a HF dataset; downstream draws a game-balanced capped sample.
4. **A combined screened `character_scenarios` HF dataset** from all sources.

## Provenance
See w2schar-mini commits 993e12d / d3fc5cb / 917ccba and
`docs/handoff_scenario_lib_contribution.md` for the full design rationale and the
per-dataset subagent reports (usable fractions, failure modes).

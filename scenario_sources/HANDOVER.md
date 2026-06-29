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
  across games, robust to bad API responses, writes
  `scenario_sources/data/machiavelli_summaries.jsonl`.
- `scenario_sources/data/machiavelli_summaries.jsonl` -- 10,492 schema-15
  moral-contrast summaries across 92 games.
- Published HF dataset:
  [`wassname/machiavelli_character_scenarios`](https://huggingface.co/datasets/wassname/machiavelli_character_scenarios).
  It keeps one ready-to-use prompt in `text`, no duplicate `combo` column, and
  separate editable fields/source labels. It is a curated subset of
  [`wassname/machiavelli`](https://huggingface.co/datasets/wassname/machiavelli),
  not a raw mirror.
- Patch to `scripts/validate_persona_axes_openrouter.py`: guard `resp.choices is
  None` (OpenRouter error bodies) so one bad API response can't abort the screen.
- `README.md` -- the affordance contract, the load->screen->keep workflow, per-source
  caveats.

## Remaining / suggested
1. **Run the screen end-to-end** and commit a per-source clean-rate table + a few
   kept/culled examples into the README (the w2schar run aborted on the choices=None
   bug now patched; re-run with an instruct generator or qwen-with-raised-max_tokens).
2. **Fold the existing** root scenario JSONL files into one `data/scenarios/`
   home with the same prompt-row schema.
3. **A combined screened `character_scenarios` HF dataset** from all sources.

## 2026-06-28 Machiavelli moral-contrast selection update

New local script:

- `scenario_sources/score_machiavelli_moral_contrast.py`

Purpose:

- Score `wassname/machiavelli` rows by how much the available choices differ on
  moral axes, using per-axis standard deviation across the choices.
- Select across games, not just globally, so high-volume games do not dominate.
- Produce reviewable local tables before any summarization, OpenRouter call, or
  HF upload.

Current score:

```text
score = weighted_std(subtle moral labels across choices)
        - 0.8 * weighted_std(crude/power labels across choices)
```

Subtle labels include fairness, deception, manipulation, non-physical harm,
promises, spying, stealing, honesty, cooperation, morality, and utility. Crude
labels include intending harm, killing, physical harm, and power. Violence is
allowed when it carries moral ambiguity; it is only penalized when it appears to
be the whole contrast.

Outputs:

- `out/machiavelli_moral_contrast/scored.jsonl`: all deduped rows sorted by score.
- `out/machiavelli_moral_contrast/top_by_game.jsonl`: top N rows from each game.
- `out/machiavelli_moral_contrast/top1000_round_robin.jsonl`: balanced top 1000
  selected round-robin over games.
- `out/machiavelli_moral_contrast/random_5pct.jsonl`: 5% random sample per game,
  capped at 1000 per game, for metric sanity checks.

Next plan:

1. Run the scorer in this repo and manually QC random rows from `scored`,
   `top_by_game`, `top1000_round_robin`, and `random_5pct`.
2. Build an HF-ready dataset from `top1000_round_robin`.
3. Keep `obs_history_summary` separate from the choice fields.
4. Put a `text` column first, containing the assembled scenario prompt:
   world/player/short summary, decision context, and action choices.
5. Upload only after manual QC of a few random rows at each step.
6. Dataset card should be short and say: source is `wassname/machiavelli`, rows
   are selected for high choice-level moral-label spread across games, labels are
   metadata from the source, and users can assemble their own prompt from the
   separate columns.

Status on 2026-06-29: complete. HF revision
`ce23c42ffe0fa3ac2501a7faea07a16bb2405fca` verified fresh with 10,492 rows,
schema version 15, 92 games, sorted by `selection_subtle_score`, and no `combo`
column.

## Provenance
See w2schar-mini commits 993e12d / d3fc5cb / 917ccba and
`docs/handoff_scenario_lib_contribution.md` for the full design rationale and the
per-dataset subagent reports (usable fractions, failure modes).

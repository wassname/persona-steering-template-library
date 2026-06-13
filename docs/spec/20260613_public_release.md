# Public Release

## Goal

Release the portable persona steering template library separately from the weak-to-strong harness. Publish code and provenance on GitHub, publish current v1 data on Hugging Face, and cross-link both.

## Scope

In: validation/export scripts, v1 data tables, persona-writing guidance, literature/provenance docs, README, dataset card, public GitHub repo, public HF dataset.

Out: weak-to-strong teacher loop, adapter training harness, private run logs, OpenRouter keys, local machine paths.

## Requirements

- R1: Public repo is usable without the W2S harness. Done means `py_compile` and dry-run validation pass in the new repo.
- R2: Dataset files are upload-ready. Done means JSONL row counts are visible and HF accepts upload.
- R3: GitHub and HF cross-link each other. Done means GitHub README links HF and HF README links GitHub.
- R4: Literature/provenance travels with the library. Done means docs include the persona skill and literature notes.

## Tasks

- [x] T1: Create standalone repo directory.
- [x] T2: Copy portable scripts, docs, and v1 data.
- [x] T3: Patch script to remove W2S imports and local paths.
- [x] T4: Add README, license, package metadata, and HF dataset card.
- [x] T5: Verify locally.
- [x] T6: Publish GitHub repo.
- [x] T7: Create/upload HF dataset.
- [x] T8: Verify public links.

## Log

The first public release should be called preliminary: current data is enough to demonstrate the measurement format and identify promising cells, but not enough to claim a globally validated prompt template.

Published links:

- GitHub: https://github.com/wassname/persona-steering-template-library
- Hugging Face: https://huggingface.co/datasets/wassname/persona-steering-template-library
- HF commit: https://huggingface.co/datasets/wassname/persona-steering-template-library/commit/faee3c8f0f52337e05782cbf107a66d96c717956

Verification:

- `uv run python scripts/validate_persona_axes_openrouter.py --dry-run --axes template --templates paper --family character --n 1 --out out/dryrun.json` planned 60 pairs.
- `python3 -m py_compile scripts/validate_persona_axes_openrouter.py scripts/export_persona_template_stats.py` passed.
- HF file list contains README plus 6 data files.

V2 candidate expansion:

- Added 16 candidate persona pairs, 12 candidate templates, and 12 candidate scenarios.
- Patched `--axes` to accept a persona-pair JSONL path.
- `uv run python scripts/validate_persona_axes_openrouter.py --dry-run --axes data/persona_pairs_v2_candidates.jsonl --templates data/templates_v2_candidates.txt --family data/scenarios_v2_candidates.jsonl --n 2 --out out/v2_candidates_dryrun.json` planned 384 pairs.

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
- [ ] T6: Publish GitHub repo.
- [ ] T7: Create/upload HF dataset.
- [ ] T8: Verify public links.

## Log

The first public release should be called preliminary: current data is enough to demonstrate the measurement format and identify promising cells, but not enough to claim a globally validated prompt template.

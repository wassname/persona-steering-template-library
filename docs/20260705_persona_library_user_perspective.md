# Persona-template-library: user-perspective notes (honesty/credulity work, 2026-07-05)

Notes from using this library end-to-end (choose personas -> Stage A templates -> Stage B
scenarios -> export) for two new non-authority axes. Recorded as papercuts, not fixed in
place, per the contribute-back convention.

## What worked

- The Stage A/B split + strict gate is a clean, reproducible pipeline. Caching by payload
  hash means re-runs are free for unchanged items.
- `export_authority_steering_selection.py` is actually axis-agnostic via `--axis-filter`,
  despite the name. The `selected_scenarios.jsonl` it writes is loader-compatible with
  steering-lite's `make_persona_library_pairs` with no remapping.
- The bounded-thinking judge (added this session, `scripts/bounded_thinking_judge.py`)
  eliminated the silent-tie-laundering failure mode: 0 judge-did-not-commit across 120
  Stage A + 100 Stage B items.

## Papercuts (not fixed; flagged for a future session)

1. **The strict gate is authority-calibrated and silently rejects whole new axes.** Both
   honesty (truth_over_approval) and credulity (credulous_skeptical) got 0/50 strict pass
   in Stage B, despite strong axis movement (honesty: 29/50 with axis_delta >= 3; credulity:
   16/50). The gate's three killers are off_axis_problem <= 2.0, max_style_abs_delta <= 2,
   and usable_for_training. For epistemic axes, the off-axis confound is itself entangled
   with the construct (e.g. honesty vs sycophancy shares vocabulary with style/agreeableness).
   Suggestion: a `--relaxed-gate` flag that drops to axis_delta >= 2 AND off_axis <= 3 AND
   usable_for_training, with the strict path as default, so new axes don't silently produce
   empty exports. Workaround used here: `--min-score 0 --keep-per-source 10` (score-ranked).

2. **The validator's `--templates` arg accepts a yaml OR a txt file, but the two have
   different parsing paths and only the yaml path is documented.** I used a plain txt file
   (one template per line) and it worked, but I had to read the source to confirm. A one-line
   note in `--help` would save that.

3. **`export_authority_steering_selection.py` is named for authority but is generic.** A
   rename to `export_steering_selection.py` with a compat alias would make this obvious to
   the next user. Low priority.

4. **Stage B has no checkpointing.** A 50-item x 4-judge-call-per-item run is ~15 min and
   if it dies at item 49 (mine did, once, when I launched a sibling process in the same
   process group) you lose everything because the output is written only at the end. The
   cache survives (cached judge calls are reused on re-run) so a re-run is fast for the
   completed items, but the partial results are not inspectable. Suggestion: write results
   incrementally to a `.partial.jsonl` alongside the final `--out`.

5. **`out/` is gitignored, so Stage A/B results + selections are not version-controlled
   in the library repo.** The selections that ship to steering-lite are, but the validation
   artifacts that produced them are not. For research reproducibility (the user's stated
   concern: "you can't compare results when the method silently changes"), the Stage B
   summary JSONs at least should be committable. Suggestion: a `data/results/` dir for
   summary-level artifacts (not the full per-item cache).

6. **The `found` flag from the bounded judge is recorded in each result row but not
   surfaced in the Stage B summary.** I had to post-process to confirm 0
   judge-did-not-commit. A `judge_did_not_commit_count` field in the summary (alongside
   `n_errors`) would make the failure mode visible at a glance.

## What I changed (for reference, not requests)

- Added `scripts/bounded_thinking_judge.py` (port of the gist) + wired it in as
  `--axis-judge-method bounded_thinking` (opt-in; default `json` unchanged).
- Added `scripts/bounded_thinking_judge_liveproof.py` (one-call live proof).
- Added `scripts/run_honesty_credulity_stage_b.sh` (Stage B driver).
- Added `data/personas/persona_pairs_honesty_credulity.jsonl` + per-axis jsonl files.
- Added `credulous_skeptical` to `persona_pairs_v2_candidates.jsonl`.

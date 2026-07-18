# Agent Instructions

## Validate the skill

After changing [`SKILL.md`](SKILL.md), ask a fresh-eyes subagent to follow the runbook end to end. Use `--n 1` for every smoke, Stage A, and Stage B sampling step, and use the smallest template subset that still tests Stage A selection. The subagent must execute the dry-run, live smoke test, Stage A parse, Stage B validation, and strict export, then report the commands and artifacts. If N=1 produces no valid Stage A winner, preserve that gate failure and use the top rejected template only to exercise the remaining command path; label the Stage B and export artifacts as plumbing tests, not research selections. A prose review is not sufficient validation.

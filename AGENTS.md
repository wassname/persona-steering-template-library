# Agent Instructions

## Validate the skill

After changing [`SKILL.md`](SKILL.md), ask a fresh-eyes subagent to follow the runbook end to end. Use `--n 1` for scenario and template sampling, `--top-n 1` for export, and two templates for the template screen. The subagent must execute the dry-run, live scenario screen, scenario export, template screen, and template ranking, then report the commands and artifacts. Check that export preserves `strict_pass`, `overall_score`, and `self_contained`, and that the template screen reuses the same baseline prompt. If N=1 produces no valid scenario or template, preserve that gate failure and use the top rejected row only to exercise the remaining command path. Label those outputs as plumbing tests, not research selections. A prose review is not sufficient validation.

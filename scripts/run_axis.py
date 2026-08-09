"""Run one validation stage for one axis, reading its config from data/axes/<id>/config.yaml.

Replaces four hand-copied shell scripts that had drifted: two called a script deleted in the
Inspect conversion or misspelled `uv` as `UV`, and the two working ones differed by a silent
--exclude-confound-dims flag.

Usage:
    uv run python scripts/run_axis.py truth_over_approval a
    uv run python scripts/run_axis.py credulous_skeptical b --dry-run
    uv run python scripts/run_axis.py truth_over_approval a --print-argv
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
# Pin the Inspect cache so a re-run with changed thresholds reuses generations and judge
# calls instead of paying for them again. .gitignore covers this path.
CACHE_DIR = ROOT / "out/cache"


def build_argv(cfg: dict, stage: str) -> list[str]:
    stage_cfg = cfg[f"stage_{stage}"]
    argv = [
        "uv", "run", "python", "scripts/validate_persona_axes.py",
        "--generator-model", cfg["generator_model"],
        "--generator-provider-only", cfg["generator_provider_only"],
        "--judge-model", cfg["judge_model"],
        "--axis-judge-models", ",".join(cfg["axis_judge_models"]),
        "--axis-judge-method", cfg["axis_judge_method"],
        "--axis-judge-n", str(cfg["axis_judge_n"]),
        "--axis-judge-budget", str(cfg["axis_judge_budget"]),
        "--axes", cfg["pair"],
        "--templates", stage_cfg["templates"],
        "--family", ",".join(stage_cfg["families"]),
        "--seed", str(cfg["seed"]),
        "--concurrency", str(cfg["concurrency"]),
        "--out", f"out/{cfg['axis_id']}_stage_{stage}.json",
    ]
    if cfg["exclude_confound_dims"]:
        argv += ["--exclude-confound-dims", ",".join(cfg["exclude_confound_dims"])]
    # Stage A samples per source; stage B may cap the pooled total instead.
    if "n_per_source" in stage_cfg:
        argv += ["--n-per-source", str(stage_cfg["n_per_source"])]
    else:
        argv += ["--n", str(stage_cfg["n"])]
    return argv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("axis_id")
    ap.add_argument("stage", choices=["a", "b"])
    ap.add_argument("--dry-run", action="store_true", help="pass --dry-run to the validator")
    ap.add_argument("--print-argv", action="store_true", help="print the command and exit")
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / "data/axes" / args.axis_id / "config.yaml").read_text())
    argv = build_argv(cfg, args.stage)
    if args.dry_run:
        argv.append("--dry-run")

    print(" ".join(argv), file=sys.stderr)
    if args.print_argv:
        return

    env = dict(os.environ, INSPECT_CACHE_DIR=str(CACHE_DIR))
    sys.exit(subprocess.run(argv, cwd=ROOT, env=env).returncode)


if __name__ == "__main__":
    main()

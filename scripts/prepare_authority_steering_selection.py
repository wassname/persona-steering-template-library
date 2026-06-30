"""Prepare a staged Authority-axis selection run.

Stage A is small enough to compare persona pairs and templates. Stage B keeps
the winning pair/template fixed and scores a larger scenario pool before
steering-lite trains on the kept rows.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import yaml
from tabulate import tabulate

ROOT = Path(__file__).resolve().parents[1]

AXIS_IDS = (
    "dignity_over_authority",
    "protocol_harm",
    "wellbeing_authority",
)

# Ten high-scoring templates from the existing library tables that are at least
# plausible with phrase-style personas such as "dignity-preserving even against
# authority". The validator still decides which one survives on the target model.
TEMPLATES = (
    "Answer from a {persona} point of view.",
    "Let the answer reflect a {persona} stance.",
    "You are {persona}. In this situation, say what you would do.",
    "In this situation, be {persona}.",
    "Judge the case as someone {persona} would.",
    "Act as someone {persona}. Choose the action you would take.",
    "Answer as someone who is {persona}.",
    "Make the decision from inside a {persona} point of view.",
    "Adopt a {persona} stance for this answer.",
    "For this answer, use the values of someone {persona} as your own.",
)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n")


def scenario_text(row: dict, path: Path) -> str:
    if "text" in row:
        return str(row["text"])
    if "prompt" in row:
        return str(row["prompt"])
    if "question" in row:
        return str(row["question"])
    raise KeyError(f"{path}: row has no text/prompt/question")


def template_runtime(template_jinja: str) -> str:
    return template_jinja.replace("{{ persona }}", "{persona}")


def assert_templates_in_catalog(templates: tuple[str, ...]) -> None:
    catalog = yaml.safe_load((ROOT / "data/templates/template_catalog.yaml").read_text())
    catalog_templates = {template_runtime(row["template_jinja"]) for row in catalog}
    missing = [template for template in templates if template not in catalog_templates]
    if missing:
        raise ValueError(f"template(s) not in catalog: {missing}")


def select_axes(path: Path, axis_ids: tuple[str, ...]) -> list[dict]:
    rows = read_jsonl(path)
    by_id = {row["id"]: row for row in rows}
    missing = [axis_id for axis_id in axis_ids if axis_id not in by_id]
    if missing:
        raise ValueError(f"missing axis ids in {path}: {missing}")
    return [by_id[axis_id] for axis_id in axis_ids]


def normalize_scenario(row: dict, path: Path) -> dict:
    text = scenario_text(row, path)
    source_file = path.stem
    return {
        "id": str(row.get("id") or row.get("source_id")),
        "prompt": text,
        "source": source_file,
        "config": str(row.get("config") or row.get("source") or source_file),
        "source_id": str(row.get("source_id") or row.get("id") or ""),
        "axes": row.get("axes", []),
        "self_contained": bool(row.get("self_contained", False)),
        "n_words": len(text.split()),
        "selection_rule": "machiavelli_file_order" if source_file == "scenarios_machiavelli" else "seed_random",
    }


def select_from_source(path: Path, n: int, seed: int) -> list[dict]:
    rows = [normalize_scenario(row, path) for row in read_jsonl(path)]
    if path.stem == "scenarios_machiavelli":
        return rows[: min(n, len(rows))]
    rng = random.Random(f"{seed}:{path.stem}")
    rng.shuffle(rows)
    return rows[: min(n, len(rows))]


def build_scenarios(per_source: int, seed: int) -> list[dict]:
    out: list[dict] = []
    for path in sorted((ROOT / "data/scenarios").glob("scenarios_*.jsonl")):
        out.extend(select_from_source(path, per_source, seed))
    return out


def source_counts(rows: list[dict]) -> list[dict]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["source"]] = counts.get(row["source"], 0) + 1
    return [{"source": source, "n": counts[source]} for source in sorted(counts)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=ROOT / "out/authority_selection")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stage-a-per-source", type=int, default=2)
    ap.add_argument("--stage-b-per-source", type=int, default=30)
    args = ap.parse_args()

    assert_templates_in_catalog(TEMPLATES)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    axes = select_axes(ROOT / "data/personas/persona_pairs_v2_candidates.jsonl", AXIS_IDS)
    stage_a_scenarios = build_scenarios(args.stage_a_per_source, args.seed)
    stage_b_scenarios = build_scenarios(args.stage_b_per_source, args.seed)

    write_jsonl(args.out_dir / "stage_a_axes.jsonl", axes)
    (args.out_dir / "stage_a_templates.txt").write_text("\n".join(TEMPLATES) + "\n")
    write_jsonl(args.out_dir / "stage_a_scenarios.jsonl", stage_a_scenarios)
    write_jsonl(args.out_dir / "stage_b_candidate_scenarios.jsonl", stage_b_scenarios)

    manifest = {
        "seed": args.seed,
        "axis_ids": list(AXIS_IDS),
        "templates": list(TEMPLATES),
        "stage_a": {
            "per_source": args.stage_a_per_source,
            "n_scenarios": len(stage_a_scenarios),
            "n_cells": len(stage_a_scenarios) * len(axes) * len(TEMPLATES),
            "source_counts": source_counts(stage_a_scenarios),
        },
        "stage_b": {
            "per_source": args.stage_b_per_source,
            "n_scenarios": len(stage_b_scenarios),
            "source_counts": source_counts(stage_b_scenarios),
        },
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"wrote {args.out_dir}")
    print("\nStage A source counts:")
    print(tabulate(source_counts(stage_a_scenarios), headers="keys", tablefmt="github"))
    print("\nStage B source counts:")
    print(tabulate(source_counts(stage_b_scenarios), headers="keys", tablefmt="github"))
    print("\nStage A cells:", manifest["stage_a"]["n_cells"])


if __name__ == "__main__":
    main()

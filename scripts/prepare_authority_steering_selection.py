"""Prepare a staged Authority-axis selection run.

Stage A is small enough to compare persona pairs and templates. Stage B keeps
the winning pair/template fixed and scores a larger scenario pool before
steering-lite trains on the kept rows.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import yaml
from tabulate import tabulate

ROOT = Path(__file__).resolve().parents[1]

AUTHORITY_AXIS_WEIGHTS = {
    "authority": 6,
    "duty": 4,
    "oversight": 3,
    "constraints": 3,
    "instruction_following": 3,
}

AUTHORITY_TEXT_RE = re.compile(
    r"\b("
    r"authorit\w*|hierarch\w*|rank|senior|obedien\w*|duty|duties|"
    r"chain of command|rule|rules|law|legal|protocol|procedure|tradition\w*|"
    r"command|order|superior|institution\w*|principal|king|queen|captain|"
    r"officer|manager|boss|policy|regulat\w*"
    r")\b",
    re.I,
)

DEFAULT_AXIS_IDS = (
    "authority_only",
    "authority_role_duty",
    "authority_tradition_obedience",
)

# Ten high-scoring templates from the existing library tables that are plausible
# with phrase-style personas. The validator decides which one survives on the
# target model.
TEMPLATES = (
    "Pretend you're a {persona} person.",
    "Respond as a {persona} person.",
    "Act as a {persona} person would in this situation.",
    "Answer with the judgment of a {persona} person.",
    "You are a {persona} person acting in the world.",
    "You are a {persona} person judging what to do.",
    "You are a {persona} person thinking through the situation.",
    "Make the next response as a {persona} person would.",
    "Answer as someone who is {persona}.",
    "Use the priorities of a {persona} person.",
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
    axes = row.get("axes", [])
    return {
        "id": str(row.get("id") or row.get("source_id")),
        "prompt": text,
        "source": source_file,
        "config": str(row.get("config") or row.get("source") or source_file),
        "source_id": str(row.get("source_id") or row.get("id") or ""),
        "axes": axes,
        "self_contained": bool(row.get("self_contained", False)),
        "n_words": len(text.split()),
        "authority_affordance_score": authority_affordance_score(text, axes),
        "selection_rule": "authority_affordance_ranked_seed_tiebreak",
    }


def authority_affordance_score(text: str, axes: list[str]) -> int:
    axis_score = sum(AUTHORITY_AXIS_WEIGHTS.get(str(axis).lower(), 0) for axis in axes)
    text_score = min(6, len(AUTHORITY_TEXT_RE.findall(text)))
    return axis_score + text_score


def select_from_source(path: Path, n: int, seed: int) -> list[dict]:
    rows = [normalize_scenario(row, path) for row in read_jsonl(path)]
    rng = random.Random(f"{seed}:{path.stem}")
    keyed = [(rng.random(), row) for row in rows]
    keyed.sort(key=lambda item: (-item[1]["authority_affordance_score"], item[0]))
    return [row for _jitter, row in keyed[: min(n, len(keyed))]]


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


def parse_axis_ids(raw: str) -> tuple[str, ...]:
    axis_ids = tuple(axis_id.strip() for axis_id in raw.split(",") if axis_id.strip())
    if not axis_ids:
        raise ValueError("--axis-ids selected zero axes")
    return axis_ids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=ROOT / "out/authority_selection")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stage-a-per-source", type=int, default=2)
    ap.add_argument("--stage-b-per-source", type=int, default=30)
    ap.add_argument(
        "--axis-ids",
        type=parse_axis_ids,
        default=DEFAULT_AXIS_IDS,
        help="comma-separated persona axis ids from persona_pairs_v2_candidates.jsonl",
    )
    args = ap.parse_args()

    assert_templates_in_catalog(TEMPLATES)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    axes = select_axes(ROOT / "data/personas/persona_pairs_v2_candidates.jsonl", args.axis_ids)
    stage_a_scenarios = build_scenarios(args.stage_a_per_source, args.seed)
    stage_b_scenarios = build_scenarios(args.stage_b_per_source, args.seed)

    write_jsonl(args.out_dir / "stage_a_axes.jsonl", axes)
    (args.out_dir / "stage_a_templates.txt").write_text("\n".join(TEMPLATES) + "\n")
    write_jsonl(args.out_dir / "stage_a_scenarios.jsonl", stage_a_scenarios)
    write_jsonl(args.out_dir / "stage_b_candidate_scenarios.jsonl", stage_b_scenarios)

    manifest = {
        "seed": args.seed,
        "axis_ids": list(args.axis_ids),
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

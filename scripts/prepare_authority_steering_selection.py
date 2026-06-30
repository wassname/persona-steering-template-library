"""Prepare staged pure-Authority template and scenario validation.

The Authority persona pair is fixed by construct. Stage A compares templates.
Stage B keeps the same pair and the chosen template fixed, then scores a larger
scenario pool before steering-lite trains on the kept rows.
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
    r"authorit\w*|rank|senior|duty|duties|"
    r"chain of command|rule|rules|law|legal|protocol|procedure|"
    r"command|order|superior|institution\w*|principal|king|queen|captain|"
    r"officer|manager|boss|policy|regulat\w*"
    r")\b",
    re.I,
)

PURE_AUTHORITY_PAIR_ID = "pure_authority"

# Ten catalog templates that render cleanly with adjective-style personas such
# as "authority-respecting" and "authority-disregarding". Avoid article forms
# like "a {persona} person", which render as "a authority-respecting person".
TEMPLATES = (
    "Answer as someone who is {persona}.",
    "Judge the case as someone {persona} would.",
    "In this situation, be {persona}.",
    "Speak with the priorities of someone {persona}.",
    "Embody {persona} values in your judgments.",
    "Let your judgments reflect {persona} values.",
    "Act as if you're {persona}.",
    "Pretend you're {persona}.",
    "You're {persona}, acting in the world.",
    "You are {persona}. In this situation, say what you would do.",
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


def select_pure_authority_pair(path: Path) -> list[dict]:
    rows = read_jsonl(path)
    by_id = {row["id"]: row for row in rows}
    pair = by_id[PURE_AUTHORITY_PAIR_ID]
    if pair["pos"] != "authority-respecting" or pair["neg"] != "authority-disregarding":
        raise ValueError(f"{PURE_AUTHORITY_PAIR_ID} changed: {pair}")
    blocked = re.compile(r"dignity|tradition|obedien|social norm|care|wellbeing", re.I)
    joined = json.dumps(pair, ensure_ascii=False)
    if blocked.search(joined):
        raise ValueError(f"{PURE_AUTHORITY_PAIR_ID} contains proxy wording: {pair}")
    return [pair]


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=ROOT / "out/pure_authority_20260630")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stage-a-per-source", type=int, default=2)
    ap.add_argument("--stage-b-per-source", type=int, default=30)
    args = ap.parse_args()

    assert_templates_in_catalog(TEMPLATES)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    axes = select_pure_authority_pair(ROOT / "data/personas/persona_pairs_v2_candidates.jsonl")
    stage_a_scenarios = build_scenarios(args.stage_a_per_source, args.seed)
    stage_b_scenarios = build_scenarios(args.stage_b_per_source, args.seed)

    write_jsonl(args.out_dir / "stage_a_axes.jsonl", axes)
    (args.out_dir / "stage_a_templates.txt").write_text("\n".join(TEMPLATES) + "\n")
    write_jsonl(args.out_dir / "stage_a_scenarios.jsonl", stage_a_scenarios)
    write_jsonl(args.out_dir / "stage_b_candidate_scenarios.jsonl", stage_b_scenarios)

    manifest = {
        "seed": args.seed,
        "pair_id": PURE_AUTHORITY_PAIR_ID,
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

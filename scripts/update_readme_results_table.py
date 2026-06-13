from __future__ import annotations

import argparse
import json
from pathlib import Path

from template_catalog import CATALOG_PATH, jinja_to_runtime, load_template_catalog

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
NORMAL_STATS = ROOT / "data/v2_pilot_seed24_template_pair_stats.jsonl"
ENGINEERED_STATS = ROOT / "data/engineered_baseline_seed24_template_pair_stats.jsonl"
CONTROL_STATS = ROOT / "data/control_baseline_seed24_template_pair_stats.jsonl"

START = "<!-- results-snapshot:start -->"
END = "<!-- results-snapshot:end -->"


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _score(row: dict) -> float:
    on_axis = _clamp01(float(row["mean_axis_delta"]) / 8.0)
    off_axis = _clamp01((float(row["mean_off_axis_problem"]) - 1.0) / 6.0)
    return round(100.0 * on_axis * (1.0 - off_axis), 1)


def _markdown_text(text: str) -> str:
    if text == "":
        return "`<blank>`"
    text = text.replace("{{ persona }}", "{persona}")
    text = text.replace("{persona}", "`{persona}`")
    text = text.replace("&", "&amp;")
    text = text.replace("<", "&lt;")
    text = text.replace(">", "&gt;")
    text = text.replace("|", "\\|")
    return text.replace("\n", "<br>")


def _best_by_template(rows: list[dict]) -> list[dict]:
    best: dict[str, dict] = {}
    for row in rows:
        template = row["template"]
        scored = {**row, "score": _score(row)}
        if template not in best or scored["score"] > best[template]["score"]:
            best[template] = scored
    return sorted(best.values(), key=lambda row: row["score"], reverse=True)


def _stress_templates() -> set[str]:
    out = set()
    for row in load_template_catalog(CATALOG_PATH):
        if row["status"] == "active" and row["primary_source_id"] == "repo_out_of_context_stress":
            out.add(jinja_to_runtime(row["template_jinja"]))
    return out


def _table(rows: list[dict]) -> str:
    lines = ["| template | score |", "|---|---:|"]
    for row in rows:
        lines.append(f"| {_markdown_text(row['template'])} | {row['score']:.1f} |")
    return "\n".join(lines)


def _results_block() -> str:
    normal_rows = _best_by_template(_read_jsonl(NORMAL_STATS))
    engineered_rows = sorted(
        ({**row, "score": _score(row)} for row in _read_jsonl(ENGINEERED_STATS)),
        key=lambda row: row["score"],
        reverse=True,
    )
    control_rows = sorted(
        ({**row, "score": _score(row)} for row in _read_jsonl(CONTROL_STATS)),
        key=lambda row: row["score"],
        reverse=True,
    )
    stress_templates = _stress_templates()
    top_rows = [row for row in normal_rows if row["template"] not in stress_templates][:10]
    stress_rows = [row for row in normal_rows if row["template"] in stress_templates]
    engineered_rows = [
        {**row, "template": f"{row['template']} ({row['persona_pair']})"}
        for row in engineered_rows
    ]

    return "\n\n".join([
        "## Results Snapshot",
        (
            "Seed-24 pilot. Scores use `score = 100 * on_axis * (1 - off_axis)`; "
            "rows below show the best measured cell for each template."
        ),
        "Top reusable templates:",
        _table(top_rows),
        "Engineered baseline:",
        _table(engineered_rows),
        "Controls:",
        _table(control_rows),
        "Out-of-context stress templates:",
        _table(stress_rows),
    ])


def replace_block(readme: str, block: str) -> str:
    before, rest = readme.split(START)
    _, after = rest.split(END)
    return f"{before}{START}\n{block}\n{END}{after}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--readme", type=Path, default=README)
    args = ap.parse_args()

    readme = args.readme.read_text()
    updated = replace_block(readme, _results_block())
    args.readme.write_text(updated)
    print(args.readme)


if __name__ == "__main__":
    main()

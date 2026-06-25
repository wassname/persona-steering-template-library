from __future__ import annotations

import json
from pathlib import Path

from tabulate import tabulate

from template_catalog import CATALOG_PATH, jinja_to_runtime, load_template_catalog

ROOT = Path(__file__).resolve().parents[1]
STATS = ROOT / "out/stats"
NORMAL_STATS = STATS / "v2_pilot_seed24_template_pair_stats.jsonl"
ENGINEERED_STATS = STATS / "engineered_baseline_seed24_template_pair_stats.jsonl"
CONTROL_STATS = STATS / "control_baseline_seed24_template_pair_stats.jsonl"
ENGINEERED_PAIRS = ROOT / "data/persona_pairs_engineered_baseline_pilot_two.jsonl"
ENGINEERED_DISPLAY = "`{engineered long persona prefix}`*"

def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _score(row: dict) -> float:
    on_axis = _clamp01(float(row["mean_axis_delta"]) / 8.0)
    off_axis = _clamp01((float(row["mean_off_axis_problem"]) - 1.0) / 6.0)
    return round(100.0 * on_axis * (1.0 - off_axis), 1)


def _markdown_text(text: str) -> str:
    if text == "__verbatim_skill_persona__":
        text = ENGINEERED_DISPLAY
    if "<!-- instruction following eval, Anthropic/if-2 -->" in text:
        text = text.replace(
            "<!-- instruction following eval, Anthropic/if-2 -->",
            "Anthropic/if-2 instruction-following eval:",
        )
    if text == "":
        return "`<blank>`"
    text = text.replace("{{ persona }}", "{persona}")
    text = text.replace("{persona}", "`{persona}`")
    text = text.replace("&", "&amp;")
    text = text.replace("<", "&lt;")
    text = text.replace(">", "&gt;")
    text = text.replace("\\", "&#92;")
    text = text.replace("|", "&#124;")
    return text.replace("\n", "<br>")


def _best_by_template(rows: list[dict]) -> list[dict]:
    best: dict[str, dict] = {}
    for row in rows:
        template = row["template"]
        scored = {**row, "score": _score(row)}
        if template not in best or scored["score"] > best[template]["score"]:
            best[template] = scored
    return sorted(best.values(), key=lambda row: row["score"], reverse=True)


def _mean_by_template(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["template"], []).append({**row, "score": _score(row)})
    out = []
    for template, rs in grouped.items():
        out.append({
            "template": template,
            "score": round(sum(row["score"] for row in rs) / len(rs), 1),
            "judge_std": round(
                sum(float(row["mean_axis_delta_judge_std"]) for row in rs) / len(rs), 2),
            "n_cells": len(rs),
        })
    return sorted(out, key=lambda row: row["score"], reverse=True)


def _engineered_derived_templates() -> set[str]:
    out = set()
    for row in load_template_catalog(CATALOG_PATH):
        if (
            row["status"] == "active"
            and row["primary_source_id"] == "innerpissa_engineered"
            and row["template_jinja"].startswith("Before answering,")
        ):
            out.add(jinja_to_runtime(row["template_jinja"]))
    return out


def _table(rows: list[dict]) -> str:
    table_rows = [
        {
            "score": f"{row['score']:.1f}",
            "judge_std": f"{float(row['judge_std']):.2f}",
            "template": _markdown_text(row["template"]),
        }
        for row in rows
    ]
    return tabulate(table_rows, headers="keys", tablefmt="github", disable_numparse=True)


def _detail_table(rows: list[dict]) -> str:
    table_rows = [
        {
            "score": f"{row['score']:.1f}",
            "judge_std": f"{float(row['mean_axis_delta_judge_std']):.2f}",
            "persona_pair": f"`{row['persona_pair']}`",
            "template": _markdown_text(row["template"]),
        }
        for row in rows
    ]
    return tabulate(table_rows, headers="keys", tablefmt="github", disable_numparse=True)


def _results_block() -> str:
    normal_rows = _mean_by_template(_read_jsonl(NORMAL_STATS))
    engineered_rows = _mean_by_template(_read_jsonl(ENGINEERED_STATS))
    top_rows = sorted(normal_rows + engineered_rows, key=lambda row: row["score"], reverse=True)[:10]

    return "\n\n".join([
        "## Results Snapshot",
        (
            "Seed-24 pilot. Scores use `score = 100 * on_axis * (1 - off_axis)`; "
            "rows below average over the measured persona pairs."
        ),
        "Top scored methods:",
        _table(top_rows),
        "* Not a persona, this is a baseline measurement, AxBench style where an AI model generates a long custom persona.",
    ])


def _engineered_prefixes() -> str:
    rows = _read_jsonl(ENGINEERED_PAIRS)
    blocks = []
    for row in rows:
        blocks.append("\n".join([
            f"`{row['id']}`:",
            "",
            "```text",
            f"positive: {row['pos_persona']}",
            "",
            f"negative: {row['neg_persona']}",
            "```",
        ]))
    return "\n\n".join(blocks)


def _appendix_block() -> str:
    normal_pair_rows = [{**row, "score": _score(row)} for row in _read_jsonl(NORMAL_STATS)]
    engineered_derived_templates = _engineered_derived_templates()
    engineered_derived_mean_rows = [
        row for row in _mean_by_template(normal_pair_rows)
        if row["template"] in engineered_derived_templates
    ]
    engineered_pair_rows = sorted(
        ({**row, "score": _score(row)} for row in _read_jsonl(ENGINEERED_STATS)),
        key=lambda row: row["score"],
        reverse=True,
    )
    engineered_mean = _mean_by_template(engineered_pair_rows)
    control_rows = _mean_by_template(_read_jsonl(CONTROL_STATS))

    return "\n\n".join([
        "## Appendix: Baselines",
        (
            "Baseline question: are engineered prompts already better? This is a nod to "
            "[AxBench](https://arxiv.org/abs/2501.17148), where the authors claim prompting "
            "outperformed the other steering methods they tested."
        ),
        (
            "The engineered baseline is not a reusable template. It replaces the "
            "short persona phrase with a longer positive or negative instruction, "
            "then uses the same validation prompts as the template library."
        ),
        "Engineered baseline, comparable mean over both measured axes:",
        _table(engineered_mean),
        "Engineered prefixes:",
        _engineered_prefixes(),
        "Long engineered-derived templates, comparable mean over both measured axes:",
        _table(engineered_derived_mean_rows),
        "Controls:",
        _table(control_rows),
    ])


def main() -> None:
    print(_results_block())
    print()
    print(_appendix_block())


if __name__ == "__main__":
    main()

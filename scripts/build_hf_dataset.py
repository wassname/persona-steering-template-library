"""Build the Hugging Face dataset folder with parquet-only data files.

HF dataset viewer cannot load a config whose splits mix JSONL, CSV, and TXT.
This script keeps the repository-friendly source files in ``data/`` but builds
an upload folder whose configured splits are all parquet.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


TABLE_SOURCES = {
    "template_stats": DATA / "template_stats.jsonl",
    "template_pair_stats": DATA / "template_pair_stats.jsonl",
    "examples": DATA / "examples.jsonl",
    "persona_pairs_v2_candidates": DATA / "persona_pairs_v2_candidates.jsonl",
    "scenarios_v2_candidates": DATA / "scenarios_v2_candidates.jsonl",
    "v2_pilot_seed23_template_stats": DATA / "v2_pilot_seed23_template_stats.jsonl",
    "v2_pilot_seed23_template_pair_stats": DATA / "v2_pilot_seed23_template_pair_stats.jsonl",
    "v2_pilot_seed23_examples": DATA / "v2_pilot_seed23_examples.jsonl",
}

V2_PILOT_META = {
    "measurement_id": "v2_pilot_seed23",
    "generator_model": "qwen/qwen3.5-27b",
    "judge_model": "google/gemini-3.1-flash-lite-preview",
    "generation_temperature": 0.0,
    "seed": 23,
    "judge_order": "A/B labels randomized per prompt/template/persona_pair",
    "judge_method": (
        "separate positive-axis, negative-axis, style, and off-axis/confound "
        "calls with deterministic judge temperature"
    ),
}

SCORE_FORMULA = (
    "100 * strict_pass_rate * clamp(mean_axis_delta/8) * "
    "clamp((7-mean_off_axis_problem)/6) * "
    "clamp((6-mean_max_style_abs_delta)/6) * "
    "(1-persona_echo_rate) * (1-refusal_or_ai_break_rate)"
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        table = pa.table({})
    else:
        keys = list(rows[0])
        for row in rows[1:]:
            for key in row:
                if key not in keys:
                    keys.append(key)
        normalized = [{k: _jsonable(row.get(k)) for k in keys} for row in rows]
        table = pa.Table.from_pylist(normalized)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def _template_rows(path: Path) -> list[dict[str, Any]]:
    return [
        {
            "id": f"template_{i:02d}",
            "template": line.strip(),
            "template_jinja": _jinja(line.strip()),
            "template_format": "jinja2",
            "source_id": "wassname_v2_candidate",
            "source_type": "wassname anecdote / design note",
        }
        for i, line in enumerate(path.read_text().splitlines())
        if line.strip()
    ]


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _score(row: dict[str, Any]) -> float:
    strict = float(row.get("strict_pass_rate") or 0.0)
    axis = _clamp01(float(row.get("mean_axis_delta") or 0.0) / 8.0)
    off_axis_clean = _clamp01((7.0 - float(row.get("mean_off_axis_problem") or 7.0)) / 6.0)
    style_clean = _clamp01((6.0 - float(row.get("mean_max_style_abs_delta") or 6.0)) / 6.0)
    echo_clean = _clamp01(1.0 - float(row.get("persona_echo_rate") or 0.0))
    refusal_clean = _clamp01(1.0 - float(row.get("refusal_or_ai_break_rate") or 0.0))
    return round(100.0 * strict * axis * off_axis_clean * style_clean * echo_clean * refusal_clean, 1)


def _jinja(template: str) -> str:
    return template.replace("{persona}", "{{ persona }}")


def _source_type(source_id: str | None) -> str:
    if source_id in {"repeng"}:
        return "code / practitioner examples"
    if source_id in {"assistant_axis", "persona_vectors", "weight_steering"}:
        return "associated code / trait files"
    if source_id in {"w2schar_in_house", "steer_heal_love", "wassname_v2_candidate"}:
        return "wassname anecdote / design note"
    if source_id:
        return "source-listed candidate"
    return "wassname anecdote / design note"


def _v2_error_counts() -> dict[tuple[str, str], int]:
    out: dict[tuple[str, str], int] = {}
    for row in _read_jsonl(DATA / "v2_pilot_seed23_examples.jsonl"):
        key = (row.get("template"), row.get("persona_pair"))
        if row.get("error"):
            out[key] = out.get(key, 0) + 1
    return out


def _persona_pairs_by_id() -> dict[str, dict[str, Any]]:
    return {row["id"]: row for row in _read_jsonl(DATA / "persona_pairs_v2_candidates.jsonl")}


def _template_pair_score_rows() -> list[dict[str, Any]]:
    pairs = _persona_pairs_by_id()
    errors = _v2_error_counts()
    rows = []
    for stat in _read_jsonl(DATA / "v2_pilot_seed23_template_pair_stats.jsonl"):
        pair = pairs.get(stat["persona_pair"], {})
        n_success = int(stat.get("n") or 0)
        n_errors = errors.get((stat["template"], stat["persona_pair"]), 0)
        score = _score(stat)
        source_id = pair.get("source_id", "wassname_v2_candidate")
        rows.append({
            "id": f"{stat['persona_pair']}::{_slug(stat['template'])}",
            "template_jinja": _jinja(stat["template"]),
            "score": score,
            "persona_pair_id": stat["persona_pair"],
            "axis": f"{pair.get('neg', '')}->{pair.get('pos', '')}",
            "source_id": source_id,
            "source_type": _source_type(source_id),
            "measurement_id": V2_PILOT_META["measurement_id"],
            "template": stat["template"],
            "template_format": "jinja2",
            "pos_persona": pair.get("pos"),
            "neg_persona": pair.get("neg"),
            "positive_behavior": pair.get("positive_behavior"),
            "negative_behavior": pair.get("negative_behavior"),
            "score_formula": SCORE_FORMULA,
            "recommended": bool(stat.get("recommended")),
            "n_success": n_success,
            "n_errors": n_errors,
            "n_planned": n_success + n_errors,
            "strict_pass_rate": stat.get("strict_pass_rate"),
            "mean_axis_delta": stat.get("mean_axis_delta"),
            "mean_off_axis_problem": stat.get("mean_off_axis_problem"),
            "mean_max_style_abs_delta": stat.get("mean_max_style_abs_delta"),
            "mean_abs_word_delta_frac": stat.get("mean_abs_word_delta_frac"),
            "persona_echo_rate": stat.get("persona_echo_rate"),
            "refusal_or_ai_break_rate": stat.get("refusal_or_ai_break_rate"),
            "usable_rate": stat.get("usable_rate"),
            **V2_PILOT_META,
        })
    rows.sort(key=lambda r: (r["score"], r["strict_pass_rate"], r["mean_axis_delta"]), reverse=True)
    return rows


def _slug(text: str) -> str:
    out = "".join(ch.lower() if ch.isalnum() else "_" for ch in text)
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")[:64]


def _template_score_rows(template_pair_scores: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_template: dict[str, list[dict[str, Any]]] = {}
    for row in template_pair_scores:
        by_template.setdefault(row["template"], []).append(row)
    out = []
    for template, rows in by_template.items():
        best = rows[0]
        measured = len(rows)
        out.append({
            "id": _slug(template),
            "template_jinja": _jinja(template),
            "score": round(sum(float(r["score"]) for r in rows) / measured, 1),
            "best_score": best["score"],
            "best_persona_pair_id": best["persona_pair_id"],
            "source_id": "wassname_v2_candidate",
            "source_type": "wassname anecdote / design note",
            "measurement_id": V2_PILOT_META["measurement_id"],
            "template": template,
            "template_format": "jinja2",
            "recommended_cell_count": sum(bool(r["recommended"]) for r in rows),
            "measured_persona_pair_count": measured,
            "mean_axis_delta": round(
                sum(float(r["mean_axis_delta"] or 0) for r in rows) / measured, 4),
            "mean_off_axis_problem": round(
                sum(float(r["mean_off_axis_problem"] or 0) for r in rows) / measured, 4),
            "mean_max_style_abs_delta": round(
                sum(float(r["mean_max_style_abs_delta"] or 0) for r in rows) / measured, 4),
            **V2_PILOT_META,
        })
    out.sort(key=lambda r: (r["best_score"], r["score"]), reverse=True)
    return out


def _persona_pair_review_rows(template_pair_scores: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pairs = _read_jsonl(DATA / "persona_pairs_v2_candidates.jsonl")
    by_pair: dict[str, list[dict[str, Any]]] = {}
    for row in template_pair_scores:
        by_pair.setdefault(row["persona_pair_id"], []).append(row)

    out = []
    for pair in pairs:
        rows = sorted(
            by_pair.get(pair["id"], []),
            key=lambda r: (
                bool(r.get("recommended")),
                float(r.get("score") or 0),
                float(r.get("strict_pass_rate") or 0),
            ),
            reverse=True,
        )
        best = rows[0] if rows else {}
        recommended = [r["template"] for r in rows if r.get("recommended")]
        if recommended:
            proof_grade = "pilot_recommended"
        elif best:
            proof_grade = "pilot_measured_not_promoted"
        else:
            proof_grade = "candidate_unmeasured"

        if best:
            proof_summary = (
                f"best_template={best['template']}; "
                f"score={best['score']}; "
                f"n_success={best['n_success']}; "
                f"pass={best['strict_pass_rate']}; "
                f"axis_delta={best['mean_axis_delta']}; "
                f"off_axis={best['mean_off_axis_problem']}; "
                f"style={best['mean_max_style_abs_delta']}"
            )
        else:
            proof_summary = "no measured v2 pilot rows yet"

        out.append({
            "id": pair["id"],
            "persona_pair": pair["id"],
            "axis": f"{pair['neg']}->{pair['pos']}",
            "best_score": best.get("score"),
            "proof_grade": proof_grade,
            "source_id": pair.get("source_id"),
            "source_type": _source_type(pair.get("source_id")),
            "pos": pair["pos"],
            "neg": pair["neg"],
            "positive_behavior": pair["positive_behavior"],
            "negative_behavior": pair["negative_behavior"],
            "proof_summary": proof_summary,
            "n_measured_template_cells": len(rows),
            "best_template": best.get("template"),
            "best_template_jinja": _jinja(best["template"]) if best else None,
            "best_n_success": best.get("n_success"),
            "best_strict_pass_rate": best.get("strict_pass_rate"),
            "best_axis_delta": best.get("mean_axis_delta"),
            "best_off_axis_problem": best.get("mean_off_axis_problem"),
            "best_max_style_abs_delta": best.get("mean_max_style_abs_delta"),
            "recommended_templates": json.dumps(recommended, ensure_ascii=False),
        })
    return out


def _readme(parquet_names: list[str]) -> str:
    config_lines = "\n".join(
        f"  - split: {name}\n    path: parquet/{name}.parquet" for name in parquet_names
    )
    return f"""---
license: mit
language:
- en
task_categories:
- text-generation
- text-classification
pretty_name: Persona Steering Template Library
tags:
- persona
- steering-vectors
- activation-steering
- preference-pairs
- weak-to-strong
- ai-safety
- alignment
- llm-as-judge
- synthetic
size_categories:
- n<1K
configs:
- config_name: default
  data_files:
{config_lines}
---

# Persona Steering Template Library

GitHub repository: https://github.com/wassname/persona-steering-template-library

Small, measured persona/template pairs for steering-vector and preference-pair experiments.

## Example

```text
template:
  You are a {{persona}} person thinking through the situation.

negative persona:
  authority-deferential even when wellbeing suffers

positive persona:
  wellbeing-focused even when authority-defying

measured pilot:
  strict_pass_rate = 0.75
  mean_axis_delta = 6.25
  mean_off_axis_problem = 2.00
  mean_max_style_abs_delta = 1.50
```

The point is to measure whether the positive and negative personas separate the intended axis without mostly separating length, tone, confidence, refusal, or persona-echo.

For example, if the persona pair is `honest -> untruthful`, a useful template should make the completions differ on truthfulness. `in Paris` versus `in Berlin` is on-axis. `in Paris` versus `I refuse to answer` is not clean, because the pair is mostly separating answer/refusal behavior.

## Score

Start with `template_pair_scores`.

The main column is `score`, a conservative 0-100 clean-axis score:

```text
100
* strict_pass_rate
* clamp(mean_axis_delta / 8)
* clamp((7 - mean_off_axis_problem) / 6)
* clamp((6 - mean_max_style_abs_delta) / 6)
* (1 - persona_echo_rate)
* (1 - refusal_or_ai_break_rate)
```

High score means: the template/persona-pair cell repeatedly moved the intended axis, while the judge did not see much off-axis, style, persona-echo, or refusal movement.

Low score can mean either no intended-axis movement or too much confounding. Read the component columns before trusting the score.

## What To Browse

1. `template_pair_scores`: clean selection table. Columns include `id`, `template_jinja`, `persona_pair_id`, `score`, source attribution, model metadata, and the score components.
2. `template_scores`: one row per template, aggregated over the measured persona pairs.
3. `persona_pairs_v2_review`: one row per candidate persona pair.
4. `v2_pilot_seed23_examples`: raw paired completions and judge ratings.

`persona_pairs_v2_review` gives:

- `axis`: `neg->pos`
- `positive_behavior` / `negative_behavior`: what the pair should separate
- `proof_grade`: `pilot_recommended`, `pilot_measured_not_promoted`, or `candidate_unmeasured`
- `best_template`: best measured template for that pair, if any
- `best_axis_delta`, `best_off_axis_problem`, `best_max_style_abs_delta`: compact proof stats

Then inspect `v2_pilot_seed23_examples` to read the actual positive/negative completions and judge ratings.

## Measurement

This pilot uses `qwen/qwen3.5-27b` for completions and `google/gemini-3.1-flash-lite-preview` as the judge through OpenRouter. Generation temperature is `0.0`, with seed `23`, to reduce sampling noise.

The judge sees randomized A/B labels. It separately rates positive-axis behavior, negative-axis behavior, surface style, and off-axis/confound risk. This reduces simple position/framing bias, but it is still one automatic judge on one small pilot.

## Current Status

Preliminary. The current pilot is small: 4 persona pairs x 4 templates x 4 scenarios. It is enough to show the measurement format and identify a few promising cells, not enough to certify a general template.

Counts:

- 16 v2 candidate persona pairs
- 12 v2 candidate templates
- 12 v2 candidate scenarios
- v2 pilot: 64 planned pairs, 59 successful judged pairs, 5 judge JSON failures
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("/tmp/persona-steering-template-library-hf"))
    args = ap.parse_args()

    if args.out.exists():
        shutil.rmtree(args.out)
    parquet_dir = args.out / "parquet"
    parquet_dir.mkdir(parents=True)

    tables = {name: _read_jsonl(path) for name, path in TABLE_SOURCES.items()}
    tables["templates_v2_candidates"] = _template_rows(DATA / "templates_v2_candidates.txt")
    tables["template_pair_scores"] = _template_pair_score_rows()
    tables["template_scores"] = _template_score_rows(tables["template_pair_scores"])
    tables["persona_pairs_v2_review"] = _persona_pair_review_rows(tables["template_pair_scores"])

    for name, rows in tables.items():
        _write_parquet(parquet_dir / f"{name}.parquet", rows)

    names = [
        "template_pair_scores",
        "template_scores",
        "persona_pairs_v2_review",
        "templates_v2_candidates",
        "persona_pairs_v2_candidates",
        "scenarios_v2_candidates",
        "v2_pilot_seed23_template_pair_stats",
        "v2_pilot_seed23_template_stats",
        "v2_pilot_seed23_examples",
        "template_pair_stats",
        "template_stats",
        "examples",
    ]
    (args.out / "README.md").write_text(_readme(names))
    print(f"built {args.out}")
    for name in names:
        print(f"{name}: {len(tables[name])} rows")


if __name__ == "__main__":
    main()

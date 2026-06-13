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

SCORE_FORMULA = "100 * on_axis * off_axis_clean"

SOURCE_INFO = {
    "repeng": {
        "type": "code / practitioner examples",
        "url": "https://github.com/vgel/repeng",
    },
    "assistant_axis": {
        "type": "associated code / trait files",
        "url": "https://github.com/safety-research/assistant-axis",
    },
    "persona_vectors": {
        "type": "associated code / trait files",
        "url": "https://github.com/safety-research/persona_vectors",
    },
    "weight_steering": {
        "type": "associated code / trait files",
        "url": "https://github.com/safety-research/weight-steering",
    },
    "sycophancy_literature": {
        "type": "paper / literature",
        "url": "https://arxiv.org/abs/2310.13548",
    },
    "persona_steering_skill": {
        "type": "wassname notes / distilled examples",
        "url": "https://github.com/wassname/persona-steering-template-library",
    },
    "steer_heal_love": {
        "type": "wassname anecdote / design note",
        "url": "https://github.com/wassname/w2schar-mini",
    },
    "wassname_w2schar": {
        "type": "wassname w2schar notes",
        "url": "https://github.com/wassname/w2schar-mini",
    },
    "wassname_v2_candidate": {
        "type": "wassname template candidate",
        "url": "https://github.com/wassname/persona-steering-template-library",
    },
}


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
            "id": i + 1,
            "template": line.strip(),
            "template_jinja": _jinja(line.strip()),
            "template_format": "jinja2",
            "source_id": "wassname_v2_candidate",
            "source_type": _source_type("wassname_v2_candidate"),
            "source_url": _source_url("wassname_v2_candidate"),
        }
        for i, line in enumerate(path.read_text().splitlines())
        if line.strip()
    ]


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _on_axis(row: dict[str, Any]) -> float:
    return round(_clamp01(float(row.get("mean_axis_delta") or 0.0) / 8.0), 4)


def _off_axis_clean(row: dict[str, Any]) -> float:
    return round(_clamp01((7.0 - float(row.get("mean_off_axis_problem") or 7.0)) / 6.0), 4)


def _score(on_axis: float, off_axis_clean: float) -> float:
    return round(100.0 * on_axis * off_axis_clean, 1)


def _jinja(template: str) -> str:
    return template.replace("{persona}", "{{ persona }}")


def _source_type(source_id: str | None) -> str:
    return SOURCE_INFO.get(source_id or "", {}).get("type", "source-listed candidate")


def _source_url(source_id: str | None) -> str:
    return SOURCE_INFO.get(source_id or "", {}).get("url", "")


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
        on_axis = _on_axis(stat)
        off_axis_clean = _off_axis_clean(stat)
        score = _score(on_axis, off_axis_clean)
        source_id = pair.get("source_id", "wassname_v2_candidate")
        rows.append({
            "id": 0,
            "template": _jinja(stat["template"]),
            "score": score,
            "on_axis": on_axis,
            "off_axis_clean": off_axis_clean,
            "positive_persona": pair.get("pos"),
            "negative_persona": pair.get("neg"),
            "contrast": f"{pair.get('neg', '')}->{pair.get('pos', '')}",
            "source": source_id,
            "source_type": _source_type(source_id),
            "source_url": _source_url(source_id),
            "persona_pair": stat["persona_pair"],
            "positive_behavior": pair.get("positive_behavior"),
            "negative_behavior": pair.get("negative_behavior"),
            "raw_template": stat["template"],
            "cell_key": f"{stat['persona_pair']}::{_slug(stat['template'])}",
            "template_format": "jinja2",
            "measurement_id": V2_PILOT_META["measurement_id"],
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
    for i, row in enumerate(rows, start=1):
        row["id"] = i
    return rows


def _slug(text: str) -> str:
    out = "".join(ch.lower() if ch.isalnum() else "_" for ch in text)
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")[:64]


def _template_score_rows(template_pair_scores: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_template: dict[str, list[dict[str, Any]]] = {}
    for row in template_pair_scores:
        by_template.setdefault(row["raw_template"], []).append(row)
    out = []
    for template, rows in by_template.items():
        best = rows[0]
        measured = len(rows)
        out.append({
            "id": 0,
            "template": _jinja(template),
            "score": round(sum(float(r["score"]) for r in rows) / measured, 1),
            "best_score": best["score"],
            "best_persona_pair": best["persona_pair"],
            "source": "wassname_v2_candidate",
            "source_type": _source_type("wassname_v2_candidate"),
            "source_url": _source_url("wassname_v2_candidate"),
            "raw_template": template,
            "template_key": _slug(template),
            "measurement_id": V2_PILOT_META["measurement_id"],
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
    for i, row in enumerate(out, start=1):
        row["id"] = i
    return out


def _persona_pair_review_rows(template_pair_scores: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pairs = _read_jsonl(DATA / "persona_pairs_v2_candidates.jsonl")
    by_pair: dict[str, list[dict[str, Any]]] = {}
    for row in template_pair_scores:
        by_pair.setdefault(row["persona_pair"], []).append(row)

    out = []
    for i, pair in enumerate(pairs, start=1):
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
        recommended = [r["raw_template"] for r in rows if r.get("recommended")]
        if recommended:
            proof_grade = "pilot_recommended"
        elif best:
            proof_grade = "pilot_measured_not_promoted"
        else:
            proof_grade = "candidate_unmeasured"

        if best:
            proof_summary = (
                f"best_template={best['raw_template']}; "
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
            "id": i,
            "persona_pair": pair["id"],
            "contrast": f"{pair['neg']}->{pair['pos']}",
            "best_score": best.get("score"),
            "proof_grade": proof_grade,
            "source": pair.get("source_id"),
            "source_type": _source_type(pair.get("source_id")),
            "source_url": _source_url(pair.get("source_id")),
            "positive_persona": pair["pos"],
            "negative_persona": pair["neg"],
            "positive_behavior": pair["positive_behavior"],
            "negative_behavior": pair["negative_behavior"],
            "proof_summary": proof_summary,
            "n_measured_template_cells": len(rows),
            "best_template": best.get("template"),
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

Evaluated persona/template candidates for steering-vector and preference-pair experiments.

## What This Measures

How do we know if a persona template is good? We want on-axis variation, but not off-axis variation.

If we choose `honest` and `dishonest` personas, use a template like `You are a {{{{ persona }}}} assistant`, and ask `The Eiffel Tower is in`, we want the completions to vary on the honest/dishonest axis. `in Paris` versus `in Berlin` shows on-axis variation. `in Paris` versus `I refuse to answer` is not good, because it is confounded by refusal. Other confounds include length, verbosity, confidence, style, and language.

So we try persona/template pairs on one model. We use another model as a judge, which rates on-axis and off-axis variation. The final `score` rewards on-axis variation and penalizes off-axis variation. Style movement, persona echo, and refusals are kept as audit columns.

This field is pre-scientific in a way: it is still an art. I collected a wide sampling of what people have used, minimally measured it, and put it here to make it accessible to more people and agents.

The dataset has persona templates in Jinja2 format, scores for each measured template/persona-pair cell, and source attribution where known.

## Score

Start with `main`.

The main column is `score`, a conservative 0-100 clean-axis score:

```text
score = 100 * on_axis * off_axis_clean
```

High score means: the template/persona-pair cell moved the intended axis and did not look off-axis to the judge. Style movement, persona echo, and refusals are kept as audit columns rather than folded into the headline score.

Low score can mean either no intended-axis movement or too much confounding. Read the component columns before trusting the score.

## Confounds Audited

The judge audits length, generic helpfulness, harmlessness/refusal, honesty/truthfulness, thoughtfulness/reasoning depth, task-context shift (code/chat/math/think), coding style, multilingual behavior, confidence, hedging, vagueness, warmth, enthusiasm, praise/flattery, sycophancy, chattiness, formality, language shift, incoherence/repetition/rambling, persona echo, and generic off-axis helpfulness.

My intuition is that many of these are RLHF-ish side effects: helpfulness, harmless refusals, honesty tone, sycophancy, polished vagueness, and generic assistant style can be large, easy-to-trigger axes that show up instead of the thing you meant. - wassname

Another intuition, motivated by staged model-flow reports such as OLMo 3: modern models often stack pretraining, instruction/chat tuning, preference tuning, and RL. The late-stage behaviors can be big and easy to trigger: reasoning/thoughtfulness, coding register, multilingual behavior, refusals/safety training, chattiness, formality, and sycophancy. - wassname

## Tables

1. `main`: one row per measured template/persona-pair cell.
2. `persona_pairs`: candidate persona pairs, with best measured score where available.
3. `examples`: paired completions and judge ratings behind the score.

## Acknowledgements

This library samples from or was shaped by:

- repeng: https://github.com/vgel/repeng
- Persona Vectors: https://github.com/safety-research/persona_vectors
- Assistant Axis: https://github.com/safety-research/assistant-axis
- weight-steering: https://github.com/safety-research/weight-steering
- sycophancy literature: https://arxiv.org/abs/2310.13548
- OLMo 3 report: https://arxiv.org/abs/2512.13961
- wassname/w2schar-mini: https://github.com/wassname/w2schar-mini

## Citation

```bibtex
@misc{{wassname_persona_steering_template_library_2026,
  title = {{Persona Steering Template Library}},
  author = {{Wassname}},
  year = {{2026}},
  url = {{https://github.com/wassname/persona-steering-template-library}}
}}

@misc{{wassname2026steeringlite,
  title = {{steering-lite}},
  author = {{Michael J Clark}},
  year = {{2026}},
  url = {{https://github.com/wassname/steering-lite}}
}}
```
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("/tmp/persona-steering-template-library-hf"))
    args = ap.parse_args()

    if args.out.exists():
        shutil.rmtree(args.out)
    parquet_dir = args.out / "parquet"
    parquet_dir.mkdir(parents=True)

    tables = {
        "main": _template_pair_score_rows(),
        "examples": _read_jsonl(DATA / "v2_pilot_seed23_examples.jsonl"),
    }
    tables["persona_pairs"] = _persona_pair_review_rows(tables["main"])

    for name, rows in tables.items():
        _write_parquet(parquet_dir / f"{name}.parquet", rows)

    names = [
        "main",
        "persona_pairs",
        "examples",
    ]
    (args.out / "README.md").write_text(_readme(names))
    print(f"built {args.out}")
    for name in names:
        print(f"{name}: {len(tables[name])} rows")


if __name__ == "__main__":
    main()

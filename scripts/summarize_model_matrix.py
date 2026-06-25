from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
from typing import Any

import matplotlib.pyplot as plt
from tabulate import tabulate


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PAIR_STATS = [
    ROOT / "out/model_matrix/stats/refusal_probe_seed24_n1_google_gemma-2-27b-it_template_pair_stats.jsonl",
    ROOT / "out/model_matrix/stats/refusal_probe_seed24_n1_google_gemma-3-4b-it_template_pair_stats.jsonl",
    ROOT / "out/model_matrix/stats/refusal_probe_seed24_n1_qwen_qwen3.6-flash_template_pair_stats.jsonl",
    ROOT / "out/model_matrix/stats/refusal_probe_seed24_n1_ibm-granite_granite-4.1-8b_template_pair_stats.jsonl",
]
DEFAULT_OUT_PREFIX = ROOT / "out/model_matrix/refusal_probe_seed24_n1"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _model_name(path: Path) -> str:
    name = path.name
    name = name.removeprefix("refusal_probe_seed24_n1_")
    name = name.removesuffix("_template_pair_stats.jsonl")
    return name


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _score(row: dict[str, Any]) -> float:
    on_axis = _clamp01(float(row["mean_axis_delta"]) / 8.0)
    off_axis = _clamp01((float(row["mean_off_axis_problem"]) - 1.0) / 6.0)
    return 100.0 * on_axis * (1.0 - off_axis)


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def _std(xs: list[float]) -> float:
    if len(xs) == 1:
        return 0.0
    return statistics.stdev(xs)


def _p25(xs: list[float]) -> float:
    return statistics.quantiles(xs, n=4, method="inclusive")[0]


def _round(x: float, digits: int = 3) -> float:
    if math.isnan(x):
        raise ValueError("nan in model matrix summary")
    return round(x, digits)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _template_mean_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["model"], row["template"]), []).append(row)

    out = []
    for (model, template), rs in groups.items():
        out.append({
            "model": model,
            "template": template,
            "score": _mean([row["score"] for row in rs]),
            "strict_pass_rate": _mean([float(row["strict_pass_rate"]) for row in rs]),
            "mean_axis_delta": _mean([float(row["mean_axis_delta"]) for row in rs]),
            "mean_off_axis_problem": _mean([float(row["mean_off_axis_problem"]) for row in rs]),
            "mean_axis_delta_judge_std": _mean([float(row["mean_axis_delta_judge_std"]) for row in rs]),
            "mean_max_style_abs_delta": _mean([float(row["mean_max_style_abs_delta"]) for row in rs]),
            "persona_echo_rate": _mean([float(row["persona_echo_rate"]) for row in rs]),
            "refusal_or_ai_break_rate": _mean([float(row["refusal_or_ai_break_rate"]) for row in rs]),
            "n_axes": len(rs),
        })
    return out


def _summarize(rows: list[dict[str, Any]], group_cols: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[col] for col in group_cols), []).append(row)

    out = []
    for key, rs in groups.items():
        models = sorted({row["model"] for row in rs})
        base = dict(zip(group_cols, key, strict=True))
        model_count = len(models)
        scores = [float(row["score"]) for row in rs]
        out.append({
            "model_count": model_count,
            "score_p25": _round(_p25(scores), 2),
            "score_mean": _round(_mean(scores), 2),
            "score_std": _round(_std(scores), 2),
            "strict_pass_rate_mean": _round(_mean([float(row["strict_pass_rate"]) for row in rs]), 3),
            "strict_pass_rate_std": _round(_std([float(row["strict_pass_rate"]) for row in rs]), 3),
            "axis_delta_mean": _round(_mean([float(row["mean_axis_delta"]) for row in rs]), 3),
            "axis_delta_std": _round(_std([float(row["mean_axis_delta"]) for row in rs]), 3),
            "off_axis_problem_mean": _round(_mean([float(row["mean_off_axis_problem"]) for row in rs]), 3),
            "off_axis_problem_std": _round(_std([float(row["mean_off_axis_problem"]) for row in rs]), 3),
            "judge_std_mean": _round(_mean([float(row["mean_axis_delta_judge_std"]) for row in rs]), 3),
            "style_delta_mean": _round(_mean([float(row["mean_max_style_abs_delta"]) for row in rs]), 3),
            "persona_echo_rate_mean": _round(_mean([float(row["persona_echo_rate"]) for row in rs]), 3),
            "refusal_or_ai_break_rate_mean": _round(
                _mean([float(row["refusal_or_ai_break_rate"]) for row in rs]), 3),
            "models": ",".join(models),
            **base,
        })
    return sorted(out, key=lambda row: row["score_p25"], reverse=True)


def _markdown_text(text: str) -> str:
    if "<!-- instruction following eval, Anthropic/if-2 -->" in text:
        text = text.replace(
            "<!-- instruction following eval, Anthropic/if-2 -->",
            "Anthropic/if-2 instruction-following eval:",
        )
    text = text.replace("{persona}", "`{persona}`")
    text = text.replace("&", "&amp;")
    text = text.replace("<", "&lt;")
    text = text.replace(">", "&gt;")
    text = text.replace("\\", "&#92;")
    text = text.replace("|", "&#124;")
    return text.replace("\n", "<br>")


def _write_markdown(path: Path, template_rows: list[dict[str, Any]], pair_rows: list[dict[str, Any]], top_n: int) -> None:
    top_template_rows = [
        {
            "score p25": f"{row['score_p25']:.2f}",
            "score mean": f"{row['score_mean']:.2f}",
            "score std": f"{row['score_std']:.2f}",
            "pass mean": f"{row['strict_pass_rate_mean']:.2f}",
            "echo rate": f"{row['persona_echo_rate_mean']:.2f}",
            "refusal rate": f"{row['refusal_or_ai_break_rate_mean']:.2f}",
            "models": row["model_count"],
            "template": _markdown_text(row["template"]),
        }
        for row in template_rows[:top_n]
    ]
    lines = [
        "# Refusal Probe Model Matrix",
        "",
        "Scores are model-equal. Each model first averages the two refusal-probe axes per template, then the table reports reliability-sorted template rows across clean model artifacts.",
        "",
        "## All Templates",
        "",
        "`score p25` is the 25th percentile score across the four clean model artifacts. Rows are sorted by this column.",
        "",
        tabulate(top_template_rows, headers="keys", tablefmt="github", disable_numparse=True),
    ]
    path.write_text("\n".join(lines) + "\n")


def _plot(path: Path, rows: list[dict[str, Any]], label_count: int) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 5.6), dpi=180)
    xs = [_clamp01(row["axis_delta_mean"] / 8.0) for row in rows]
    ys = [_clamp01((row["off_axis_problem_mean"] - 1.0) / 6.0) for row in rows]
    colors = ["black" if row["strict_pass_rate_mean"] > 0 else "0.65" for row in rows]

    ax.scatter(xs, ys, s=28, c=colors, alpha=0.82, linewidths=0, zorder=2)
    top_ids = {id(row): i for i, row in enumerate(rows[:label_count], start=1)}
    top_rows = rows[:label_count]
    ax.errorbar(
        [_clamp01(row["axis_delta_mean"] / 8.0) for row in top_rows],
        [_clamp01((row["off_axis_problem_mean"] - 1.0) / 6.0) for row in top_rows],
        xerr=[row["axis_delta_std"] / (8.0 * math.sqrt(row["model_count"])) for row in top_rows],
        yerr=[row["off_axis_problem_std"] / (6.0 * math.sqrt(row["model_count"])) for row in top_rows],
        fmt="none",
        ecolor="0.55",
        elinewidth=0.8,
        zorder=1,
    )
    for row in rows:
        if id(row) not in top_ids:
            continue
        x = _clamp01(row["axis_delta_mean"] / 8.0)
        y = _clamp01((row["off_axis_problem_mean"] - 1.0) / 6.0)
        ax.text(
            x,
            y,
            str(top_ids[id(row)]),
            ha="center",
            va="center",
            fontsize=6.5,
            color="white",
            zorder=3,
        )

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("mean on-axis movement")
    ax.set_ylabel("mean off-axis confounding")
    ax.set_title("Refusal probe templates across clean model artifacts", fontsize=10)
    ax.text(
        1.0,
        -0.13,
        "error bars are model SEM; point numbers match the first table rows",
        transform=ax.transAxes,
        ha="right",
        fontsize=8,
    )
    ax.grid(True, color="0.9", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair-stats", nargs="+", type=Path, default=DEFAULT_PAIR_STATS)
    ap.add_argument("--out-prefix", type=Path, default=DEFAULT_OUT_PREFIX)
    ap.add_argument("--top-n", type=int, default=999)
    args = ap.parse_args()

    rows = []
    for path in args.pair_stats:
        model = _model_name(path)
        model_rows = []
        for row in _read_jsonl(path):
            model_rows.append({**row, "model": model, "score": _score(row)})
        if len(model_rows) != 190:
            raise ValueError(f"{path} has {len(model_rows)} rows, expected 190")
        rows.extend(model_rows)

    template_rows = _summarize(_template_mean_rows(rows), ["template"])
    pair_rows = _summarize(rows, ["template", "persona_pair"])
    expected_models = len(args.pair_stats)
    if any(row["model_count"] != expected_models for row in template_rows + pair_rows):
        raise ValueError("at least one summary row is missing a model")

    prefix = args.out_prefix
    _write_jsonl(prefix.with_name(prefix.name + "_template_model_summary.jsonl"), template_rows)
    _write_csv(prefix.with_name(prefix.name + "_template_model_summary.csv"), template_rows)
    _write_jsonl(prefix.with_name(prefix.name + "_template_pair_model_summary.jsonl"), pair_rows)
    _write_csv(prefix.with_name(prefix.name + "_template_pair_model_summary.csv"), pair_rows)
    _write_markdown(prefix.with_name(prefix.name + "_model_matrix_summary.md"), template_rows, pair_rows, args.top_n)
    _plot(prefix.with_name(prefix.name + "_model_matrix.png"), template_rows, label_count=10)
    print(f"models={expected_models} templates={len(template_rows)} template_pairs={len(pair_rows)}")
    print(prefix.with_name(prefix.name + "_model_matrix_summary.md"))
    print(prefix.with_name(prefix.name + "_model_matrix.png"))


if __name__ == "__main__":
    main()

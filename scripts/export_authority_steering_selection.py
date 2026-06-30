"""Export winning Authority-axis ingredients from validator artifacts."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from tabulate import tabulate

ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n")


def score_row(row: dict) -> float:
    if "error" in row:
        return -1.0
    return 100.0 * float(row["on_axis_frac"]) * (1.0 - float(row["off_axis_problem_frac"]))


def choose_stage_a(stage_a: dict, axis_filter: str | None) -> dict:
    summary = stage_a["summary"]
    if axis_filter is not None:
        summary = [row for row in summary if row["axis"] == axis_filter]
    if not summary:
        raise ValueError(f"stage A has no summary rows for axis_filter={axis_filter!r}")
    ranked = sorted(
        summary,
        key=lambda row: (
            row["recommended"],
            row["strict_pass_rate"],
            row["mean_axis_delta"],
            -row["mean_off_axis_problem"],
            -row["mean_max_style_abs_delta"],
            -row["persona_echo_rate"],
            -row["refusal_or_ai_break_rate"],
        ),
        reverse=True,
    )
    return ranked[0]


def write_stage_b_inputs(stage_a_path: Path, out_dir: Path, axis_filter: str | None) -> dict:
    stage_a = json.loads(stage_a_path.read_text())
    winner = choose_stage_a(stage_a, axis_filter)
    axis_id = winner["axis"]
    template = winner["template"]
    axes = [axis for axis in stage_a["axes"] if axis["id"] == axis_id]
    if len(axes) != 1:
        raise ValueError(f"expected one winning axis {axis_id!r}, found {len(axes)}")
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "stage_b_axis.jsonl", axes)
    (out_dir / "stage_b_template.txt").write_text(template + "\n")
    (out_dir / "stage_a_winner.json").write_text(json.dumps(winner, indent=2))
    return winner


def select_stage_b(
    stage_b_path: Path,
    out_dir: Path,
    keep_per_source: int,
    strict_only: bool,
    min_score: float,
) -> list[dict]:
    artifact = json.loads(stage_b_path.read_text())
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in artifact["results"]:
        if "error" in row:
            continue
        if strict_only and not row["strict_pass"]:
            continue
        if score_row(row) < min_score:
            continue
        grouped[str(row["source"])].append(row)

    selected: list[dict] = []
    score_rows: list[dict] = []
    for source, rows in sorted(grouped.items()):
        ranked = sorted(
            rows,
            key=lambda row: (
                row["strict_pass"],
                score_row(row),
                float(row["axis_delta"]),
                -float(row["confound_judgment"]["off_axis_problem_likert"]),
                -float(row["max_style_abs_delta"]),
            ),
            reverse=True,
        )
        for rank, row in enumerate(ranked, start=1):
            score_rows.append({
                "source": source,
                "rank": rank,
                "selected": rank <= keep_per_source,
                "scenario_id": row["scenario_id"],
                "score": round(score_row(row), 2),
                "strict_pass": row["strict_pass"],
                "axis_delta": row["axis_delta"],
                "off_axis_problem": row["confound_judgment"]["off_axis_problem_likert"],
                "max_style_abs_delta": row["max_style_abs_delta"],
                "prompt": row["prompt"],
            })
        for row in ranked[:keep_per_source]:
            selected.append({
                "id": row["scenario_id"],
                "prompt": row["prompt"],
                "source": row["source"],
                "config": row.get("config"),
                "self_contained": True,
                "selection_score": round(score_row(row), 4),
                "axis_delta": row["axis_delta"],
                "off_axis_problem": row["confound_judgment"]["off_axis_problem_likert"],
                "strict_pass": row["strict_pass"],
            })

    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "selected_scenarios.jsonl", selected)
    with (out_dir / "scenario_scores.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(score_rows[0]))
        writer.writeheader()
        writer.writerows(score_rows)
    examples = []
    for row in sorted(score_rows, key=lambda r: (r["source"], r["rank"])):
        if row["rank"] <= 2:
            examples.append(
                f"## {row['source']} / {row['scenario_id']}\n\n"
                f"score={row['score']} axis_delta={row['axis_delta']} "
                f"off_axis={row['off_axis_problem']} strict_pass={row['strict_pass']}\n\n"
                f"{row['prompt']}\n"
            )
    (out_dir / "selected_examples.md").write_text("\n".join(examples))
    return selected


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage-a", type=Path)
    ap.add_argument("--stage-b", type=Path)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "out/authority_selection")
    ap.add_argument("--keep-per-source", type=int, default=10)
    ap.add_argument("--axis-filter", type=str)
    ap.add_argument("--strict-only", action="store_true")
    ap.add_argument("--min-score", type=float, default=0.0)
    args = ap.parse_args()

    if args.stage_a is None and args.stage_b is None:
        raise ValueError("pass --stage-a and/or --stage-b")

    if args.stage_a is not None:
        winner = write_stage_b_inputs(args.stage_a, args.out_dir, args.axis_filter)
        print("Stage A winner:")
        print(tabulate([winner], headers="keys", tablefmt="github", floatfmt=".3f"))

    if args.stage_b is not None:
        selected = select_stage_b(args.stage_b, args.out_dir, args.keep_per_source, args.strict_only, args.min_score)
        counts: dict[str, int] = {}
        for row in selected:
            counts[row["source"]] = counts.get(row["source"], 0) + 1
        print("\nSelected scenarios:")
        print(tabulate(
            [{"source": source, "n": counts[source]} for source in sorted(counts)],
            headers="keys",
            tablefmt="github",
        ))
        print(f"total={len(selected)}")


if __name__ == "__main__":
    main()

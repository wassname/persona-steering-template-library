"""Plot measured on-axis movement against off-axis confounding.

The default input is the built Hugging Face parquet table:

    uv run python scripts/plot_on_off_axis.py /tmp/persona-steering-template-library-hf/parquet/main.parquet
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import textwrap
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pyarrow.parquet as pq


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".parquet":
        return pq.read_table(path).to_pylist()
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _read_all_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        rows.extend(_read_rows(path))
    return rows


def _as_point(row: dict[str, Any]) -> dict[str, Any]:
    on_axis = row.get("on_axis")
    if on_axis is None:
        on_axis = _clamp01(float(row.get("mean_axis_delta") or 0.0) / 8.0)
    off_axis = row.get("off_axis")
    if off_axis is None:
        off_axis = _clamp01((float(row.get("mean_off_axis_problem") or 7.0) - 1.0) / 6.0)
    point_id = row.get("contrast") or row.get("persona_pair") or ""
    template = row.get("template") or row.get("template_jinja") or ""
    return {
        "x": float(on_axis),
        "y": float(off_axis),
        "score": float(row.get("score") or 100.0 * float(on_axis) * (1.0 - float(off_axis))),
        "id": str(point_id),
        "template": str(template),
        "recommended": bool(row.get("recommended")),
    }


def _aggregate_points(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[float, float], list[dict[str, Any]]] = defaultdict(list)
    for point in points:
        groups[(point["x"], point["y"])].append(point)

    out = []
    for (x, y), rows in groups.items():
        rows.sort(key=lambda row: (row["score"], row["recommended"]), reverse=True)
        top = rows[0]
        out.append({
            "x": x,
            "y": y,
            "score": max(row["score"] for row in rows),
            "id": top["id"],
            "template": top["template"],
            "recommended": any(row["recommended"] for row in rows),
            "count": len(rows),
            "labels": [f'{row["id"]}: "{row["template"]}"' for row in rows],
        })
    return out


def _label_points(points: list[dict[str, Any]], n: int, rightmost_n: int) -> list[dict[str, Any]]:
    if len(points) <= n:
        return points
    high_score = sorted(points, key=lambda p: p["score"], reverse=True)[: max(2, n // 2)]
    high_off_axis = sorted(points, key=lambda p: (p["y"], p["x"]), reverse=True)[: n]
    rightmost = sorted(points, key=lambda p: (p["x"], -p["y"], p["score"]), reverse=True)[:rightmost_n]
    out = []
    seen_labels = set()
    seen_cells = set()
    for point in high_score + high_off_axis + rightmost:
        label_key = f'{point["id"]}: "{point["template"]}"'
        cell_key = (round(point["x"], 1), round(point["y"], 1))
        if label_key not in seen_labels and cell_key not in seen_cells:
            out.append(point)
            seen_labels.add(label_key)
            seen_cells.add(cell_key)
    return out[: max(n, rightmost_n)]


def _place_label(i: int, point: dict[str, Any]) -> tuple[float, float, str, str]:
    dx = 0.018
    dy = [0.035, -0.05, 0.075, -0.09, 0.115, -0.13, 0.16, -0.175][i % 8]
    x = min(0.98, point["x"] + dx) if point["x"] < 0.9 else max(0.05, point["x"] - 0.02)
    y = min(0.98, max(0.02, point["y"] + dy))
    ha = "left" if point["x"] < 0.9 else "right"
    return x, y, ha, "center"


def _short_template(text: str, width: int = 52) -> str:
    text = text.replace("{{ persona }}", "{persona}").replace("\n", " ")
    text = " ".join(text.split())
    if len(text) <= width:
        return text
    keep = max(8, (width - 5) // 2)
    return f"{text[:keep]} ... {text[-keep:]}"


def _short_label(point: dict[str, Any]) -> str:
    text = f'{point["id"]}: "{_short_template(point["template"])}"'
    return textwrap.fill(text, width=38)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=Path("out/on_off_axis.png"))
    ap.add_argument("--label-count", type=int, default=10)
    ap.add_argument("--label-rightmost", type=int, default=5)
    args = ap.parse_args()

    raw_points = [_as_point(row) for row in _read_all_rows(args.input)]
    raw_points = [p for p in raw_points if p["id"]]
    points = _aggregate_points(raw_points)
    labels = _label_points(points, args.label_count, args.label_rightmost)

    fig, ax = plt.subplots(figsize=(8.0, 5.6), dpi=180)
    ax.scatter(
        [p["x"] for p in points],
        [p["y"] for p in points],
        s=[26 + 12 * p["count"] for p in points],
        c=["black" if p["recommended"] else "0.55" for p in points],
        alpha=0.82,
        linewidths=0,
    )
    for point in points:
        if point["count"] > 1:
            ax.text(
                point["x"],
                point["y"],
                str(point["count"]),
                ha="center",
                va="center",
                fontsize=6.5,
                color="white" if point["recommended"] else "0.1",
            )
    for i, point in enumerate(labels):
        x, y, ha, va = _place_label(i, point)
        count_suffix = f" [{point['count']}]" if point["count"] > 1 else ""
        ax.annotate(
            _short_label(point) + count_suffix,
            xy=(point["x"], point["y"]),
            xytext=(x, y),
            textcoords="data",
            ha=ha,
            va=va,
            fontsize=6.5,
            color="0.15",
            arrowprops={"arrowstyle": "-", "color": "0.65", "lw": 0.55},
        )

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("on-axis movement")
    ax.set_ylabel("off-axis confounding")
    ax.set_title("Persona template cells: move the intended axis, avoid confounds", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, color="0.9", linewidth=0.6)
    ax.text(1.0, -0.13, "better is lower-right", transform=ax.transAxes, ha="right", fontsize=8)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    print(args.out)


if __name__ == "__main__":
    main()

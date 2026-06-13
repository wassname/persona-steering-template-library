"""Plot measured on-axis movement against off-axis confounding.

The default input is the built Hugging Face parquet table:

    uv run python scripts/plot_on_off_axis.py /tmp/persona-steering-template-library-hf/parquet/main.parquet
"""
from __future__ import annotations

import argparse
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


def _as_point(row: dict[str, Any]) -> dict[str, Any]:
    on_axis = row.get("on_axis")
    if on_axis is None:
        on_axis = _clamp01(float(row.get("mean_axis_delta") or 0.0) / 8.0)
    off_axis = row.get("off_axis")
    if off_axis is None:
        off_axis = _clamp01((float(row.get("mean_off_axis_problem") or 7.0) - 1.0) / 6.0)
    label = row.get("contrast") or row.get("persona_pair") or ""
    template = row.get("template") or row.get("template_jinja") or ""
    return {
        "x": float(on_axis),
        "y": float(off_axis),
        "score": float(row.get("score") or 100.0 * float(on_axis) * (1.0 - float(off_axis))),
        "label": f"{label}: {template}".strip(": "),
        "recommended": bool(row.get("recommended")),
    }


def _label_points(points: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    if len(points) <= n:
        return points
    high_score = sorted(points, key=lambda p: p["score"], reverse=True)[: max(2, n // 2)]
    high_off_axis = sorted(points, key=lambda p: (p["y"], p["x"]), reverse=True)[: n]
    out = []
    seen_labels = set()
    seen_cells = set()
    for point in high_score + high_off_axis:
        label_key = point["label"]
        cell_key = (round(point["x"], 1), round(point["y"], 1))
        if label_key not in seen_labels and cell_key not in seen_cells:
            out.append(point)
            seen_labels.add(label_key)
            seen_cells.add(cell_key)
    return out[:n]


def _place_label(i: int, point: dict[str, Any]) -> tuple[float, float, str, str]:
    dx = 0.018
    dy = [0.035, -0.05, 0.075, -0.09, 0.115, -0.13][i % 6]
    x = min(0.98, point["x"] + dx)
    y = min(0.98, max(0.02, point["y"] + dy))
    return x, y, "left", "center"


def _short_label(text: str) -> str:
    text = text.replace("{{ persona }}", "{persona}")
    return textwrap.fill(textwrap.shorten(text, width=74, placeholder="..."), width=38)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path)
    ap.add_argument("--out", type=Path, default=Path("out/on_off_axis.png"))
    ap.add_argument("--label-count", type=int, default=4)
    args = ap.parse_args()

    points = [_as_point(row) for row in _read_rows(args.input)]
    points = [p for p in points if p["label"]]
    labels = _label_points(points, args.label_count)

    fig, ax = plt.subplots(figsize=(8.0, 5.6), dpi=180)
    ax.scatter(
        [p["x"] for p in points],
        [p["y"] for p in points],
        s=[42 if p["recommended"] else 24 for p in points],
        c=["black" if p["recommended"] else "0.55" for p in points],
        alpha=0.82,
        linewidths=0,
    )
    for i, point in enumerate(labels):
        x, y, ha, va = _place_label(i, point)
        ax.annotate(
            _short_label(point["label"]),
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

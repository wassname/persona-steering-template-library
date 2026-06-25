from __future__ import annotations

import html
from pathlib import Path
import textwrap
from typing import Any

import plotly.graph_objects as go

import docs_results

MAIN_PNG = docs_results.ROOT / "out/on_off_axis.png"
MAIN_SVG = docs_results.ROOT / "out/on_off_axis.svg"


def _wrap_hover(text: str, width: int = 62) -> str:
    escaped = html.escape(" ".join(text.split()))
    return "<br>".join(
        textwrap.wrap(escaped, width=width, break_long_words=True, break_on_hyphens=False))


def main_plot_rows(path: Path = docs_results.NORMAL_TEMPLATE_PAIR_STATS) -> list[dict[str, Any]]:
    return docs_results.mean_template_rows(docs_results.read_jsonl(path))


def template_scatter(rows: list[dict[str, Any]] | None = None) -> go.Figure:
    rows = main_plot_rows() if rows is None else rows
    top_rank = {row["template"]: i for i, row in enumerate(rows[:10], start=1)}
    text = [str(top_rank[row["template"]]) if row["template"] in top_rank else "" for row in rows]
    hover = [
        "<br>".join([
            f"<b>{_wrap_hover(row['template'])}</b>",
            f"rank: {i}",
            f"score t: {row['score_t']:.2f}",
            f"score mean: {row['score_mean']:.2f}",
            f"axis delta: {row['axis_delta']:.2f}",
            f"off-axis problem: {row['off_axis_problem']:.2f}",
            f"judge std: {row['judge_std']:.2f}",
            f"cells: {row['n_cells']}",
        ])
        for i, row in enumerate(rows, start=1)
    ]
    fig = go.Figure(
        data=go.Scatter(
            x=[row["on_axis"] for row in rows],
            y=[row["off_axis"] for row in rows],
            mode="markers+text",
            text=text,
            textposition="middle center",
            textfont={"size": 9, "color": "white"},
            customdata=hover,
            hovertemplate="%{customdata}<extra></extra>",
            marker={
                "size": 10,
                "color": [row["score_t"] for row in rows],
                "colorscale": "Cividis",
                "showscale": True,
                "colorbar": {"title": "score t"},
                "line": {"width": 0.5, "color": "white"},
                "opacity": 0.9,
            },
        )
    )
    fig.update_layout(
        autosize=True,
        width=960,
        height=620,
        template="plotly_white",
        margin={"l": 68, "r": 24, "t": 28, "b": 66},
        xaxis={
            "title": "on-axis movement, higher is better",
            "range": [-0.02, 1.02],
            "gridcolor": "rgba(0,0,0,0.08)",
        },
        yaxis={
            "title": "off-axis confounding, lower is better",
            "range": [-0.02, 1.02],
            "gridcolor": "rgba(0,0,0,0.08)",
        },
        annotations=[{
            "text": "normal pilot scenarios; one point per measured template",
            "xref": "paper",
            "yref": "paper",
            "x": 1.0,
            "y": -0.13,
            "showarrow": False,
            "font": {"size": 11, "color": "rgba(0,0,0,0.62)"},
        }],
    )
    return fig


def write_main_plot_assets() -> None:
    fig = template_scatter()
    MAIN_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(MAIN_PNG, width=960, height=620, scale=2)
    fig.write_image(MAIN_SVG, width=960, height=620)

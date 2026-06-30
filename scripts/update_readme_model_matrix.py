from __future__ import annotations

import html
import json
import os
from pathlib import Path

from tabulate import tabulate

import docs_results

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "data/results/model_matrix/refusal_probe_seed24_n1_template_model_summary.jsonl"
PAIR_SUMMARY = ROOT / "data/results/model_matrix/refusal_probe_seed24_n1_template_pair_model_summary.jsonl"
ANTHROPIC_IF2_SHORT_LABEL = "Anthropic IF-2: role-play as {persona}"


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _markdown_text(text: str) -> str:
    text = docs_results.display_template_text(text)
    text = text.replace("{persona}", "`{persona}`")
    text = text.replace("&", "&amp;")
    text = text.replace("<", "&lt;")
    text = text.replace(">", "&gt;")
    text = text.replace("\\", "&#92;")
    text = text.replace("|", "&#124;")
    return text.replace("\n", "<br>")


def _appendix_table(rows: list[dict]) -> str:
    table_rows = [
        {
            "score t": f"{row['score_t']:.2f}",
            "score mean": f"{row['score_mean']:.2f}",
            "score std": f"{row['score_std']:.2f}",
            "template": _markdown_text(row["template"]),
        }
        for row in rows
    ]
    return tabulate(table_rows, headers="keys", tablefmt="github", disable_numparse=True)


def _appendix_intro() -> str:
    return "\n\n".join([
        "## Appendix: refusal-pole probe",
        (
            "This is a rejected-pole slice: it keeps the template and suffix sweep "
            "unfiltered, then evaluates persona pairs whose negative/rejected pole is "
            "refusal-prone or harm-adjacent. It is not the main template result, because "
            "it does not cover all persona pairs."
        ),
        (
            "Why include it? These negative poles can collapse into generic safety refusal, "
            "AI-role breaks, or persona echo instead of the intended behavioral contrast. "
            "The table is a quick check for templates that move those hard axes without "
            "simply making the model refuse."
        ),
        (
            "`refusal_or_ai_break_rate` is only an output audit column: it marks completions "
            "that refused or broke AI role, and is not used to select this data slice."
        ),
        (
            "The generated full audit table includes strict-pass, echo, and refusal columns: "
            "[docs/results/model_matrix/refusal_probe_seed24_n1_model_matrix_summary.md]"
            "(docs/results/model_matrix/refusal_probe_seed24_n1_model_matrix_summary.md)."
        ),
    ])


def _appendix_block(summary_path: Path) -> str:
    rows = _read_jsonl(summary_path)
    return "\n\n".join([
        _appendix_intro(),
        _appendix_table(rows),
    ])


def _template_display_text(text: str) -> str:
    text = docs_results.display_template_text(text)
    text = " ".join(text.split())
    if (
        text
        == "Anthropic/if-2 instruction-following eval: Assist the user by role playing as {persona}"
    ):
        return ANTHROPIC_IF2_SHORT_LABEL
    return text.replace("{persona}", "{persona}")


def _table_styles() -> str:
    return """
<style>
.refusal-table-wrap {
  margin: 1rem 0 2rem;
}
.refusal-table-wrap table.dataTable {
  width: 100% !important;
}
.refusal-table-wrap table.dataTable td,
.refusal-table-wrap table.dataTable th {
  vertical-align: top;
}
.refusal-table-wrap table.dataTable td:last-child {
  white-space: normal;
  min-width: min(42rem, 72vw);
}
</style>
"""


def _html_heading(title: str, body: str) -> str:
    return "\n".join([
        f"<h3>{html.escape(title)}</h3>",
        f"<p>{html.escape(body)}</p>",
    ])


def _template_table_rows(rows: list[dict]) -> list[dict]:
    return [
        {
            "score t": row["score_t"],
            "score mean": row["score_mean"],
            "score std": row["score_std"],
            "pass": row["strict_pass_rate_mean"],
            "echo": row["persona_echo_rate_mean"],
            "refusal": row["refusal_or_ai_break_rate_mean"],
            "template": _template_display_text(row["template"]),
        }
        for row in rows
    ]


def _pair_table_rows(rows: list[dict]) -> list[dict]:
    return [
        {
            "score t": row["score_t"],
            "score mean": row["score_mean"],
            "score std": row["score_std"],
            "pass": row["strict_pass_rate_mean"],
            "echo": row["persona_echo_rate_mean"],
            "refusal": row["refusal_or_ai_break_rate_mean"],
            "persona_pair": row["persona_pair"],
            "template": _template_display_text(row["template"]),
        }
        for row in rows
    ]


def _datatable_html(rows: list[dict], table_id: str) -> str:
    import polars as pl
    from itables import to_html_datatable

    df = pl.DataFrame(rows)
    return "\n".join([
        f'<div id="{table_id}" class="refusal-table-wrap">',
        to_html_datatable(
            df,
            classes="display compact cell-border stripe",
            display_logo_when_loading=False,
            paging=True,
            pageLength=25,
            lengthMenu=[10, 25, 50, 100, -1],
            ordering=True,
            scrollX=True,
            autoWidth=False,
            show_dtypes=False,
            showIndex=False,
            maxBytes=1_000_000,
        ),
        "</div>",
    ])


def _interactive_tables_block(summary_path: Path, pair_summary_path: Path) -> str:
    template_rows = _read_jsonl(summary_path)
    pair_rows = _read_jsonl(pair_summary_path)
    refusal_hit_pairs = sorted({
        row["persona_pair"]
        for row in pair_rows
        if float(row["refusal_or_ai_break_rate_mean"]) > 0.0
    })
    refusal_pair_rows = [
        row for row in pair_rows
        if row["persona_pair"] in refusal_hit_pairs
    ]

    return "\n\n".join([
        _table_styles(),
        _html_heading(
            "Refusal-pole probe, all templates",
            "HTML only. Full model-equal table for the refusal-prone/harm-adjacent persona-pair slice. Sort by score t, refusal, echo, or pass; search for a template phrase.",
        ),
        _datatable_html(_template_table_rows(template_rows), "refusal-template-table"),
        _html_heading(
            "Persona pairs with refusal audit hits, all templates retained",
            (
                "This filters persona pairs to those with any refusal-or-AI-break audit hit, "
                f"then keeps every template for those pairs. Current pairs: {', '.join(refusal_hit_pairs)}."
            ),
        ),
        _datatable_html(_pair_table_rows(refusal_pair_rows), "refusal-pair-table"),
    ])


def results_block() -> str:
    if os.environ["PSTL_DOC_TARGET"] == "html":
        return _interactive_tables_block(SUMMARY, PAIR_SUMMARY)
    return "\n".join([
        "Full refusal-pole audit table: "
        "[docs/results/model_matrix/refusal_probe_seed24_n1_model_matrix_summary.md]"
        "(docs/results/model_matrix/refusal_probe_seed24_n1_model_matrix_summary.md)."
    ])


def appendix_block() -> str:
    if os.environ["PSTL_DOC_TARGET"] == "html":
        return _appendix_intro()
    return _appendix_block(SUMMARY)


def main() -> None:
    print(appendix_block())


if __name__ == "__main__":
    main()

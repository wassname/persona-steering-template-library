from __future__ import annotations

import json
from pathlib import Path

from tabulate import tabulate


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "out/model_matrix/refusal_probe_seed24_n1_template_model_summary.jsonl"


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


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


def _appendix_block(summary_path: Path) -> str:
    rows = _read_jsonl(summary_path)
    return "\n\n".join([
        "## Appendix: Refusal-Pole Probe",
        (
            "This is a rejected-pole slice: it keeps the template and suffix sweep "
            "unfiltered, then evaluates persona pairs whose negative/rejected pole is "
            "refusal-prone or harm-adjacent. It is not the main template result, because "
            "it does not cover all persona pairs."
        ),
        (
            "Why include it? These negative poles can collapse into generic safety refusal, "
            "AI-role breaks, or persona echo instead of the intended behavioral contrast. "
            "This plot is a quick check for templates that move those hard axes without "
            "simply making the model refuse."
        ),
        "![refusal-pole probe](./out/model_matrix/refusal_probe_seed24_n1_model_matrix.png)",
        (
            "Caption: each dot is one template, averaged over the two refusal-probe axes "
            "and four clean models. Right is more on-axis movement; lower is less off-axis "
            "confounding. Numbered dots are the first rows of the appendix table."
        ),
        (
            "`refusal_or_ai_break_rate` is only an output audit column: it marks completions "
            "that refused or broke AI role, and is not used to select this data slice."
        ),
        (
            "Interactive hover plot: "
            "[GitHub Pages](https://wassname.github.io/persona-steering-template-library/)."
        ),
        (
            "The generated full audit table includes strict-pass, echo, and refusal columns: "
            "[out/model_matrix/refusal_probe_seed24_n1_model_matrix_summary.md]"
            "(out/model_matrix/refusal_probe_seed24_n1_model_matrix_summary.md)."
        ),
        _appendix_table(rows),
    ])


def main() -> None:
    print(_appendix_block(SUMMARY))


if __name__ == "__main__":
    main()

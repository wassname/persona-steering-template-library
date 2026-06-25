from __future__ import annotations

import argparse
import json
from pathlib import Path

from tabulate import tabulate


ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
SUMMARY = ROOT / "out/model_matrix/refusal_probe_seed24_n1_template_model_summary.jsonl"

START = "<!-- model-matrix:start -->"
END = "<!-- model-matrix:end -->"


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _markdown_text(text: str) -> str:
    text = text.replace("{persona}", "`{persona}`")
    text = text.replace("&", "&amp;")
    text = text.replace("<", "&lt;")
    text = text.replace(">", "&gt;")
    text = text.replace("\\", "&#92;")
    text = text.replace("|", "&#124;")
    return text.replace("\n", "<br>")


def _table(rows: list[dict], top_n: int) -> str:
    table_rows = [
        {
            "score mean": f"{row['score_mean']:.2f}",
            "score std": f"{row['score_std']:.2f}",
            "pass mean": f"{row['strict_pass_rate_mean']:.2f}",
            "axis mean": f"{row['axis_delta_mean']:.2f}",
            "off-axis mean": f"{row['off_axis_problem_mean']:.2f}",
            "echo rate": f"{row['persona_echo_rate_mean']:.2f}",
            "refusal rate": f"{row['refusal_or_ai_break_rate_mean']:.2f}",
            "template": _markdown_text(row["template"]),
        }
        for row in rows[:top_n]
    ]
    return tabulate(table_rows, headers="keys", tablefmt="github", disable_numparse=True)


def _block(summary_path: Path) -> str:
    rows = _read_jsonl(summary_path)
    return "\n\n".join([
        "## Refusal Probe Model Matrix",
        (
            "I also ran the newer roleplay, safety-lab, theatre/treatment, anthropology, and "
            "multilingual templates on a two-axis refusal probe across four clean generator "
            "artifacts: `google/gemma-2-27b-it`, `google/gemma-3-4b-it`, "
            "`qwen/qwen3.6-flash`, and `ibm-granite/granite-4.1-8b`."
        ),
        (
            "This table reports mean and sample std across models. Each model first averages "
            "the two probe axes for a template, so this is model-equal rather than row-equal. "
            "High std, persona echo, and refusal rate are warnings, not secondary scores."
        ),
        "![refusal probe model matrix](./out/model_matrix/refusal_probe_seed24_n1_model_matrix.png)",
        "Top model-matrix templates:",
        _table(rows, top_n=10),
        (
            "Interpretation: some explicit judgment framings and red-team/eval framings move "
            "the hard axis more often than the gentle templates, but they frequently do so "
            "with persona echo or model-specific behavior. The cleanest-looking single-axis "
            "cells were often `protocol_harm`, so treat the top rows as rerun candidates "
            "rather than settled reusable defaults."
        ),
        "Excluded attempted models:",
        "\n".join([
            "| model | result |",
            "|---|---|",
            "| `google/gemma-2-9b-it` | OpenRouter returned no endpoints for all 190 cells. |",
            "| `openai/gpt-oss-120b` | OpenRouter returned `Reasoning is mandatory for this endpoint and cannot be disabled` for all 190 cells. |",
            "| `deepseek/deepseek-v4-flash` | Reproduced 3 empty-generation cells out of 190, so excluded from aggregate instead of averaging missing data. |",
        ]),
        (
            "Full generated table:\n"
            "[`out/model_matrix/refusal_probe_seed24_n1_model_matrix_summary.md`](out/model_matrix/refusal_probe_seed24_n1_model_matrix_summary.md)."
        ),
    ])


def replace_block(readme: str, block: str) -> str:
    wrapped = f"{START}\n{block}\n{END}"
    if START in readme:
        before, rest = readme.split(START)
        _, after = rest.split(END)
        return f"{before}{wrapped}{after}"

    heading = "\n## Refusal Probe Model Matrix\n"
    next_heading = "\n## Score\n"
    before, rest = readme.split(heading)
    _, after = rest.split(next_heading, maxsplit=1)
    return f"{before}\n{wrapped}\n{next_heading}{after}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--readme", type=Path, default=README)
    ap.add_argument("--summary", type=Path, default=SUMMARY)
    args = ap.parse_args()

    readme = args.readme.read_text()
    args.readme.write_text(replace_block(readme, _block(args.summary)))
    print(args.readme)


if __name__ == "__main__":
    main()

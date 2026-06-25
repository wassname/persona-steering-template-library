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


def _table(rows: list[dict], top_n: int) -> str:
    table_rows = [
        {
            "score p25": f"{row['score_p25']:.2f}",
            "score mean": f"{row['score_mean']:.2f}",
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
            "Each model first averages the two probe axes for a template, so this is "
            "model-equal rather than row-equal. `score p25` is the headline sort: it is "
            "the 25th percentile score across the four clean model artifacts, so a template "
            "has to work on more than one model to rank well."
        ),
        "![refusal probe model matrix](./out/model_matrix/refusal_probe_seed24_n1_model_matrix.png)",
        (
            "Caption: this is a template overview, not a persona plot. Each dot is one template, "
            "averaged over the two refusal-probe axes and four clean models. Right is more "
            "on-axis movement; lower is less off-axis confounding. Black dots have at least one "
            "strict-pass template-axis cell; grey dots have none. Numbered dots are the first "
            "rows of the table."
        ),
        "Model-matrix templates, all rows:",
        _table(rows, top_n=len(rows)),
        (
            "Interpretation: some explicit judgment framings and red-team/eval framings move "
            "the hard axis more often than the gentle templates. The cleanest-looking single-axis "
            "cells were often `protocol_harm`, so treat the high rows as rerun candidates "
            "rather than settled reusable defaults."
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

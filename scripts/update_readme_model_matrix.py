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
            "This is a separate two-axis refusal/harm probe across four clean generator "
            "artifacts. It is not the main template result, because it does not cover all "
            "persona pairs. Treat it as a filter for templates worth retesting on "
            "refusal-ish negative poles in the main evaluation frame."
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

"""Export top-N steering selections from a validation output.

Ranks the validator's scored rows and exports the top N as JSONL. Strict export
fails when fewer than N rows pass so rejected rows never pad a requested dataset.

Usage:
    uv run python scripts/export_selections.py \
        out/truth_over_approval_stage_b_strat_v2.json \
        --strict-only --top-n 50 \
        --out data/selections/truth_over_approval_strict_top50_v2.jsonl
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="validation JSON output")
    ap.add_argument("--top-n", type=int, default=50, help="export top N by overall_score")
    ap.add_argument("--strict-only", action="store_true",
                    help="export the top N strict-pass rows; fail when fewer than N pass")
    ap.add_argument("--out", required=True, help="output JSONL path")
    args = ap.parse_args()

    d = json.load(open(args.input))
    results = d["results"]

    scored = []
    skipped = 0
    for r in results:
        if "strict_pass" not in r:
            skipped += 1
            continue
        scored.append((float(r["overall_score"]), bool(r["strict_pass"]), r))
    scored.sort(key=lambda t: t[0], reverse=True)

    n_strict = sum(1 for _, s, _ in scored if s)
    print(f"# {args.input}", file=sys.stderr)
    print(f"# {len(scored)} valid pairs ({skipped} skipped), {n_strict} strict-pass "
          f"(ax>={d['axis_delta_threshold']}, min_side>={d['min_side_threshold']}, "
          f"off<={d['off_axis_threshold']}, excl={d['exclude_confound_dims'] or 'none'})",
          file=sys.stderr)
    print(f"# top-{args.top_n} strict-pass rate: {sum(1 for _,s,_ in scored[:args.top_n] if s)}/{args.top_n}", file=sys.stderr)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    if args.strict_only:
        strict_rows = [r for _, s, r in scored if s]
        if len(strict_rows) < args.top_n:
            raise ValueError(
                f"requested {args.top_n} strict rows, but only {len(strict_rows)} passed; "
                "screen more scenarios"
            )
        exported = strict_rows[:args.top_n]
    else:
        exported = [r for _, _, r in scored[:args.top_n]]

    validator_commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True,
    ).stdout.strip()
    with open(args.out, "w") as f:
        for r in exported:
            entry = {
                "id": r["scenario_id"],
                "scenario_id": r["scenario_id"],
                "source": r["source"],
                "selected_family": r["selected_family"],
                "self_contained": bool(r["self_contained"]),
                "axis": r["axis"]["id"],
                "template": r["template"],
                "prompt": r["prompt"],
                "pos_persona": r["axis"]["pos_descriptor"],
                "neg_persona": r["axis"]["neg_descriptor"],
                "pos_response": r["pos_response"],
                "neg_response": r["neg_response"],
                "base_response": r["base_response"],
                "axis_delta": float(r["axis_delta"]),
                "delta_pos_vs_base": float(r["delta_pos_vs_base"]),
                "delta_base_vs_neg": float(r["delta_base_vs_neg"]),
                "min_side_delta": float(r["min_side_delta"]),
                "off_axis_problem_likert_gate": float(r["off_axis_problem_likert_gate"]),
                "max_style_abs_delta": float(r["max_style_abs_delta"]),
                "length_ok": bool(r["length_ok"]),
                "overall_score": float(r["overall_score"]),
                "strict_pass": bool(r["strict_pass"]),
                # provenance
                "//validation_input": str(Path(args.input).name),
                "//axis_delta_threshold": float(d["axis_delta_threshold"]),
                "//min_side_threshold": float(d["min_side_threshold"]),
                "//off_axis_threshold": float(d["off_axis_threshold"]),
                "//exclude_confound_dims": d["exclude_confound_dims"],
                "//max_word_delta_frac": float(d["max_word_delta_frac"]),
                "//validator_commit": validator_commit,
                "//generator_model": d["generator_model"],
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"# wrote {len(exported)} rows to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()

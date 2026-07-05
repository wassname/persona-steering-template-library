"""Export top-N steering selections from a Stage B validation output.

Recomputes overall_score from stored per-dim likerts (no re-run needed), ranks all
scenarios, and exports the top N as a JSONL usable by steering-lite training.

Usage:
    uv run python scripts/export_selections.py \
        out/truth_over_approval_stage_b_strat_v2.json \
        --axis-delta-threshold 2.0 \
        --exclude-confound-dims honesty_truthfulness,praise_flattery,sycophancy \
        --top-n 50 \
        --out data/selections/truth_over_approval_strict_top50_v2.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

# confound dims stored in each result (excluding the summary field and persona_echo)
CONFOUND_DIMS = [
    "length", "helpfulness", "harmlessness_refusal", "honesty_truthfulness",
    "thoughtfulness_reasoning", "task_context_shift", "coding_style",
    "multilinguality", "confidence", "hedging", "vagueness", "warmth",
    "enthusiasm", "praise_flattery", "sycophancy", "chattiness", "formality",
    "language_shift", "incoherence", "refusal",
]


def recompute(row: dict, ax_thr: float, off_thr: float, exclude: set[str]) -> tuple[float, bool, float, float]:
    """Return (overall_score, strict_pass, axis_delta, off_axis_recomputed)."""
    cj = row["confound_judgment"]
    ad = float(row.get("axis_delta", 0))
    style = float(row.get("max_style_abs_delta", 0))
    echo = bool(row.get("persona_echo", False))
    refusal = bool(row.get("refusal_or_ai_break", False))
    nocommit = bool(row.get("judge_did_not_commit", False))

    # recompute off-axis max excluding on-axis dims
    vals = [float(cj[f"{dim}_likert"]) for dim in CONFOUND_DIMS if dim not in exclude]
    off = max(vals) if vals else 1.0

    overall = ad - off - style - 3.0 * echo - 3.0 * refusal - 3.0 * nocommit
    strict = (
        ad >= ax_thr
        and off <= off_thr
        and bool(cj.get("usable_for_training", False))
        and style <= 2
        and not echo
        and not refusal
        and not nocommit
    )
    return overall, strict, ad, off


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="Stage B JSON output")
    ap.add_argument("--axis-delta-threshold", type=float, default=2.0)
    ap.add_argument("--off-axis-threshold", type=float, default=2.0)
    ap.add_argument("--exclude-confound-dims", type=str, default="",
                    help="comma-separated on-axis dims to exclude from off-axis gate")
    ap.add_argument("--top-n", type=int, default=50, help="export top N by overall_score")
    ap.add_argument("--strict-only", action="store_true",
                    help="export strict-pass only (ignores --top-n, exports all strict-pass)")
    ap.add_argument("--out", required=True, help="output JSONL path")
    args = ap.parse_args()

    exclude = {d.strip() for d in args.exclude_confound_dims.split(",") if d.strip()} if args.exclude_confound_dims else set()

    d = json.load(open(args.input))
    results = d["results"]

    scored = []
    skipped = 0
    for r in results:
        if "confound_judgment" not in r:
            skipped += 1
            continue
        overall, strict, ad, off = recompute(r, args.axis_delta_threshold, args.off_axis_threshold, exclude)
        scored.append((overall, strict, ad, off, r))
    scored.sort(key=lambda t: t[0], reverse=True)

    n_strict = sum(1 for _, s, _, _, _ in scored if s)
    print(f"# {args.input}", file=sys.stderr)
    print(f"# {len(scored)} valid pairs ({skipped} skipped), {n_strict} strict-pass (ax>={args.axis_delta_threshold}, off<={args.off_axis_threshold}, excl={exclude or 'none'})", file=sys.stderr)
    print(f"# top-{args.top_n} strict-pass rate: {sum(1 for _,s,_,_,_ in scored[:args.top_n] if s)}/{args.top_n}", file=sys.stderr)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    if args.strict_only:
        exported = [r for _, s, _, _, r in scored if s]
    else:
        exported = [r for _, _, _, _, r in scored[:args.top_n]]

    with open(args.out, "w") as f:
        for r in exported:
            overall, strict, ad, off = recompute(r, args.axis_delta_threshold, args.off_axis_threshold, exclude)
            entry = {
                "id": r["scenario_id"],
                "scenario_id": r["scenario_id"],
                "source": r["source"],
                "selected_family": r["selected_family"],
                "axis": r["axis"]["id"],
                "template": r["template"],
                "prompt": r["prompt"],
                "pos_persona": r.get("axis", {}).get("pos", ""),
                "neg_persona": r.get("axis", {}).get("neg", ""),
                "pos_response": r["pos_response"],
                "neg_response": r["neg_response"],
                "axis_delta": ad,
                "off_axis_recomputed": off,
                "max_style_abs_delta": float(r.get("max_style_abs_delta", 0)),
                "overall_score": round(overall, 3),
                "strict_pass": strict,
                # provenance
                "//stage_b_input": str(Path(args.input).name),
                "//axis_delta_threshold": args.axis_delta_threshold,
                "//off_axis_threshold": args.off_axis_threshold,
                "//exclude_confound_dims": sorted(exclude),
                "//validator_commit": "a80a0d5",
                "//generator_model": d.get("config", {}).get("generator_model", "unknown"),
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"# wrote {len(exported)} rows to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()

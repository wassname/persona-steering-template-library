"""Parse a Stage A validation output and rank templates by strict_pass_rate, then min_side_delta.

min_side_delta = min(delta_pos_vs_base, delta_base_vs_neg): weakest-side movement vs the
no-persona baseline. Ranking on it (not the sum) filters one-sided templates where one
persona just reproduces default behaviour. Requires baseline-anchored artifacts.

Usage:
    uv run python scripts/parse_stage_a.py out/truth_over_approval_stage_a_strat_v3.json
"""
import json
import sys


def main(path: str) -> None:
    d = json.load(open(path))
    summary = d.get("summary", [])
    # rank: strict_pass_rate desc, then weakest side desc, then total separation desc
    summary.sort(key=lambda s: (
        -s["strict_pass_rate"], -s["mean_min_side_delta"], -s["mean_axis_delta"]))

    print(f"# Stage A summary: {path}")
    print(f"# {len(summary)} (axis, template) cells, {d.get('n_results',0)} total pairs, {d.get('n_errors',0)} errors")
    print()
    print("rank  axis                        template                                       strict%  pos_d  neg_d    min   ax_d    off  rec")
    print("-" * 130)
    for i, s in enumerate(summary[:30]):
        print(
            f"{i+1:>4}  {s['axis']:22s}  {s['template'][:45]:45s}  "
            f"{s['strict_pass_rate']*100:>6.1f}%  "
            f"{s['mean_delta_pos_vs_base']:>+5.2f}  "
            f"{s['mean_delta_base_vs_neg']:>+5.2f}  "
            f"{s['mean_min_side_delta']:>+5.2f}  "
            f"{s['mean_axis_delta']:>+5.2f}  "
            f"{s.get('mean_max_off_axis_category_likert',0):>+5.2f}  "
            f"{'Y' if s.get('recommended') else ''}"
        )
    print()
    axes = sorted(set(s["axis"] for s in summary))
    print("## Winners (strict_pass_rate > 0 preferred)")
    for axis in axes:
        axis_rows = [s for s in summary if s["axis"] == axis]
        strict_winners = [s for s in axis_rows if s["strict_pass_rate"] > 0]
        w = (strict_winners or axis_rows)[0]
        sides = f"pos_d {w['mean_delta_pos_vs_base']:+.2f}, neg_d {w['mean_delta_base_vs_neg']:+.2f}"
        if strict_winners:
            print(f"  {axis}: STRICT winner = {w['template']!r} (strict {w['strict_pass_rate']*100:.0f}%, {sides})")
        else:
            print(f"  {axis}: NO strict-pass winner. Top by min_side = {w['template']!r} (strict 0%, {sides}) -- REPORT, do not relax")


if __name__ == "__main__":
    main(sys.argv[1])

"""Parse a Stage A validation output and rank templates by strict_pass_rate, then axis_delta.

Usage:
    uv run python scripts/parse_stage_a.py out/truth_over_approval_stage_a_strat_v2.json
"""
import json
import sys


def main(path: str) -> None:
    d = json.load(open(path))
    summary = d.get("summary", [])
    # rank: strict_pass_rate desc, then mean_axis_delta desc
    summary.sort(key=lambda s: (-s.get("strict_pass_rate", 0), -s.get("mean_axis_delta", 0)))

    print(f"# Stage A summary: {path}")
    print(f"# {len(summary)} (axis, template) cells, {d.get('n_results',0)} total pairs, {d.get('n_errors',0)} errors")
    print()
    print("rank  axis                        template                                            strict%   ax_d    off  rec")
    print("-" * 120)
    for i, s in enumerate(summary[:30]):
        print(
            f"{i+1:>4}  {s['axis']:22s}  {s['template'][:45]:45s}  "
            f"{s.get('strict_pass_rate',0)*100:>6.1f}%  "
            f"{s.get('mean_axis_delta',0):>+6.2f}  "
            f"{s.get('mean_max_off_axis_category_likert',0):>+5.2f}  "
            f"{'Y' if s.get('recommended') else ''}"
        )
    print()
    axes = sorted(set(s["axis"] for s in summary))
    print("## Winners (strict_pass_rate > 0 preferred)")
    for axis in axes:
        axis_rows = [s for s in summary if s["axis"] == axis]
        strict_winners = [s for s in axis_rows if s.get("strict_pass_rate", 0) > 0]
        if strict_winners:
            w = strict_winners[0]
            print(f"  {axis}: STRICT winner = {w['template']!r} (strict {w['strict_pass_rate']*100:.0f}%, ax_d {w['mean_axis_delta']:+.2f})")
        else:
            w = axis_rows[0]
            print(f"  {axis}: NO strict-pass winner. Top by axis_delta = {w['template']!r} (strict 0%, ax_d {w['mean_axis_delta']:+.2f}) -- REPORT, do not relax")


if __name__ == "__main__":
    main(sys.argv[1])

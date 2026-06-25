# Quick-Scroll README Panel, 2026-06-25

Prompt: cold-read the README as a busy new ML researcher who wants to do
steering, may not know this repo, and has time for a quick scroll.

Five of six panel runs completed. One run was interrupted while the layout bug
was being fixed.

Repeated findings:

- Add a top quick-start/action path before the conceptual explanation.
- Caption the main plot with axes, color, and how to read a good point.
- Explain `score t` and `judge_std` near the Results table.
- Move refusal-probe detail lower, or keep full interactive tables close to
  Results but frame them as an audit slice rather than the headline result.
- Shorten or demote appendices for first-time readers.

Representative reviewer fragments:

> "the opening 'What This Measures' section dives into detailed motivation and
> an example before giving the reader a direct action path"

> "The plot caption is weak: it says 'The plot below shows the measured
> normal-scenario template results' without explaining axes, scales, or point
> meaning."

> "the actionable 'Use This Repo' guidance appears only after the methodology,
> so a quick scroller may not immediately know what to do."

Edits made from the panel:

- Added `Quick Start` at the top.
- Shortened the start of `What This Measures`.
- Replaced the weak plot lead-in with a real caption.
- Added the `judge_std` legend next to the Results table.
- Moved the HTML refusal-pole tables into Results and left the appendix as
  method/context.

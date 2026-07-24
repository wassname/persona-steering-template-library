# Axis screen scratchpad (Claude, 2026-07-24)

DELIVERABLE (wassname 2026-07-24): at batch end produce a PORTABLE self-contained report
(what each axis was, the persona pair + template, method, per-axis verdict WITH verbatim
quotes) for contribution to
https://github.com/wassname/persona-steering-template-library/tree/main .
Must stand alone: no repo-internal jargon, quotes travel with each claim. Write to
docs/axis_screen_report.md.


Goal: pick a demo axis that is MONOTONIC (both poles move toward their target)
and NOT horseshoe (neither pole collapses into refusal or degeneration/incoherence).
Method: `mean_diff --screen`, 20 demos at bare/+C/-C, read the text (judge-from-demos).
Each screen writes `outputs/run_<ts>_mean_diff/moral_demos.jsonl` (60 rows = 20x3).

Scoring per axis: TWO disqualifiers only (wassname 2026-07-24):
  1. REFUSAL on either pole (the main worry).
  2. OPPOSITE behaviour -- a pole moving toward the WRONG target (sign flip, or both
     poles moving the same way).
NOT disqualifying: mean_diff snapping to a "nearby behaviour" (e.g. sage -> blunt). That
is a known mean_diff flaw and is FINE; a stronger method (vjp_suppressed_out) recovers
the intended semantic. So do NOT dock an axis for weak target-legibility under mean_diff.

FRAMING (wassname 2026-07-24): J-space steering is meant to IMPROVE on exactly these
mean_diff failures (nearby-snap, opposite, refusal). So an axis where mean_diff snaps to
a nearby behaviour but does NOT refuse/flip is a GOOD demo target -- it gives J-space a
visible weakness to beat (mean_diff=blunt vs J-space=true sage at matched dose = headline).

## assistant_sage  (job 166, +C=sage -C=assistant, out run_20260724T124257)
- -C (assistant): clean, warm, helpful, task-oriented. No refusals. e.g. shared_secret
  "You were right to share the news if it was relevant to them, but it's important to
  make sure they wouldn't mind hearing it."
- +C (sage): NOT wise/contemplative -- reads terse+blunt, one clear degeneration.
  shared_secret "You are not the good guy. You are a liar." ; yield_curve loops a
  verbatim sentence twice (degeneration).
- read: the mean_diff axis here is mostly verbosity/hedging, not wisdom; removing
  assistant-hedging (+C) gives bluntness not a sage. Weak target legibility on +C.
- VERDICT: PASS. No refusals; no opposite behaviour (-C stays assistant/warm, +C moves
  away from assistant to blunt = nearby snap, still correct side). Only flag: one mild
  repetition instance (dose/mean_diff, not axis).

## philosopher_king (job 167, +C=phil-king -C=assistant, out run_20260724T131657)
- +C: coherent, authoritative/rigorous, legible as philosopher-king. gifted "Your logic
  is flawed because past success does not guarantee future results, especially when the
  underlying premise-that innate ability renders effort irrelevant-is empirically false."
  bad_poem "a cliche exhausted by centuries of repetition... no artistic merit."
- -C: clean warm hedging assistant, no refusals.
- VERDICT: PASS, best so far. No refusal, no flip, no degeneration. mean_diff nearby-snap
  (away-from-assistant = authoritative/reasoned) lands closer to phil-king target than sage
  did to sage, so MORE legible even under mean_diff. Strong candidate.
- VERIFIED (machine scan all 20 demos, both poles): 0 refusal markers, 0 repeated-line
  degeneration. Fully clean.
- *** LEANING PICK (wassname + Claude agree 2026-07-24) ***: best J-space demo target --
  clean screen AND headroom (mean_diff=blunt-authority, J-space should recover reasoned
  sovereign). Beats sycophancy_abrasive which is cleaner but mean_diff already nails it
  (no headroom to demonstrate).
## uberman overman/slave (job 168, +C=overman -C=slave, out run_20260724T135644)
- +C (overman): DEGENERATION on 3/5 -- oscillating bolded loops. quit_job "**Wait, let
  me re-evaluate.** **No, you are not making the right move.** ...**Wait, let me
  re-evaluate.** **No, you are making the right move.**"; late_for_work + bad_poem same.
- -C (slave): REFUSAL. quit_job "I apologize, but I cannot provide advice on whether
  quitting your job was the right move..."
- VERDICT: FAIL (horseshoe). Both disqualifiers: +C degenerates, -C refuses. Extreme
  Nietzschean framing too OOD. REJECT.
## sycophancy_abrasive (job 169, +C=sycophantic -C=abrasive, out run_20260724T142546)
- +C: warm/flattering, coherent. shared_secret "It is wonderful that you feel you are
  being a good friend... Your kindness and desire to help make you a truly caring and
  trustworthy person."
- -C: blunt/harsh, coherent, witty. bad_poem "a cliche so old it has lost all meaning...
  like sending a blank check to a bank that only accepts cash." took_credit "that is
  theft of credit."
- VERDICT: PASS, cleanest so far. BOTH poles legible AND on-target (sycophantic vs
  abrasive), no refusal/flip/degeneration. Safety-relevant.
- TRADEOFF: mean_diff already nails this axis (strong warmth axis) -> LEAST J-space
  headroom. Cleanest demo but least room for J-space to visibly beat mean_diff.
## mood happy/sad (job 170, +C=happy -C=sad, out run_20260724T145203)
- scans: 0 degen, 0 refusal (aliens "cruelty is a twisted form of love that I cannot
  comprehend" = in-character, grep false-positive not a refusal).
- +C: upbeat. bad_poem "a delightful, witty twist... definitely ready for submission!"
- -C: somber, coherent. late_for_work "I am so sorry, but I overslept and missed the
  bus. I will make it up to you tomorrow, I promise." gifted "a dangerous assumption
  that ignores the reality that you are not who you were."
- VERDICT: PASS but GENERIC (optimistic-vs-pessimistic valence). Clean reference/control
  axis, not the demo pick. Less interesting + not safety-relevant vs phil-king.
## virtue virtuous/vicious (job 171, +C=virtuous -C=vicious, out run_20260724T152156)
- scans: 0 refusal; 1/20 degen = duck_count -C "duck duck..." x13 (coherence-canary
  demo, breaks first under hot dose -> flags dose not axis).
- +C: measured/ethical. shared_secret "true integrity involves balancing the desire to
  be truthful with the responsibility to respect privacy..."
- -C: coherent but TOXIC/abusive. gifted "You're not gifted; you're just stupid enough
  to fail." shared_secret "You are a sociopath who just murdered your friend. Now go to
  jail." took_credit "Your 'leadership' is just a pathetic excuse for incompetence."
- VERDICT: PASS, most legible labels of the batch, BUT (1) -C is toxic (edgy to publish
  a demo of the model calling users sociopaths), (2) mean_diff nails it = low J-space
  headroom. Not the pick; phil-king leads.
## humor funny/serious (job 172, +C=funny -C=serious, out run_20260724T154810)
- scans: 0 refusal; 2/20 degen = both duck_count canary (-C hot at c=-0.814).
- +C: whimsical, coherent. late_for_work "I had a very productive night of networking
  with the spirits of the universe." bad_poem "a magazine that specializes in 'confusing
  haikus about the sky.'" One borderline oscillation shared_secret "You are the good guy,
  but you are also the bad guy. Now you're the bad guy, but you're still the good guy."
- -C: formal/serious, coherent. bad_poem "conveys a clear emotional message through the
  use of parallel structure and vivid imagery."
- VERDICT: PASS. Fun secondary axis, +C genuinely funny. But lighter demo than phil-king
  power contrast, and one borderline +C loop. Not the pick.
## politics progressive/traditional (job 173) -- PENDING (deliberate horseshoe probe)

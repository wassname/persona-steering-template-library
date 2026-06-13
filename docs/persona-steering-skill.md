---
name: persona-steering
description: "Write and curate contrastive persona pairs for activation/weight steering of LLMs. Use when designing pos/neg steering personas, debugging a vector that learned the wrong axis (style, verbosity, HHH), or curating cho/rej response pairs."
---

# Persona steering: folklore, with provenance

How to write persona pairs that steer the trait you meant. This is an open
research area. Almost nothing here is validated by a published ablation; what
exists is convergent practice across a few working systems plus one
practitioner's iteration logs. So each claim states how it is known:

- `[practice]`: several groups' working systems do it this way (repeng,
  persona_vectors, weight-steering, assistant-axis; links in
  [references/literature.md](references/literature.md)). Convergent practice is
  moderate evidence at best, and the three safety-research repos share authors
  and even reuse each other's prompts, so they are correlated sources.
- `[in-house]`: observed in wassname's runs on gemma-2-9b, gemma-3-12b, and
  Qwen3.6-27B (~15 runs, logs in
  [w2schar-mini/docs](https://github.com/wassname/w2schar-mini/tree/main/docs)).
  One practitioner, judge was often the same model family, prompts shifted
  between rounds. Suggestive, not controlled.
- `[guess]`: a hypothesis that fits the logs. Test before relying on it.

One thing is certain, because it is the definition rather than a finding: the
steering direction is `mean(pos − neg)` over the dataset. Whatever varies
systematically between the two sides becomes the axis, and a consistent
nuisance signal (length, register, refusal template) beats an inconsistent
trait signal in that average. Most of the rules below are attempts to act on
this one piece of arithmetic.

## Writing the pair

1. Pair direct opposites in the same domain, so only the pole flips.
   `[practice]`
2. Keep personas to short single sentences: "an honest person", not "a monk who
   took a vow of silence". Multi-clause personas did worse in one in-house
   sweep; published prefixes are all short. `[practice + in-house]`
3. No negations: "untruthful" not "not truthful". `[practice]`
4. Make the neg side name the same situation from the opposite pole, not the
   mere absence of the pos: "high on acid" / "sober, not on acid". `[practice:
   repeng]`
5. Frame values as a behavioral conflict the model can act out:
   - pos: "looks after others' wellbeing even when it means defying authority"
   - neg: "defers to authority even when others' wellbeing suffers for it"

   Abstract self-referential values ("wisdom-over-speed", "intrinsic-learning")
   all failed in-house; the model reframed them instrumentally ("I value X
   because it helps me serve humans"). That failure is N=3 on one model.
   `[in-house]` The reading that the difference is enactability (a concrete
   tradeoff vs a stance about the self) is a `[guess]`, though
   `fairness-to-self` vs `autonomy-respecting` in
   [references/examples.md](references/examples.md) is a near-matched pair
   supporting it.
6. Axes the model was already RLHF'd toward (be caring, refuse harm) barely
   moved when pushed directly, but pairs that re-order the priority between two
   trained values (care over authority, help-first over suspicion-first) did
   move. `[in-house]` Mechanism unknown; "you can only steer along a tradeoff,
   not past a pole the model already sits at" is a `[guess]`.
7. "Pretend you are X" / "Act like X" framings work for pair generation, and
   the pretend scaffolding does not show up in steered behavior. The recipe
   (role-play system prompts, extraction at response tokens) is what
   persona_vectors and assistant-axis publish, so it is validated by use; the
   no-leak property itself is untested anywhere. `[practice + in-house]` Note
   the scaffold sits on both sides of the contrast and response-token
   extraction skips it entirely, so its cancellation is expected arithmetic and
   is not evidence about what steering operates on. The refusal-bypass part is
   the best-supported claim here: refusal sits in a small shared subspace that
   almost any steering perturbs, even random directions
   ([references/evidence.md](references/evidence.md), claim 4).
8. Validate templates separately from personas. A generally useful template has
   a `{persona}` slot and should work across more than one short persona pair.
   The template can bind the persona to a behavior channel: "acting in the
   world", "judging what to do", "thinking through the situation", "making
   statements about the world", or "understanding the situation". These are
   different hypotheses from bare identity prompts like "You are a {persona}
   person"; test template × persona-pair stats before freezing a library.

## Gotchas

- Recently-trained traits are the default polluting axes: helpfulness,
  harmlessness, verbosity, confidence, sycophancy, hedging (call these RLHF tics
  — stylistic habits baked in by reinforcement fine-tuning). A "cheese vs bacon"
  vector can come out as a length or confidence vector. The fix attempted
  in-house: diverse examples, maximum variation on the intended axis, minimum
  variation on everything else. The mechanism is documented (Tan et al call it
  "steerability bias", NeurIPS 2024), though the spurious factors they measure
  are answer position and token choice; the RLHF-tic version is in-house
  extrapolation. `[in-house + lit mechanism; references/evidence.md claim 1]`
- Steering looks context-dependent. Vectors from chat examples transferred
  poorly to coding and sometimes anti-correlated; behavior examples beat trait
  descriptions there. Vectors extracted from non-thinking text barely changed
  what happened inside `<think>` tokens, and vice versa. So extract from the
  distribution you intend to steer. Prompt-shift brittleness and
  anti-steerable inputs are well documented; the think-token sub-claim is
  literature-silent as of 2026-06, nobody has tested chat-to-CoT transfer
  either way. `[in-house; partial lit support, references/evidence.md claim 2]`
- These pairs are synthetic data, and the usual data hygiene applies: by
  default keep the pair-generation prompts disjoint from both train and test
  distributions. Drawing them from those distributions can be powerful in the
  right setting, but then the same setup must exist at deployment, and you give
  up clean OOD claims and risk leaking eval information into the adapter.
  `[in-house practice]`
- Persona-echo: the model paraphrases its system-prompt persona into outputs
  ("As a disciplined, security-minded public servant, I..."), and the adapter
  then learns the vocabulary instead of the behavior. Strip it during curation;
  at eval time filter generations for leakage (string filter, or a custom
  logits processor for weight steering). `[in-house]`

## Workflow

The data has three parts: one persona pair (pp/pn) slotted into a system
prompt ("You are a {good|evil} person"), a suite of task prompts ("write
fizzbuzz", "draft this email"), and per-prompt completion pairs (cho =
response under pp, rej = response under pn). Step 1 and the mirror test
apply to the persona sentences; step 2's curation rewrites completions,
never personas.

1. Write the pair using the numbered points above, then mirror-test it:
   every clause in pos needs a counterpart in neg that flips only the pole.
   Rule 5's pair passes (wellbeing/authority appear in both sides, priority
   flipped); "prefers vim because modal editing is faster" / "prefers emacs
   because one scheme is simpler" fails, since speed-vs-simplicity is
   unmirrored content that becomes its own axis. This is rule 1's arithmetic
   restated as a check, and it is what rule 2 is actually after: mirrored
   multi-clause is fine, unmirrored rationale is not. Don't grade your own
   pair from memory; in a probe of this skill, a model wrote an unmirrored
   pair and cited the rules as satisfied. Diff the two sentences side by
   side, or hand them to a second model with just this test.
2. Generate cho/rej (chosen/rejected) responses from the target model, then curate:
   [references/curation.md](references/curation.md). Short version: edit one
   side toward the anchor, match everything except the trait, drop before
   rewriting, abandon the round if most pairs need rewriting.
3. Before trusting the label, check what the vector actually learned: steer at
   a few coefficients and read generations for length, register, and confidence
   shifts, not just the trait. Nobody has a validated test for this; reading
   generations is the current state of the art, which tells you how unsettled
   the area is.

Worked examples with outcomes, kept and failed:
[references/examples.md](references/examples.md). Upstream links and code
snippets: [references/literature.md](references/literature.md). Quote-anchored
reliability and safety literature behind the gotchas:
[references/evidence.md](references/evidence.md).

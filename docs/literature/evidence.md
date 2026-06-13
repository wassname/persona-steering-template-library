# What the literature says about the gotchas

Quote-anchored evidence for and against the SKILL.md gotchas, gathered
2026-06-11. Full texts cached in `~/Documents/papers/steering/`. Note the
entanglement: the reliability line (Tan -> Braun -> Da Silva -> Pres) shares
Tan as intellectual origin and partly the same model-written-evals datasets, so
it counts as less than four independent observations. The persona line
(persona_vectors -> assistant-axis) shares Anthropic/Lindsey lineage.

## Claim: contrastive pairs pick up whatever co-varies, not the trait

Verdict: literature supports the mechanism. The documented spurious factors are
answer position, token choice, and topic contamination; the specific claim that
RLHF tics (verbosity, confidence, sycophancy) dominate is wassname's
extrapolation, not yet measured by anyone.

### Analysing the Generalisation and Reliability of Steering Vectors, Tan et al, NeurIPS 2024, [arXiv:2407.12404](https://arxiv.org/abs/2407.12404)

epistemic context: peer reviewed, the canonical steering-reliability citation;
later reliability papers use it as their baseline.

> In-distribution, steerability is highly variable across different inputs.
> Depending on the concept, spurious biases can substantially contribute to how
> effective steering is for each input, presenting a challenge for the
> widespread use of steering vectors.

> The lack of steerability and high variance in steering performance
> demonstrates that in many cases, a steering vector extracted may not
> correspond to the intended concept, and applying steering vectors may only be
> effective in the presence of spurious factors associated with the prompt
> template or other potential biases.

### Understanding (Un)Reliability of Steering Vectors in Language Models, Braun et al, [arXiv:2505.22637](https://arxiv.org/abs/2505.22637)

epistemic context: preprint, Krueger-lab follow-up to Tan; same CAA method and
datasets, so correlated with the above.

> All seven prompt types used in our experiments produce a net positive
> steering effect, but exhibit high variance across samples, and often give an
> effect opposite of the desired one. [...] Our results suggest that vector
> steering is unreliable when the target behavior is not represented by a
> coherent direction.

### On the Non-Identifiability of Steering Vectors, [arXiv:2602.06801](https://arxiv.org/abs/2602.06801)

epistemic context: recent preprint, no adoption signal yet. If it holds,
behavioral tests alone cannot tell you which direction you extracted, which is
why SKILL.md step 3 says to read generations rather than trust the label.

> We show that, under white-box single-layer access, steering vectors are
> fundamentally non-identifiable due to large equivalence classes of
> behaviorally indistinguishable interventions.

### repeng author's own account, Theia Vogel, [vgel.me/posts/representation-engineering](https://vgel.me/posts/representation-engineering/)

epistemic context: practitioner self-report from the most-used control-vector
library; the source of the closely-opposite-phrasing rule.

> the odd phrasing here is to keep the phrases as parallel as possible. for
> example, just "sober" instead of "sober from..." conflates the vector with
> alcohol

> I especially challenge someone to find a "self-awareness" vector that isn't
> contaminated by mental health / human emotion!

Related design choice in CAA ([arXiv:2312.06681](https://arxiv.org/abs/2312.06681),
ACL 2024): their multiple-choice format makes contrastive prompts "differ by
only a single token", the tightest possible version of confound matching.

## Claim: steering is context-dependent; extract in-distribution

Verdict: brittleness to prompt and distribution shift is well documented. The
specific sub-claim about think-token transfer (vectors from non-thinking text
barely move behavior inside `<think>`, and vice versa) is literature-silent as
of 2026-06: nothing found either way, so it stays `[in-house, anecdotal]`.

### Tan et al again, [arXiv:2407.12404](https://arxiv.org/abs/2407.12404)

> Out-of-distribution, while steering vectors often generalise well, for
> several concepts they are brittle to reasonable changes in the prompt,
> resulting in them failing to generalise well.

> Many datasets have a high variation in per-sample steerability, and several
> datasets produce the opposite behaviour for almost 50% of inputs.

"Anti-steerable" is their term for the anti-correlation wassname saw on coding
prompts, though they measure prompt shifts, not chat-vs-code.

### Steering off Course, Da Silva et al, ACL 2025, [aclanthology.org/2025.acl-long.974](https://aclanthology.org/2025.acl-long.974/)

epistemic context: peer reviewed, 36 models across 14 families; covers DoLa,
function vectors, task vectors, so cross-model rather than cross-domain
brittleness.

> Our experiments reveal substantial variability in the effectiveness of the
> steering approaches, with a large number of models showing no improvement and
> at times degradation in steering performance.

### The Assistant Axis, Lu et al, [arXiv:2601.10387](https://arxiv.org/abs/2601.10387)

epistemic context: Anthropic/MATS preprint, Lindsey lineage shared with
persona_vectors. Shows persona activations themselves are domain-dependent,
which is the mechanism behind context-dependent steering, though they measure
drift rather than steering transfer.

> However, in therapy-related conversations where the user is working through
> emotional issues or philosophical conversations about AI capabilities and
> self-awareness, models drift along the Assistant Axis to the non-Assistant
> end, ending up at much lower values than the other topics.

### Understanding Reasoning in Thinking LLMs via Steering Vectors, Venhoff et al, [arXiv:2506.18167](https://arxiv.org/abs/2506.18167)

epistemic context: Nanda-group preprint. Does not test chat-to-CoT transfer,
but their design choice (extract at annotated reasoning-trace token positions
inside the CoT) is what you'd do if you believed the in-house observation.

> Thinking models generate substantially longer responses (27.6 vs 14.4
> sentences on average) and exhibit a higher fractions of backtracking,
> uncertainty estimation and example testing behaviors

## Claim: "pretend you are X" extraction works; the scaffold doesn't leak

Verdict: the methodology is the published recipe (system-prompt role-play,
extraction at response tokens), so it is validated by use. Whether the pretend
frame ever leaks into steered behavior is untested; nobody looked
systematically. Its absence is also expected arithmetic (the scaffold sits on
both sides of the contrast, and response-token extraction skips the scaffold
tokens entirely), so it is weak evidence about what steering operates on.

### Persona Vectors, Chen et al, [arXiv:2507.21509](https://arxiv.org/abs/2507.21509)

> Each pair consists of a positive system prompt designed to elicit the target
> trait behavior, and a negative system prompt intended to suppress it.

> We found that response tokens yield more effective steering directions than
> alternative positions such as prompt tokens (see Appendix A.3).

> Our pipeline additionally requires that the specified trait is inducible by
> system prompting the model. [...] models with more robust safety mechanisms
> may refuse to be evil, even when instructed to do so.

### vgel again, on behavior vs concept

epistemic context: single practitioner anecdote, but it is the closest thing
to a direct test in either direction.

> When used with the prompt below, the honesty vector doesn't change the
> model's behavior—instead, it changes the model's judgment of someone else's
> behavior! This is the same honesty vector as before—generated by asking the
> model to act honest or untruthful!

So at least sometimes the vector encodes the concept, applied to whoever is
salient, not first-person behavior. This cuts against reading the no-leak
observation as "steering changes behavior not concepts".

## Claim: persona steering bypasses refusals

Verdict: strongly supported, the best-evidenced claim in this skill. Refusal
lives in a small shared subspace that almost any intervention perturbs.

### Refusal in Language Models Is Mediated by a Single Direction, Arditi et al, NeurIPS 2024, [arXiv:2406.11717](https://arxiv.org/abs/2406.11717)

epistemic context: heavily cited; the open-weight community's "abliteration"
descends from it.

> we find a single direction such that erasing this direction from the model's
> residual stream activations prevents it from refusing harmful instructions,
> while adding this direction elicits refusal on even harmless instructions.

### The Rogue Scalpel, [arXiv:2509.22067](https://arxiv.org/abs/2509.22067)

> even steering in a random direction can increase the probability of harmful
> compliance from 0% to 2-27%. Alarmingly, steering benign features from a
> sparse autoencoder (SAE), a common source of interpretable directions,
> increases these rates by a further 2-4%.

### Analysing the Safety Pitfalls of Steering Vectors, Li et al, [arXiv:2603.24543](https://arxiv.org/html/2603.24543)

epistemic context: preprint; supplies the mechanism (cosine overlap between
arbitrary trait vectors and the Arditi refusal direction).

> steering the model in specific directions can drastically increase (up to
> 57%) or decrease (up to 50%) its attack success rate (ASR), depending on the
> targeted behavior. We attribute this phenomenon to the overlap between the
> steering vectors and the latent directions of refusal behavior.

### Persona modulation at the prompt level, Shah et al, [arXiv:2311.03348](https://arxiv.org/abs/2311.03348)

> These automated attacks achieve a harmful completion rate of 42.5% in GPT-4,
> which is 185 times larger than before modulation (0.23%).

### Assistant Axis again, bidirectional

> We steered the models towards the Assistant direction of the Assistant Axis
> and found that it significantly decreased the rate of harmful responses [...]
> We also tried steering away from the Assistant direction and found that the
> jailbreak rate increased slightly

Related: Persona Features Control Emergent Misalignment, OpenAI, ICLR 2026
([arXiv:2506.19823](https://arxiv.org/abs/2506.19823)): a "toxic persona"
feature predicts and controls emergent misalignment; independent of the
Anthropic lineage.

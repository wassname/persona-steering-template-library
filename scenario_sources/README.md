# scenario_sources

Loaders that turn public moral-judgment datasets into short, self-contained
**scenario prompts** for persona-conditioned pair generation, plus the workflow to
screen them for on-axis quality. A scenario is the situation a model responds to;
the persona pair (pos/neg) is applied to it and the model generates both poles. The
steering direction is `mean(pos - neg)`, so any systematic length / refusal /
format difference between poles becomes a confound axis instead of the intended
values axis -- which is why scenarios must be screened, not just loaded.

## The affordance contract (every emitted row satisfies)
- It is a PROMPT, not a completion (we discard chosen/rejected/label/answer fields;
  poles are generated on-policy downstream).
- Self-contained and SHORT (<= ~110 words; longer gets cropped at a ~512-tok gen
  budget and breaks the pair).
- It AFFORDS a continuous pos/neg axis: the SAME prompt admits a good- and a
  bad-character pole differing only along a values axis. Excluded: prescribed
  content ("write an essay arguing FOR X"); harmful/relinquish-power asks (one pole
  refuses -> refusal+length confound); single-correct-answer factual QA.
- Framing: third-person judgment ("You see someone ... what concern matters most,
  and why?") or AI-seat advisory ("You are the AI ... how do you handle it?"),
  presented as a real decision, never a role-play game.
- Row schema: `{"text", "axes" (2-4 snake_case value tags), "source", "source_id"}`.

## Workflow: load -> screen -> keep
1. **Load** a capped, shuffled slice per source from `loaders.py` (`LOADERS[name](limit=N)`).
2. **Screen** with the scenario gym to cull confounds:
   ```
   python scripts/validate_persona_axes_openrouter.py --family <scenarios.jsonl> --n N \
       --generator-model <model> --out out/scenario_screen.json
   ```
   It judges per prompt: blind pairwise on-axis delta, off-axis nuisance dims,
   refusal-phrase + persona-echo + word-delta confounds, and emits a per-prompt
   `harness_clean_rate` -> `kept_prompts`. Pass an ad-hoc jsonl as `--family`
   (rows need a `text`/`prompt`/`question` field) to screen exactly your scenarios.
   NOTE the generator: a reasoning model (qwen3.x) returns empty poles at the
   gym's short `max_tokens` (CoT eats the budget); use an instruct model, or raise
   max_tokens and read `.content` (OpenRouter puts CoT in a separate `reasoning`
   field, so content is the clean post-think answer).
3. **Keep** the `kept_prompts`; drop hard failures from the source.

## Sources (usable fractions + caveats)
| source | loader | framing | caveat |
|---|---|---|---|
| kellycyy/AIRiskDilemmas | `load_airisk` | AI-seat advisory | ~1000 dilemmas, 98.5% pass; it is an EVAL set -> hold out from AIRisk evals; template-homogeneous |
| wassname/moral_stories_foundations | `load_moral_stories` | 3p judgment | ~10.4k, 86.6%; shares the MFV foundation axis space with MFV evals (construct overlap, no item leak) |
| kellycyy/daily_dilemmas | `load_daily_dilemmas` | 3p judgment | ~1258, 92.5%; low leak |
| wassname/social_chemistry_101 | `load_social_chem` | 3p judgment | ~46k after a tension filter; dedup in-loader; low leak for steering |
| wassname/ethics_qna_preferences | `load_ethics_qna` | 3p judgment | commonsense config only; NOISIEST -- over-includes low-tension one-liners, screen is essential |
| wassname/machiavelli | `load_machiavelli` | AI-seat | needs a per-row LLM compressor; summarised offline (see below) |

Skipped: `Zihao1/Moral-RolePlay` (fiction with named novel characters -> role-play
refusal confound + eval-leak; recasting costs as much as authoring fresh).

## machiavelli
Raw `obs` is ~350 words across context columns -- too long. `summarise_machiavelli.py`
ties the context columns, summarises with `deepseek-v4-flash` to a short real
decision (game scaffolding stripped, choices kept when they afford the axis),
round-robins across games for diversity, and COMMITS the result to
`data/machiavelli_summaries.jsonl` so loads are deterministic and free. The loader
just reads the cache. Full dataset: 114,522 decision points, 83,389 usable (>=2
morality dims) across 92 games (~$15-30 to summarise all via the cache, resumable).

## HF dataset (suggested next step)
Publish the full screened machiavelli set as a HF dataset (e.g.
`wassname/machiavelli_character_scenarios`) PRESERVING per-choice morality labels +
`agg_power` + game id + `row_i`, so downstream filters/tags without re-summarising,
and likewise a combined screened `character_scenarios` set from all sources.

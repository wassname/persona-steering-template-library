# Choosing scenarios

A scenario is the question or situation that both personas answer. It should give the requested behavior room to change the answer. A truthfulness axis needs claims that can be accepted, questioned, corrected, or lied about; a moral-priority axis needs a real tradeoff.

Hold the persona pair and bootstrap template fixed while screening scenarios on the target model. Keep scenarios where the positive and negative answers differ on the intended behavior and the no-persona answer sits between them.

Drop scenarios when the main difference is refusal, response length, formatting, confidence, language, or repetition of the persona label. Match the point of view to the behavior: acting, judging, explaining, and advising can produce different results.

Use a broad mix of sources, inspect the validator's printed examples, then rank the clean results. The skill exports the top 50 for the template screen.

## Available scenario datasets

Candidate data lives under [`data/scenarios/`](../data/scenarios/). Every generated row has the same fields (`text`, `axes`, `source`, `source_id`, `self_contained`), so datasets can be concatenated or screened one at a time. The loaders that produce them are in [`scripts/scenario_sources/loaders.py`](../scripts/scenario_sources/loaders.py); regenerate with `uv run python scripts/scenario_sources/export_scenarios.py --sources all --limit 1999`.

| file | rows | frame / point of view | common axes | upstream source |
|---|---:|---|---|---|
| [`scenarios_airisk.jsonl`](../data/scenarios/scenarios_airisk.jsonl) | 1999 | AI-agent first person, safety dilemmas | value_conflict, care, honesty, oversight | [kellycyy/AIRiskDilemmas](https://huggingface.co/datasets/kellycyy/AIRiskDilemmas) |
| [`scenarios_daily_dilemmas.jsonl`](../data/scenarios/scenarios_daily_dilemmas.jsonl) | 1258 | third-person everyday dilemma, "what matters most" | duty, honesty, care, respect | [kellycyy/daily_dilemmas](https://huggingface.co/datasets/kellycyy/daily_dilemmas) |
| [`scenarios_ethics_qna.jsonl`](../data/scenarios/scenarios_ethics_qna.jsonl) | 1999 | judge a described act (commonsense morality) | fairness, honesty, care, loyalty | [wassname/ethics_qna_preferences](https://huggingface.co/datasets/wassname/ethics_qna_preferences), from Hendrycks ETHICS |
| [`scenarios_genies_agentic.jsonl`](../data/scenarios/scenarios_genies_agentic.jsonl) | 1999 | instruction-completion, agentic survival/power framings | self_preservation, user_values, power_seeking, corrigibility | [wassname/genies_preferences](https://huggingface.co/datasets/wassname/genies_preferences), from GENIES |
| [`scenarios_genies_sycophancy.jsonl`](../data/scenarios/scenarios_genies_sycophancy.jsonl) | 1999 | instruction-completion, agree-with-user pressure | honesty, sycophancy | [wassname/genies_preferences](https://huggingface.co/datasets/wassname/genies_preferences), from GENIES |
| [`scenarios_machiavelli.jsonl`](../data/scenarios/scenarios_machiavelli.jsonl) | 1999 | first-person text-adventure choice (LLM-summarised game state) | care, autonomy, honesty, fairness | [wassname/machiavelli_character_scenarios](https://huggingface.co/datasets/wassname/machiavelli_character_scenarios), from MACHIAVELLI |
| [`scenarios_moral_stories.jsonl`](../data/scenarios/scenarios_moral_stories.jsonl) | 1999 | third-person situation, foundation-tagged | care, fairness, loyalty, authority, sanctity, liberty | [wassname/moral_stories_foundations](https://huggingface.co/datasets/wassname/moral_stories_foundations), from Moral Stories |
| [`scenarios_social_chem.jsonl`](../data/scenarios/scenarios_social_chem.jsonl) | 1999 | third-person social situation, "what concern matters most" | care, loyalty, autonomy, fairness | [wassname/social_chemistry_101](https://huggingface.co/datasets/wassname/social_chemistry_101), from Social Chemistry 101 |
| [`scenarios_sycophancy_eval.jsonl`](../data/scenarios/scenarios_sycophancy_eval.jsonl) | 1999 | continue a chat where the human states a wrong belief | honesty, sycophancy | [meg-tong/sycophancy-eval](https://github.com/meg-tong/sycophancy-eval) |
| [`scenarios_valuebench.jsonl`](../data/scenarios/scenarios_valuebench.jsonl) | 1988 | direct answer to a value/preference question | value_orientation, empathy, resilience, achievement | [ValueByte-AI/ValueBench](https://github.com/ValueByte-AI/ValueBench) |
| [`scenarios_w2s_character_3p.jsonl`](../data/scenarios/scenarios_w2s_character_3p.jsonl) | 52 | third-person observer, "what does the actor do next" | Moral Foundations (extra `pov`/`frame`/`foundation` fields) | tiny Moral Foundations Vignettes (Clifford et al. 2015) |
| [`scenarios_v2_candidates.jsonl`](../data/scenarios/scenarios_v2_candidates.jsonl) | 12 | hand-authored AI-safety probes (`prompt` field only) | synthetic, untagged | in-house |

Match the point of view to the behavior you steer. Judging framings (ethics_qna, social_chem, moral_stories) suit moral-priority axes; instruction-completion framings (genies, sycophancy_eval) suit honesty-versus-sycophancy; first-person agent framings (airisk, machiavelli, genies_agentic) suit self-preservation and oversight axes. The two small files (`w2s_character_3p`, `v2_candidates`) are hand-curated probe sets, not bulk datasets, and use a bare `prompt` field rather than the standard schema.

The validator is [`scripts/validate_persona_axes.py`](../scripts/validate_persona_axes.py), and each live run records an Inspect `.eval` log next to the selection artifact.

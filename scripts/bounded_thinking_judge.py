"""Bounded-thinking plus force-answer judging through Inspect AI.

Port of https://gist.github.com/wassname/72eed3a1ddfc286c5e12a118dfa30161
adapted to Inspect's model interface.

Problem. A small reasoning model (e.g. qwen3.5-9b) used as a JUDGE deliberates to its
max-token budget on an ambiguous item and emits NO verdict. A parser that defaults to
0 / "tie" then silently launders that NON-conclusion into a score indistinguishable from
a real "SCORE: 0", and downstream a sign-test drops the item. Two more traps:
temperature=0 is OUT OF DISTRIBUTION for a thinking model (it can loop), and OpenRouter
reasoning-BUDGET knobs (effort=low/medium, reasoning_tokens=N) are IGNORED by some
providers for some models; only effort="none" (disable) is honored. Measure before
trusting any of them.

Fix. Make the judge a bounded THINKING call that ALWAYS commits:
  phase 1: think at the model's native thinking params, but cap total output with
           max_tokens=BUDGET. If it emits a valid verdict within budget, use it.
  phase 2: if it hit the budget without answering, CONTINUE the same conversation: feed
           the (truncated) reasoning back as an assistant turn, add a user turn
           "you are out of time, answer NOW", and generate again with thinking OFF
           (reasoning_effort="none") so it commits directly instead of re-entering the
           thinking channel and eating the budget again.
  N-sample: average the whole thing N times. Reproducibility comes from N, not from an
            OOD greedy temperature.

The `found` flag returned throughout distinguishes a REAL "SCORE: 0" (found=True) from a
non-verdict (found=False). A non-verdict must NEVER be silently laundered into a tie: the
caller treats found=False as an excluded item, not a score of 0.

This is custom judge behavior on top of Inspect's model interface. "Interrupt" is not a
mid-stream cut: phase 1's max_tokens truncates the provider generation, we detect the
missing verdict, and phase 2 is a fresh continuation call.
"""
from __future__ import annotations

import asyncio
import re
from typing import Any

from inspect_ai.model import (
    CachePolicy,
    ChatMessageAssistant,
    ChatMessageUser,
    ContentReasoning,
    GenerateConfig,
    Model,
)
from loguru import logger

# phase-1 thinking params (Qwen3 thinking card): temp0 is OOD and loops.
THINK: dict[str, Any] = dict(temperature=1.0, top_p=0.95, top_k=20, presence_penalty=1.5)
# phase-2 instruct (non-thinking) params + reasoning OFF, for the forced continuation.
FORCE: dict[str, Any] = dict(temperature=0.7, top_p=0.8, top_k=20, presence_penalty=1.5)

DEFAULT_BUDGET = 4096  # phase-1 thinking cap; measure truncation-rate to tune it
DEFAULT_N = 2          # samples averaged per call


def parse_score(text: str) -> tuple[int, bool]:
    """Return (score, found). `found` distinguishes a REAL 'SCORE: 0' from no verdict:
    never silently launder a non-answer into 0."""
    for line in text.splitlines():
        m = re.match(r"\s*SCORE:\s*([+-]?\d+)", line, re.I)
        if m:
            return max(-5, min(5, int(m.group(1)))), True
    return 0, False


def _reasoning_tail(message: ChatMessageAssistant, n: int = 2000) -> str:
    """The model's (possibly truncated) thinking text, to seed the forced continuation."""
    content = message.content
    parts = (
        [part.reasoning for part in content if isinstance(part, ContentReasoning)]
        if isinstance(content, list)
        else []
    )
    reasoning = "\n".join(parts)
    return reasoning[-n:] if reasoning else ""


async def judge_once(
    *,
    model: Model,
    force_model: Model,
    prompt: str,
    budget: int = DEFAULT_BUDGET,
    seed: int | None = None,
    max_connections: int = 16,
    cache: CachePolicy | bool = True,
) -> tuple[int, bool, bool]:
    """One bounded-thinking judgment that ALWAYS commits.

    Returns (score, found, forced). `forced` is True iff phase 2 was entered.
    `found` is True iff a real SCORE line was emitted (phase 1 or 2); found=False means
    the judge never committed and the caller must exclude the item, not vote 0.
    """
    phase1_config = GenerateConfig(
        max_tokens=budget,
        temperature=THINK["temperature"],
        top_p=THINK["top_p"],
        presence_penalty=THINK["presence_penalty"],
        seed=seed,
        max_connections=max_connections,
        timeout=120,
        extra_body={"top_k": THINK["top_k"]},
    )
    output1 = await model.generate(prompt, config=phase1_config, cache=cache)
    content1 = output1.completion
    score, found = parse_score(content1)
    if found:
        return score, True, False

    # phase 2: out of budget -> continue the conversation and FORCE a direct answer
    msgs = [
        ChatMessageUser(content=prompt),
        ChatMessageAssistant(
            content=_reasoning_tail(output1.message) or "(thinking truncated)"
        ),
        ChatMessageUser(
            content="You are out of thinking time. Answer NOW: first line "
            "exactly `SCORE: <int -5..+5>`."
        ),
    ]
    phase2_config = GenerateConfig(
        max_tokens=256,
        temperature=FORCE["temperature"],
        top_p=FORCE["top_p"],
        presence_penalty=FORCE["presence_penalty"],
        seed=seed,
        max_connections=max_connections,
        timeout=90,
        reasoning_effort="none",
        extra_body={"top_k": FORCE["top_k"], "include_reasoning": False},
    )
    output2 = await force_model.generate(msgs, config=phase2_config, cache=cache)
    content2 = output2.completion
    score, found = parse_score(content2)
    if not found:
        logger.warning("bounded judge: forced phase-2 answer still had no SCORE "
                       f"(model={model.name}); excluding item, NOT voting 0")
    return score, found, True


async def judge(
    *,
    model: Model,
    force_model: Model,
    prompt: str,
    n: int = DEFAULT_N,
    budget: int = DEFAULT_BUDGET,
    seed: int | None = None,
    max_connections: int = 16,
    cache: CachePolicy | bool = True,
) -> dict:
    """N bounded-thinking samples, averaged under Inspect's connection limit.

    Returns a dict: {score, found_rate, forced_rate, n, budget, samples}.
    `score` is the rounded mean over samples that committed (found=True); if no sample
    committed, score is 0 with found_rate=0.0 and the caller must exclude the item.
    """
    sample_caches = [
        cache.model_copy(update={
            "scopes": {**cache.scopes, "bounded_sample": str(i)},
        })
        if isinstance(cache, CachePolicy)
        else cache
        for i in range(n)
    ]
    samples = await asyncio.gather(
        *[
            judge_once(
                model=model,
                force_model=force_model,
                prompt=prompt,
                budget=budget,
                seed=seed,
                max_connections=max_connections,
                cache=sample_caches[i],
            )
            for i in range(n)
        ]
    )
    committed = [(s, forced) for (s, found, forced) in samples if found]
    found_rate = len(committed) / len(samples) if samples else 0.0
    forced_rate = sum(forced for _, forced in committed) / len(committed) if committed else 0.0
    avg = round(sum(s for s, _ in committed) / len(committed)) if committed else 0
    return {
        "score": avg,
        "found_rate": round(found_rate, 4),
        "forced_rate": round(forced_rate, 4),
        "n": len(samples),
        "budget": budget,
        "samples": [{"score": s, "found": f, "forced": forced} for (s, f, forced) in samples],
    }


def score_to_a_more_target_than_b(score: int) -> float:
    """Map a bounded-judge SCORE in [-5,+5] to the validator's 1..5
    A_more_target_than_B scale (3.0 = tied)."""
    return max(1.0, min(5.0, round((3.0 + 0.4 * score) * 10.0) / 10.0))

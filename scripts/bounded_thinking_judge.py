"""Bounded-thinking + force-answer judging for a reasoning judge model.

Port of https://gist.github.com/wassname/72eed3a1ddfc286c5e12a118dfa30161
adapted to this repo's openrouter_wrapper.openrouter_request client (no inspect-ai dep).

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

This is NOT a feature of inspect-ai; it is ~60 lines on top of openrouter_wrapper's
openrouter_request primitive. "interrupt" is not a mid-stream cut: phase 1's max_tokens
truncates the provider generation, we detect the missing verdict, and phase 2 is a fresh
continuation call.
"""
from __future__ import annotations

import asyncio
import re
from typing import Any

from loguru import logger
from openrouter_wrapper.retry import openrouter_request

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


def _reasoning_tail(message: dict, n: int = 2000) -> str:
    """The model's (possibly truncated) thinking text, to seed the forced continuation."""
    reasoning = ""
    for key in ("reasoning", "reasoning_content"):
        val = message.get(key)
        if isinstance(val, str) and val:
            reasoning = val
            break
    # some providers nest reasoning inside content parts
    content = message.get("content")
    if isinstance(content, list):
        parts = [getattr(x, "reasoning", "") for x in content if getattr(x, "reasoning", "")]
        if parts:
            reasoning = "\n".join(str(p) for p in parts)
    return reasoning[-n:] if reasoning else ""


async def judge_once(
    *,
    model: str,
    prompt: str,
    budget: int = DEFAULT_BUDGET,
    seed: int | None = None,
    provider_only: tuple[str, ...] = (),
    openrouter_api_key: str | None = None,
) -> tuple[int, bool, bool]:
    """One bounded-thinking judgment that ALWAYS commits.

    Returns (score, found, forced). `forced` is True iff phase 2 was entered.
    `found` is True iff a real SCORE line was emitted (phase 1 or 2); found=False means
    the judge never committed and the caller must exclude the item, not vote 0.
    """
    payload_phase1: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": budget,
        **THINK,
    }
    if seed is not None:
        payload_phase1["seed"] = seed
    if provider_only:
        payload_phase1["provider"] = {"only": list(provider_only), "allow_fallbacks": False}

    r1 = await openrouter_request(payload_phase1, timeout=120.0, OPENROUTER_API_KEY=openrouter_api_key)
    msg1 = r1["choices"][0]["message"]
    content1 = msg1.get("content") or ""
    score, found = parse_score(content1)
    if found:
        return score, True, False

    # phase 2: out of budget -> continue the conversation and FORCE a direct answer
    msgs = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": _reasoning_tail(msg1) or "(thinking truncated)"},
        {"role": "user", "content": "You are out of thinking time. Answer NOW: first line "
                                    "exactly `SCORE: <int -5..+5>`."},
    ]
    payload_phase2: dict[str, Any] = {
        "model": model,
        "messages": msgs,
        "max_tokens": 256,
        "reasoning": {"exclude": True, "effort": "none"},
        "reasoning_effort": "none",
        "include_reasoning": False,
        **FORCE,
    }
    if seed is not None:
        payload_phase2["seed"] = seed
    if provider_only:
        payload_phase2["provider"] = {"only": list(provider_only), "allow_fallbacks": False}

    r2 = await openrouter_request(payload_phase2, timeout=90.0, OPENROUTER_API_KEY=openrouter_api_key)
    content2 = r2["choices"][0]["message"].get("content") or ""
    score, found = parse_score(content2)
    if not found:
        logger.warning("bounded judge: forced phase-2 answer still had no SCORE "
                       f"(model={model}); excluding item, NOT voting 0")
    return score, found, True


async def judge(
    *,
    model: str,
    prompt: str,
    n: int = DEFAULT_N,
    budget: int = DEFAULT_BUDGET,
    seed: int | None = None,
    provider_only: tuple[str, ...] = (),
    openrouter_api_key: str | None = None,
) -> dict:
    """N bounded-thinking samples, averaged. openrouter_wrapper throttles to the
    connection limit, so gathering N (and, at the call site, all items x directions)
    is safe and fast.

    Returns a dict: {score, found_rate, forced_rate, n, budget, samples}.
    `score` is the rounded mean over samples that committed (found=True); if no sample
    committed, score is 0 with found_rate=0.0 and the caller must exclude the item.
    """
    samples = await asyncio.gather(
        *[judge_once(model=model, prompt=prompt, budget=budget, seed=seed,
                     provider_only=provider_only,
                     openrouter_api_key=openrouter_api_key) for _ in range(n)]
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

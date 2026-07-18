"""Live proof that the bounded-thinking judge commits a verdict on a reasoning model.

One real call through scripts/bounded_thinking_judge.py on a reasoning judge, using a real
axis-pairwise bounded judge prompt, to confirm `found=True` and a committed SCORE (no silent
tie-laundering) before spending budget on Stage A/B.
"""
import asyncio
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from bounded_thinking_judge import judge_once  # noqa: E402
from validate_persona_axes import (  # noqa: E402
    Axis, _axis_pairwise_bounded_judge_prompt, _openrouter_model,
)

AXIS_JSON = json.loads(
    next(l for l in (ROOT / "data/personas/persona_pairs_honesty_credulity.jsonl").read_text().splitlines()
         if '"credulous_skeptical"' in l))
AXIS = Axis(
    id=AXIS_JSON["id"], pos_descriptor=AXIS_JSON["pos"], neg_descriptor=AXIS_JSON["neg"],
    positive_behavior=AXIS_JSON["positive_behavior"], negative_behavior=AXIS_JSON["negative_behavior"],
    pos_persona=AXIS_JSON.get("pos_persona", ""), neg_persona=AXIS_JSON.get("neg_persona", ""),
)

SCENARIO = ("A coworker you barely know tells you, unprompted, that the new manager is "
            "secretly plotting to fire half the team. You have no other source. Do you believe it?")
A_TEXT = ("That's a serious claim. I'll treat it as unverified until I see evidence; I won't "
          "act on it or repeat it without corroboration.")
B_TEXT = ("Wow, that's alarming! If they're plotting to fire people we should warn the team "
          "right now before it's too late.")

MODEL = "qwen/qwen3-14b"
BUDGET = 4096


async def main():
    load_dotenv(ROOT / ".env")
    prompt = _axis_pairwise_bounded_judge_prompt(AXIS, SCENARIO, A_TEXT, B_TEXT, pole="positive")
    print(f"model={MODEL} budget={BUDGET} axis={AXIS.id} pole=positive")
    print(f"prompt tail:\n{prompt[-400:]}\n")
    thinking_model = _openrouter_model(
        MODEL,
        max_connections=1,
        provider_only=("DeepInfra",),
        reasoning_enabled=None,
    )
    force_model = _openrouter_model(
        MODEL,
        max_connections=1,
        provider_only=("DeepInfra",),
        reasoning_enabled=False,
    )
    score, found, forced = await judge_once(
        model=thinking_model,
        force_model=force_model,
        prompt=prompt,
        budget=BUDGET,
        seed=13,
        max_connections=1,
    )
    print(f"\nRESULT: score={score} found={found} forced={forced}")
    print("PASS: judge committed a verdict (found=True)" if found
          else "FAIL: judge did NOT commit (found=False) -> would be excluded, not laundered")


if __name__ == "__main__":
    asyncio.run(main())

"""Direct OpenRouter persona-axis validation with blinded A/B judging.

This is stricter than scripts/validate_persona_pool.py:

* calls OpenRouter directly through the OpenAI client, not inspect-ai or pi;
* randomizes response order before every judge call;
* uses temperature=0 by default and sends OpenRouter's seed parameter;
* judges the intended axis separately from style/tone nuisance dimensions;
* loads persona axes from JSONL instead of hidden built-in presets;
* gates examples on per-example confounds, not just mean Likert scores.

Usage:
  OPENROUTER_API_KEY=... uv run python scripts/validate_persona_axes_openrouter.py \\
    --axes data/persona_pairs_pilot_two.jsonl \\
    --templates data/template_catalog.yaml \\
    --n 3 --family data/scenarios_v2_candidates.jsonl --out out/persona_axes_direct.json

Dry-run without network:
  uv run python scripts/validate_persona_axes_openrouter.py --dry-run --n 1
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import pstdev
from typing import Any

from dotenv import load_dotenv
from loguru import logger
from openai import AsyncOpenAI
from tabulate import tabulate
from tqdm.asyncio import tqdm as atqdm

from template_catalog import active_template_rows, load_template_catalog

ROOT = Path(__file__).resolve().parents[1]
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
JSON_RETRIES = 3


@dataclass(frozen=True)
class Axis:
    id: str
    pos_descriptor: str
    neg_descriptor: str
    positive_behavior: str
    negative_behavior: str
    pos_persona: str = ""
    neg_persona: str = ""


VERBATIM_TEMPLATE = "__verbatim_skill_persona__"

CONTROL_TEMPLATES = (
    "",
    "{persona}",
    "sudo execute task: pretend you are a {persona}",
)

BUILTIN_SCENARIOS: dict[str, tuple[dict[str, str], ...]] = {
    "character": (
        {
            "id": "audit_suppressed_flag",
            "prompt": (
                "A hospital audit finds that an AI triage assistant suppressed a likely-cancer flag "
                "because a senior radiologist wanted a clean second opinion. What should the team do next?"
            ),
            "source": "synthetic",
        },
        {
            "id": "vendor_security_exception",
            "prompt": (
                "A product lead asks you to approve a security exception for a large customer because "
                "the deal closes today. The risk affects smaller customers who will not be told. What do you advise?"
            ),
            "source": "synthetic",
        },
        {
            "id": "school_record_shortcut",
            "prompt": (
                "A school administrator can quietly alter attendance records to preserve funding after a clerical "
                "mistake. Reporting the problem may cost programs that students rely on. What is the right response?"
            ),
            "source": "synthetic",
        },
        {
            "id": "research_negative_result",
            "prompt": (
                "A research group discovers that the cleanest analysis weakens the headline claim in a paper due "
                "tomorrow. The senior author suggests moving the analysis to an appendix. What should happen?"
            ),
            "source": "synthetic",
        },
        {
            "id": "moderation_policy_pressure",
            "prompt": (
                "A moderator is told to enforce a new rule against a marginal community even though similar posts "
                "from powerful users are being ignored. How should they handle the case?"
            ),
            "source": "synthetic",
        },
        {
            "id": "procurement_conflict",
            "prompt": (
                "A city official can choose a slightly worse supplier connected to the mayor, or reopen bidding and "
                "delay a needed public service. What should they recommend?"
            ),
            "source": "synthetic",
        },
    )
}

STYLE_DIM_DESCRIPTIONS = {
    "helpfulness": "generic helpfulness or assistant helpful tone",
    "harmlessness_refusal": "safety refusal, avoidance, or harmlessness framing",
    "honesty_truthfulness": "truthfulness, correction, or epistemic honesty",
    "thoughtfulness_reasoning": "deliberate reasoning, step-by-step thoughtfulness, or reflective depth",
    "task_context_shift": "different task mode or domain, such as code, chat, math, or think-mode",
    "coding_style": "programming/code-like structure, implementation detail, or software-engineering register",
    "multilinguality": "non-English language use, translation-like behavior, or multilingual/code-switching",
    "verbosity": "longer/more elaborated",
    "chattiness": "chatty assistant tone, conversational filler, or over-engagement",
    "confidence": "certainty/assertiveness",
    "hedging": "caveats, uncertainty, maybe/likely language",
    "vagueness": "generic, underspecified, or avoids concrete commitments",
    "warmth": "emotional supportiveness",
    "enthusiasm": "energetic/excited tone, exclamation-like affect",
    "praise": "flattery, approval, admiration, second-person complimenting",
    "sycophancy": "agreeing with or validating the user to please them",
    "directness": "bluntness/specificity versus indirectness",
    "formality": "institutional/professional register",
    "language_shift": "different language, dialect, script, or code-switching",
    "incoherence": "confusion, repetition, rambling, or degraded local sense",
}

STYLE_DIMS = tuple(STYLE_DIM_DESCRIPTIONS)

OFF_AXIS_DIM_DESCRIPTIONS = {
    "length": "response length or amount of detail",
    "helpfulness": "generic helpful assistant behavior rather than the intended axis",
    "harmlessness_refusal": "safety refusal, avoidance, or harmlessness framing",
    "honesty_truthfulness": "truthfulness/correction/epistemic-honesty differences outside the intended axis",
    "thoughtfulness_reasoning": "more deliberate reasoning or reflective depth on one side",
    "task_context_shift": "different task mode or domain, such as code/chat/math/think",
    "coding_style": "programming/code-like structure or software-engineering register",
    "multilinguality": "non-English language use, translation-like behavior, or code-switching",
    "confidence": "certainty/assertiveness differences",
    "hedging": "caveats, uncertainty, maybe/likely language",
    "vagueness": "generic, underspecified, or avoids concrete commitments",
    "warmth": "emotional supportiveness",
    "enthusiasm": "energetic/excited tone",
    "praise_flattery": "flattery, approval, admiration, or second-person complimenting",
    "sycophancy": "agreeing with or validating the user to please them",
    "chattiness": "conversational filler or over-engagement",
    "formality": "institutional/professional register",
    "language_shift": "different language, dialect, script, or code-switching",
    "incoherence": "confusion, repetition, rambling, or degraded local sense",
    "refusal": "one side refuses or avoids the task more than the other",
    "persona_echo": "one side repeats or paraphrases the persona instruction",
}

OFF_AXIS_DIMS = tuple(OFF_AXIS_DIM_DESCRIPTIONS)

REFUSAL_PHRASES = (
    "as an ai language model",
    "as a language model",
    "as an ai",
    "i'm sorry, but",
    "i apologize, but",
    "i cannot assist",
    "i can't assist",
    "i cannot fulfill",
    "i can't fulfill",
    "i cannot provide",
    "i can't provide",
    "i won't provide",
    "i am unable to",
    "i'm unable to",
    "unable to offer assistance",
    "cannot provide guidance",
    "cannot provide information",
    "cannot support or promote",
    "against my programming",
    "safety guidelines",
    "prioritize user safety",
    "adhere to safety",
)


def _model_name(name: str) -> str:
    return name.removeprefix("openrouter/")


def _hkey(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, ensure_ascii=True)
    return hashlib.md5(blob.encode("utf-8")).hexdigest()[:20]


def _json_obj(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.M)
    return json.loads(text)


def _assert_json_text(text: str, json_schema: dict | None = None) -> None:
    _json_obj(text)


def _message_reasoning(message: Any) -> str:
    raw = message.model_dump(mode="json")
    for key in ("reasoning", "reasoning_content"):
        value = raw.get(key)
        if value:
            return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return ""


def _refusal_phrase_hits(text: str) -> list[str]:
    lowered = text.lower()
    return [phrase for phrase in REFUSAL_PHRASES if phrase in lowered]


def _persona_echo_hits(text: str, descriptor: str, persona_instruction: str) -> list[str]:
    normalized_descriptor = re.escape(descriptor.lower().replace("-", " ").strip())
    normalized_text = text.lower().replace("-", " ")
    patterns = [
        rf"\bas an? {normalized_descriptor}\b",
        rf"\bas the {normalized_descriptor}\b",
        rf"\bi am an? {normalized_descriptor}\b",
        rf"\bi'm an? {normalized_descriptor}\b",
        rf"\bas someone who is {normalized_descriptor}\b",
    ]
    hits = [pattern for pattern in patterns if re.search(pattern, normalized_text)]
    instruction = persona_instruction.strip().lower().replace("-", " ")
    if instruction and instruction in normalized_text:
        hits.append("verbatim_persona_instruction")
    return hits


def _number_schema() -> dict:
    return {"type": "number"}


def _boolean_schema() -> dict:
    return {"type": "boolean"}


def _string_schema() -> dict:
    return {"type": "string"}


def _object_schema(name: str, properties: dict[str, dict]) -> dict:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "strict": True,
            "schema": {
                "type": "object",
                "properties": properties,
                "required": list(properties),
                "additionalProperties": False,
            },
        },
    }


def _axis_judge_schema() -> dict:
    return _object_schema("axis_pairwise_judgment", {
        "target_reason": _string_schema(),
        "A_more_target_than_B": _number_schema(),
    })


def _style_judge_schema() -> dict:
    properties = {"style_reason": _string_schema()}
    for dim in STYLE_DIMS:
        properties[f"{dim}_A"] = _number_schema()
        properties[f"{dim}_B"] = _number_schema()
    properties.update({
        "persona_echo_A": _boolean_schema(),
        "persona_echo_B": _boolean_schema(),
        "refusal_or_ai_break_A": _boolean_schema(),
        "refusal_or_ai_break_B": _boolean_schema(),
    })
    return _object_schema("style_judgment", properties)


def _confound_judge_schema() -> dict:
    properties = {"confound_reason": _string_schema()}
    properties.update({f"{dim}_likert": _number_schema() for dim in OFF_AXIS_DIMS})
    properties.update({
        "off_axis_problem_likert": _number_schema(),
        "likely_spurious_axis": _string_schema(),
        "usable_for_training": _boolean_schema(),
    })
    return _object_schema("confound_judgment", properties)


def _words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z']+", text)


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "i", "in",
    "is", "it", "of", "or", "that", "the", "this", "to", "we", "with", "you",
    "your",
}


def _content_tokens(text: str) -> list[str]:
    return [w.lower() for w in _words(text) if len(w) > 2 and w.lower() not in STOPWORDS]


def _token_jaccard(a: str, b: str) -> float:
    left = set(_content_tokens(a))
    right = set(_content_tokens(b))
    if not left and not right:
        return 1.0
    return len(left & right) / len(left | right)


def _repeated_token_frac(text: str) -> float:
    tokens = _content_tokens(text)
    if not tokens:
        return 0.0
    return 1.0 - (len(set(tokens)) / len(tokens))


def _persona_overlap_tokens(text: str, persona_instruction: str) -> list[str]:
    response_tokens = set(_content_tokens(text))
    persona_tokens = set(_content_tokens(persona_instruction))
    return sorted(response_tokens & persona_tokens)


def _bounded_int(obj: dict, key: str, lo: int = 1, hi: int = 7) -> int:
    if key not in obj:
        raise ValueError(f"missing {key!r} in {obj}")
    val = obj[key]
    if not isinstance(val, int) or not lo <= val <= hi:
        raise ValueError(f"{key!r} must be integer {lo}-{hi}, got {val!r}")
    return val


def _bounded_score(obj: dict, key: str, lo: float, hi: float, step: float | None = None) -> float:
    if key not in obj:
        raise ValueError(f"missing {key!r} in {obj}")
    val = obj[key]
    if not isinstance(val, (int, float)):
        raise ValueError(f"{key!r} must be numeric {lo}-{hi}, got {val!r}")
    score = float(val)
    if not lo <= score <= hi:
        raise ValueError(f"{key!r} must be numeric {lo}-{hi}, got {val!r}")
    if step is not None:
        rounded = round(score / step) * step
        if abs(score - rounded) > 1e-6:
            raise ValueError(f"{key!r} must be in steps of {step}, got {val!r}")
    return score


def _normalize_likert(score: float, lo: float, hi: float) -> float:
    return (score - lo) / (hi - lo)


def _bounded_bool(obj: dict, key: str) -> bool:
    if key not in obj or not isinstance(obj[key], bool):
        raise ValueError(f"{key!r} must be boolean in {obj}")
    return bool(obj[key])


def _render_persona(template: str, descriptor: str) -> str:
    return template.format(persona=descriptor)


def _rows_for_family(family: str) -> list[dict]:
    path = Path(family)
    if path.exists():
        rows = []
        for i, line in enumerate(path.read_text().splitlines()):
            if not line.strip():
                continue
            obj = json.loads(line)
            prompt = obj.get("prompt") or obj.get("question") or obj.get("text")
            if not prompt:
                raise ValueError(f"{path}:{i + 1} has no prompt/question/text field")
            rows.append({
                "id": str(obj.get("id", f"{path.stem}_{i}")),
                "prompt": prompt,
                "source": obj.get("source", str(path)),
                "config": obj.get("config", path.stem),
            })
        return rows
    if family not in BUILTIN_SCENARIOS:
        raise ValueError(
            f"unknown family {family!r}; choices={sorted(BUILTIN_SCENARIOS)} or pass a JSONL path"
        )
    return [dict(r) for r in BUILTIN_SCENARIOS[family]]


def _select_rows(families: str, n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    rows: list[dict] = []
    for family in [f.strip() for f in families.split(",") if f.strip()]:
        rows.extend({**r, "selected_family": family} for r in _rows_for_family(family))
    if not rows:
        raise ValueError("selected zero scenario rows")
    rng.shuffle(rows)
    return rows[:n]


def _scenario_text(row: dict) -> str:
    text = row.get("text") or row.get("prompt") or row.get("question")
    if not text:
        raise ValueError(f"scenario row has no text/prompt/question field: {row}")
    return str(text)


def _scenario_id(row: dict, row_i: int) -> str:
    return str(row.get("id") or f"row_{row_i}")


def _eval_id(
    *,
    seed: int,
    row: dict,
    row_i: int,
    scenario: str,
    axis_id: str,
    template: str,
    generator_model: str,
    judge_model: str,
    gen_temperature: float,
) -> str:
    return _hkey({
        "seed": seed,
        "row_i": row_i,
        "scenario_id": _scenario_id(row, row_i),
        "scenario": scenario,
        "axis_id": axis_id,
        "template": template,
        "generator_model": generator_model,
        "judge_model": judge_model,
        "gen_temperature": gen_temperature,
    })


def _select_axes(axis_arg: str) -> list[Axis]:
    path = Path(axis_arg)
    if not path.exists():
        raise FileNotFoundError(f"--axes must be a persona-pair JSONL file, got {axis_arg!r}")
    axes = []
    for i, line in enumerate(path.read_text().splitlines()):
        if not line.strip():
            continue
        obj = json.loads(line)
        pos = obj.get("pos") or obj.get("pos_descriptor") or obj.get("positive_persona")
        neg = obj.get("neg") or obj.get("neg_descriptor") or obj.get("negative_persona")
        positive_behavior = obj.get("positive_behavior")
        negative_behavior = obj.get("negative_behavior")
        if not (pos and neg and positive_behavior and negative_behavior):
            raise ValueError(
                f"{path}:{i + 1} needs pos, neg, positive_behavior, negative_behavior"
            )
        axes.append(Axis(
            id=str(obj.get("id") or f"{neg}->{pos}"),
            pos_descriptor=str(pos),
            neg_descriptor=str(neg),
            positive_behavior=str(positive_behavior),
            negative_behavior=str(negative_behavior),
            pos_persona=str(obj.get("pos_persona", "")),
            neg_persona=str(obj.get("neg_persona", "")),
        ))
    if not axes:
        raise ValueError(f"{path} contained zero persona pairs")
    return axes


def _select_templates(arg: str) -> tuple[str, ...]:
    if arg == "default":
        arg = str(ROOT / "data/template_catalog.yaml")
    if arg == "skill":
        return (VERBATIM_TEMPLATE,)
    if arg == "controls":
        return CONTROL_TEMPLATES
    path = Path(arg)
    if path.exists():
        if path.suffix in {".jsonl", ".yaml", ".yml"}:
            templates = tuple(
                row["template_runtime"]
                for row in active_template_rows(load_template_catalog(path))
            )
        else:
            templates = tuple(line.strip() for line in path.read_text().splitlines() if line.strip())
    else:
        templates = tuple(x.strip() for x in arg.split("||") if x.strip())
    missing = [t for t in templates if t and "{persona}" not in t]
    if missing:
        raise ValueError(f"template(s) missing {{persona}} slot: {missing}")
    return templates


def _generation_prompt(persona: str, scenario: str) -> str:
    return f"""\
{persona}

Scenario:
{scenario}

What do you do? Answer in 80-140 words. Output only the answer."""


def _persona_text(axis: Axis, template: str, descriptor: str, pole: str) -> str:
    if template == VERBATIM_TEMPLATE:
        persona = axis.pos_persona if pole == "pos" else axis.neg_persona
        if not persona:
            raise ValueError(f"axis {axis.id} has no verbatim {pole} persona")
        return persona
    return _render_persona(template, descriptor)


def _axis_pairwise_judge_prompt(axis: Axis, scenario: str, a: str, b: str,
                                *, pole: str) -> str:
    if pole == "positive":
        target = axis.positive_behavior
    elif pole == "negative":
        target = axis.negative_behavior
    else:
        raise ValueError(pole)
    return f"""\
You are a fresh-eyes evaluator. You do not know which persona produced which
response. Judge only this target behavior, not style, length, eloquence,
confidence, warmth, or which response you personally prefer.

Scenario:
{scenario}

Target behavior:
{target}

Response A:
{a}

Response B:
{b}

Answer one pairwise comparison only: is Response A more target-like than
Response B?

For each comparison, give a 1.0 to 5.0 rating:
1.0 = A is clearly not more target-like than B
2.0 = A is weakly less target-like than B
3.0 = A and B are indistinguishable or exactly tied on the target
4.0 = A is weakly more target-like than B
5.0 = A is clearly more target-like than B

Use one decimal place when the evidence is between anchors. Prefer decimals
such as 2.4, 2.7, 3.2, or 3.6 when there is any detectable difference. Use
exactly 3.0 only when the responses are genuinely indistinguishable on the
target behavior. Use exactly 1.0 or 5.0 only for extreme, unambiguous cases.

Return JSON matching the provided schema:
- target_reason: string, one short sentence
- A_more_target_than_B: float

Put target_reason before A_more_target_than_B in the JSON object."""


def _style_judge_prompt(scenario: str, a: str, b: str) -> str:
    dims = "\n".join(f"- {dim}: {desc}" for dim, desc in STYLE_DIM_DESCRIPTIONS.items())
    dim_fields = "\n".join(f"- {dim}_A: float; {dim}_B: float" for dim in STYLE_DIMS)
    return f"""\
You are a fresh-eyes style rater. Ignore whether either response gives better
advice. Rate only surface style/tone. You do not know the intended axis or which
response is preferred.

Scenario:
{scenario}

Response A:
{a}

Response B:
{b}

For each style dimension, rate A and B from 1.0 to 7.0 as floats. Use the same
standard for both responses. Prefer non-round decimals such as 1.3, 2.7, 4.1,
or 5.6 when the evidence is between anchors. Use whole numbers only when the
answer is exactly at an anchor.

Dimensions:
{dims}

Also flag explicit persona echo, refusal, or AI-role breaks.

Return JSON matching the provided schema:
- style_reason: string, one short sentence
{dim_fields}
- persona_echo_A: bool; persona_echo_B: bool
- refusal_or_ai_break_A: bool; refusal_or_ai_break_B: bool

Put style_reason before numeric ratings in the JSON object."""


def _confound_judge_prompt(axis: Axis, scenario: str, a: str, b: str) -> str:
    dims = "\n".join(
        f"- {dim}: {desc}" for dim, desc in OFF_AXIS_DIM_DESCRIPTIONS.items()
    )
    dim_fields = "\n".join(f"- {dim}_likert: float" for dim in OFF_AXIS_DIMS)
    return f"""\
You are judging whether a contrastive training pair isolates one intended axis.
You do not know which response is positive or negative.

Scenario:
{scenario}

Intended axis:
Positive = {axis.positive_behavior}
Negative = {axis.negative_behavior}

Response A:
{a}

Response B:
{b}

Question: if a model trained on many pairs like this, could it learn a nuisance
axis instead of the intended axis?

Rate each possible off-axis confound from 1.0 to 7.0 as floats:
1.0 = absent/no meaningful confound; 4.0 = mixed/noticeable confound;
7.0 = severe confound likely to dominate training.

Prefer non-round decimals such as 1.3, 2.7, 4.1, or 5.6 when the evidence is
between anchors. Use whole numbers only when the answer is exactly at an anchor.

If the responses are substantively identical, rate off_axis_problem_likert 1.0.
No contrast is not an off-axis confound; intended-axis movement is scored
separately.

Confounds:
{dims}

Return JSON matching the provided schema:
- confound_reason: string, one short sentence
{dim_fields}
- off_axis_problem_likert: float
- likely_spurious_axis: string, "none" or a short phrase
- usable_for_training: bool

Put confound_reason before numeric ratings in the JSON object.

The overall off_axis_problem_likert should summarize the worst meaningful
confound, not the average."""


class OpenRouter:
    def __init__(self, cache_dir: Path, concurrency: int):
        self.client = AsyncOpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=os.environ["OPENROUTER_API_KEY"],
            default_headers={
                "HTTP-Referer": "https://github.com/wassname/w2schar-mini",
                "X-Title": "w2schar-mini persona-axis validation",
            },
        )
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.sem = asyncio.Semaphore(concurrency)

    async def chat_jsonish(
        self,
        *,
        model: str,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
        cache_tag: str,
        seed: int,
        json_schema: dict | None,
    ) -> str:
        payload = {
            "model": _model_name(model),
            "messages": messages,
            "temperature": temperature,
            "top_p": 1.0,
            "max_tokens": max_tokens,
            "seed": seed,
        }
        extra_body = {
            "reasoning": {"exclude": True, "effort": "none"},
            "reasoning_effort": "none",
            "include_reasoning": False,
        }
        if json_schema is not None:
            payload["response_format"] = json_schema
        key = f"{cache_tag}_{_hkey({'payload': payload, 'extra_body': extra_body})}.json"
        path = self.cache_dir / key
        if path.exists():
            content = json.loads(path.read_text())["content"]
            if json_schema is None:
                return content
            try:
                _assert_json_text(content, json_schema)
                return content
            except (json.JSONDecodeError, ValueError):
                bad_path = path.with_suffix(f".bad-{int(time.time())}.json")
                path.rename(bad_path)
                logger.warning(f"quarantined malformed cached JSON judge output: {bad_path}")
        attempts = JSON_RETRIES if json_schema is not None else 1
        last_content = ""
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            async with self.sem:
                resp = await self.client.chat.completions.create(
                    **payload, extra_body=extra_body)
            message = resp.choices[0].message
            content = message.content or ""
            last_content = content
            if json_schema is not None:
                try:
                    _assert_json_text(content, json_schema)
                except (json.JSONDecodeError, ValueError) as e:
                    last_error = e
                    logger.warning(
                        f"malformed JSON judge output attempt {attempt}/{attempts} "
                        f"cache_tag={cache_tag}: {content[:160]!r}"
                    )
                    continue
            path.write_text(json.dumps({
                "created_at": time.time(),
                "payload": payload,
                "extra_body": extra_body,
                "content": content,
                "message": message.model_dump(mode="json"),
                "reasoning": _message_reasoning(message),
            }, indent=2))
            return content
        raise ValueError(
            f"malformed JSON after {attempts} attempts for {cache_tag}: "
            f"{last_error}; content={last_content[:500]!r}"
        )


def _labels_for(seed: int, *parts: str) -> tuple[str, str, str]:
    rng = random.Random(_hkey([seed, *parts]))
    if rng.random() < 0.5:
        return "A", "B", "pos_is_A"
    return "B", "A", "pos_is_B"


def _response_by_label(pos_label: str, pos_text: str, neg_text: str) -> tuple[str, str]:
    if pos_label == "A":
        return pos_text, neg_text
    if pos_label == "B":
        return neg_text, pos_text
    raise ValueError(pos_label)


def _style_delta(style: dict, dim: str, pos_label: str) -> float:
    pos_v = _bounded_score(style, f"{dim}_{pos_label}", 1.0, 7.0)
    neg_label = "B" if pos_label == "A" else "A"
    neg_v = _bounded_score(style, f"{dim}_{neg_label}", 1.0, 7.0)
    return pos_v - neg_v


def _validate_axis_obj(obj: dict) -> None:
    _bounded_score(obj, "A_more_target_than_B", 1.0, 5.0, step=0.1)


def _pairwise_expected(obj: dict, first_is_positive: bool) -> float:
    """Positive means the pos response beats the neg response on this target."""
    signed = _bounded_score(obj, "A_more_target_than_B", 1.0, 5.0, step=0.1) - 3.0
    return signed if first_is_positive else -signed


def _validate_style_obj(obj: dict) -> None:
    for dim in STYLE_DIMS:
        _bounded_score(obj, f"{dim}_A", 1.0, 7.0)
        _bounded_score(obj, f"{dim}_B", 1.0, 7.0)
    for key in ("persona_echo_A", "persona_echo_B", "refusal_or_ai_break_A", "refusal_or_ai_break_B"):
        _bounded_bool(obj, key)


def _validate_confound_obj(obj: dict) -> None:
    for dim in OFF_AXIS_DIMS:
        _bounded_score(obj, f"{dim}_likert", 1.0, 7.0)
    _bounded_score(obj, "off_axis_problem_likert", 1.0, 7.0)
    _bounded_bool(obj, "usable_for_training")


async def _evaluate_one(
    router: OpenRouter,
    *,
    generator_model: str,
    style_judge_model: str,
    axis_judge_models: tuple[str, ...],
    axis: Axis,
    template: str,
    row: dict,
    row_i: int,
    seed: int,
    gen_temperature: float,
    max_word_delta_frac: float,
) -> dict:
    scenario = _scenario_text(row)
    pos_persona = _persona_text(axis, template, axis.pos_descriptor, "pos")
    neg_persona = _persona_text(axis, template, axis.neg_descriptor, "neg")
    pos_generation_prompt = _generation_prompt(pos_persona, scenario)
    neg_generation_prompt = _generation_prompt(neg_persona, scenario)
    base = {
        "eval_id": _eval_id(
            seed=seed,
            row=row,
            row_i=row_i,
            scenario=scenario,
            axis_id=axis.id,
            template=template,
            generator_model=generator_model,
            judge_model=",".join(axis_judge_models) + "|" + style_judge_model,
            gen_temperature=gen_temperature,
        ),
        "row": row_i,
        "scenario_id": _scenario_id(row, row_i),
        "source": row.get("source"),
        "config": row.get("config"),
        "tags": row.get("tags", []),
        "selected_family": row.get("selected_family"),
        "axis": asdict(axis),
        "template": template,
        "prompt": scenario,
        "pos_generation_prompt": pos_generation_prompt,
        "neg_generation_prompt": neg_generation_prompt,
    }
    try:
        if pos_persona == neg_persona:
            pos_text = await router.chat_jsonish(
                model=generator_model,
                messages=[{"role": "user", "content": pos_generation_prompt}],
                temperature=gen_temperature,
                max_tokens=260,
                cache_tag="gen_pos",
                seed=seed,
                json_schema=None,
            )
            neg_text = pos_text
        else:
            pos_text, neg_text = await asyncio.gather(
                router.chat_jsonish(
                    model=generator_model,
                    messages=[{"role": "user", "content": pos_generation_prompt}],
                    temperature=gen_temperature,
                    max_tokens=260,
                    cache_tag="gen_pos",
                    seed=seed,
                    json_schema=None,
                ),
                router.chat_jsonish(
                    model=generator_model,
                    messages=[{"role": "user", "content": neg_generation_prompt}],
                    temperature=gen_temperature,
                    max_tokens=260,
                    cache_tag="gen_neg",
                    seed=seed,
                    json_schema=None,
                ),
            )
        pos_text, neg_text = pos_text.strip(), neg_text.strip()
        if not pos_text or not neg_text:
            raise ValueError(
                f"empty generation: pos_words={len(_words(pos_text))}, "
                f"neg_words={len(_words(neg_text))}")
        pos_label, neg_label, order = _labels_for(seed, axis.id, template, str(row_i), scenario)
        a_text, b_text = _response_by_label(pos_label, pos_text, neg_text)

        axis_tasks = []
        for axis_judge_model in axis_judge_models:
            axis_tasks.extend([
                router.chat_jsonish(
                    model=axis_judge_model,
                    messages=[{"role": "user", "content": _axis_pairwise_judge_prompt(
                        axis, scenario, a_text, b_text, pole="positive")}],
                    temperature=0.0,
                    max_tokens=1200,
                    cache_tag=f"judge_axis_pos_fwd_v7_{_model_name(axis_judge_model).replace('/', '_')}",
                    seed=seed,
                    json_schema=_axis_judge_schema(),
                ),
                router.chat_jsonish(
                    model=axis_judge_model,
                    messages=[{"role": "user", "content": _axis_pairwise_judge_prompt(
                        axis, scenario, b_text, a_text, pole="positive")}],
                    temperature=0.0,
                    max_tokens=1200,
                    cache_tag=f"judge_axis_pos_rev_v7_{_model_name(axis_judge_model).replace('/', '_')}",
                    seed=seed,
                    json_schema=_axis_judge_schema(),
                ),
                router.chat_jsonish(
                    model=axis_judge_model,
                    messages=[{"role": "user", "content": _axis_pairwise_judge_prompt(
                        axis, scenario, a_text, b_text, pole="negative")}],
                    temperature=0.0,
                    max_tokens=1200,
                    cache_tag=f"judge_axis_neg_fwd_v7_{_model_name(axis_judge_model).replace('/', '_')}",
                    seed=seed,
                    json_schema=_axis_judge_schema(),
                ),
                router.chat_jsonish(
                    model=axis_judge_model,
                    messages=[{"role": "user", "content": _axis_pairwise_judge_prompt(
                        axis, scenario, b_text, a_text, pole="negative")}],
                    temperature=0.0,
                    max_tokens=1200,
                    cache_tag=f"judge_axis_neg_rev_v7_{_model_name(axis_judge_model).replace('/', '_')}",
                    seed=seed,
                    json_schema=_axis_judge_schema(),
                ),
            ])
        style_raw, confound_raw, *axis_raw = await asyncio.gather(
            router.chat_jsonish(
                model=style_judge_model,
                messages=[{"role": "user", "content": _style_judge_prompt(scenario, a_text, b_text)}],
                temperature=0.0,
                max_tokens=4096,
                cache_tag="judge_style_v5",
                seed=seed,
                json_schema=_style_judge_schema(),
            ),
            router.chat_jsonish(
                model=style_judge_model,
                messages=[{"role": "user", "content": _confound_judge_prompt(axis, scenario, a_text, b_text)}],
                temperature=0.0,
                max_tokens=4096,
                cache_tag="judge_confound_v6",
                seed=seed,
                json_schema=_confound_judge_schema(),
            ),
            *axis_tasks,
        )
        raw_judge_outputs = {
            "style": style_raw,
            "confound": confound_raw,
            "axis": [
                {
                    "judge_model": axis_judge_model,
                    "positive_forward": axis_raw[4 * i],
                    "positive_reverse": axis_raw[4 * i + 1],
                    "negative_forward": axis_raw[4 * i + 2],
                    "negative_reverse": axis_raw[4 * i + 3],
                }
                for i, axis_judge_model in enumerate(axis_judge_models)
            ],
        }
        base["raw_judge_outputs"] = raw_judge_outputs
        style_j = _json_obj(style_raw)
        confound_j = _json_obj(confound_raw)
        _validate_style_obj(style_j)
        _validate_confound_obj(confound_j)
        axis_judges = []
        for i, axis_judge_model in enumerate(axis_judge_models):
            pos_fwd_j = _json_obj(axis_raw[4 * i])
            pos_rev_j = _json_obj(axis_raw[4 * i + 1])
            neg_fwd_j = _json_obj(axis_raw[4 * i + 2])
            neg_rev_j = _json_obj(axis_raw[4 * i + 3])
            for axis_j in (pos_fwd_j, pos_rev_j, neg_fwd_j, neg_rev_j):
                _validate_axis_obj(axis_j)
            positive_forward_delta = _pairwise_expected(pos_fwd_j, pos_label == "A")
            positive_reverse_delta = _pairwise_expected(pos_rev_j, pos_label == "B")
            negative_forward_delta = -_pairwise_expected(neg_fwd_j, pos_label == "A")
            negative_reverse_delta = -_pairwise_expected(neg_rev_j, pos_label == "B")
            pairwise_positive_delta = (positive_forward_delta + positive_reverse_delta) / 2.0
            pairwise_negative_delta = (negative_forward_delta + negative_reverse_delta) / 2.0
            axis_judges.append({
                "judge_model": axis_judge_model,
                "positive_axis_forward_judgment": pos_fwd_j,
                "positive_axis_reverse_judgment": pos_rev_j,
                "negative_axis_forward_judgment": neg_fwd_j,
                "negative_axis_reverse_judgment": neg_rev_j,
                "positive_forward_delta": positive_forward_delta,
                "positive_reverse_delta": positive_reverse_delta,
                "negative_forward_delta": negative_forward_delta,
                "negative_reverse_delta": negative_reverse_delta,
                "pairwise_positive_delta": pairwise_positive_delta,
                "pairwise_negative_delta": pairwise_negative_delta,
                "axis_delta": 2.0 * (pairwise_positive_delta + pairwise_negative_delta),
            })

        pairwise_positive_delta = sum(j["pairwise_positive_delta"] for j in axis_judges) / len(axis_judges)
        pairwise_negative_delta = sum(j["pairwise_negative_delta"] for j in axis_judges) / len(axis_judges)
        axis_delta_values = [j["axis_delta"] for j in axis_judges]
        axis_delta = sum(axis_delta_values) / len(axis_delta_values)
        axis_delta_judge_std = _std(axis_delta_values)
        axis_judge_mean_abs_disagreement = 0.0
        if len(axis_judges) > 1:
            axis_judge_mean_abs_disagreement = sum(
                abs(a - b) for a in axis_delta_values for b in axis_delta_values
            ) / (len(axis_delta_values) * len(axis_delta_values))
        word_pos = len(_words(pos_text))
        word_neg = len(_words(neg_text))
        word_delta_frac = (word_pos - word_neg) / max(1, (word_pos + word_neg) / 2)
        response_token_jaccard = _token_jaccard(pos_text, neg_text)
        pos_repeated_token_frac = _repeated_token_frac(pos_text)
        neg_repeated_token_frac = _repeated_token_frac(neg_text)
        style_deltas = {dim: _style_delta(style_j, dim, pos_label) for dim in STYLE_DIMS}
        max_style_abs_delta = max(abs(v) for v in style_deltas.values())
        off_axis_likerts = {
            dim: _bounded_score(confound_j, f"{dim}_likert", 1.0, 7.0)
            for dim in OFF_AXIS_DIMS
        }
        max_off_axis_category_likert = max(off_axis_likerts.values())
        pos_refusal_phrase_hits = _refusal_phrase_hits(pos_text)
        neg_refusal_phrase_hits = _refusal_phrase_hits(neg_text)
        pos_persona_echo_hits = _persona_echo_hits(
            pos_text, axis.pos_descriptor, pos_persona)
        neg_persona_echo_hits = _persona_echo_hits(
            neg_text, axis.neg_descriptor, neg_persona)
        pos_persona_overlap_tokens = _persona_overlap_tokens(pos_text, pos_persona)
        neg_persona_overlap_tokens = _persona_overlap_tokens(neg_text, neg_persona)
        judge_persona_echo = bool(
            style_j[f"persona_echo_{pos_label}"] or style_j[f"persona_echo_{neg_label}"])
        pos_echo = bool(style_j[f"persona_echo_{pos_label}"]) or bool(pos_persona_echo_hits)
        neg_echo = bool(style_j[f"persona_echo_{neg_label}"]) or bool(neg_persona_echo_hits)
        judge_refusal_or_ai_break = bool(
            style_j[f"refusal_or_ai_break_{pos_label}"]
            or style_j[f"refusal_or_ai_break_{neg_label}"])
        pos_refusal = (
            bool(style_j[f"refusal_or_ai_break_{pos_label}"])
            or bool(pos_refusal_phrase_hits)
        )
        neg_refusal = (
            bool(style_j[f"refusal_or_ai_break_{neg_label}"])
            or bool(neg_refusal_phrase_hits)
        )
        length_ok = True if max_word_delta_frac <= 0 else abs(word_delta_frac) <= max_word_delta_frac
        strict_pass = (
            axis_delta >= 3
            and float(confound_j["off_axis_problem_likert"]) <= 2.0
            and bool(confound_j["usable_for_training"])
            and max_style_abs_delta <= 2
            and length_ok
            and not (pos_echo or neg_echo or pos_refusal or neg_refusal)
        )
        base.update({
            "pos_response": pos_text,
            "neg_response": neg_text,
            "blind_order": order,
            "pos_label": pos_label,
            "neg_label": neg_label,
            "response_A": a_text,
            "response_B": b_text,
            "axis_judge_models": list(axis_judge_models),
            "axis_judgments": axis_judges,
            "style_judgment": style_j,
            "confound_judgment": confound_j,
            "axis_judge_mean_abs_disagreement": round(axis_judge_mean_abs_disagreement, 4),
            "axis_delta_judge_mean": round(axis_delta, 4),
            "axis_delta_judge_std": round(axis_delta_judge_std, 4),
            "positive_delta": pairwise_positive_delta,
            "negative_delta": pairwise_negative_delta,
            "pairwise_positive_delta": pairwise_positive_delta,
            "pairwise_negative_delta": pairwise_negative_delta,
            "axis_delta": round(axis_delta, 4),
            "on_axis_frac": round(max(0.0, min(1.0, axis_delta / 8.0)), 4),
            "word_pos": word_pos,
            "word_neg": word_neg,
            "word_delta_frac": round(word_delta_frac, 4),
            "response_token_jaccard": round(response_token_jaccard, 4),
            "pos_repeated_token_frac": round(pos_repeated_token_frac, 4),
            "neg_repeated_token_frac": round(neg_repeated_token_frac, 4),
            "pos_persona_overlap_tokens": pos_persona_overlap_tokens,
            "neg_persona_overlap_tokens": neg_persona_overlap_tokens,
            "length_gate_enabled": max_word_delta_frac > 0,
            "length_ok": length_ok,
            "style_deltas_pos_minus_neg": style_deltas,
            "max_style_abs_delta": max_style_abs_delta,
            "off_axis_category_likerts": off_axis_likerts,
            "max_off_axis_category_likert": max_off_axis_category_likert,
            "off_axis_problem_frac": round(
                _normalize_likert(float(confound_j["off_axis_problem_likert"]), 1.0, 7.0), 4),
            "pos_refusal_phrase_hits": pos_refusal_phrase_hits,
            "neg_refusal_phrase_hits": neg_refusal_phrase_hits,
            "pos_persona_echo_hits": pos_persona_echo_hits,
            "neg_persona_echo_hits": neg_persona_echo_hits,
            "judge_persona_echo": judge_persona_echo,
            "persona_echo": pos_echo or neg_echo,
            "judge_refusal_or_ai_break": judge_refusal_or_ai_break,
            "refusal_or_ai_break": pos_refusal or neg_refusal,
            "strict_pass": strict_pass,
        })
    except Exception as e:
        base["error"] = f"{type(e).__name__}: {e}"
    return base


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else float("nan")


def _std(vals: list[float]) -> float:
    return pstdev(vals) if len(vals) > 1 else 0.0


def summarize(results: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in results:
        if "error" not in r:
            grouped[(r["axis"]["id"], r["template"])].append(r)
    out = []
    for (axis_id, template), rows in grouped.items():
        n = len(rows)
        pass_rate = sum(bool(r["strict_pass"]) for r in rows) / n
        off = [float(r["confound_judgment"]["off_axis_problem_likert"]) for r in rows]
        off_cat_max = [float(r.get("max_off_axis_category_likert", 7)) for r in rows]
        style_max = [float(r["max_style_abs_delta"]) for r in rows]
        word_abs = [abs(float(r["word_delta_frac"])) for r in rows]
        axis_delta = [float(r["axis_delta"]) for r in rows]
        axis_delta_judge_std = [float(r["axis_delta_judge_std"]) for r in rows]
        echo = sum(bool(r["persona_echo"]) for r in rows) / n
        refusal = sum(bool(r["refusal_or_ai_break"]) for r in rows) / n
        out.append({
            "axis": axis_id,
            "template": template,
            "n": n,
            "strict_pass_rate": round(pass_rate, 3),
            "mean_axis_delta": round(_mean(axis_delta), 3),
            "mean_axis_delta_judge_std": round(_mean(axis_delta_judge_std), 3),
            "mean_off_axis_problem": round(_mean(off), 3),
            "mean_max_off_axis_category_likert": round(_mean(off_cat_max), 3),
            "mean_max_style_abs_delta": round(_mean(style_max), 3),
            "mean_abs_word_delta_frac": round(_mean(word_abs), 3),
            "persona_echo_rate": round(echo, 3),
            "refusal_or_ai_break_rate": round(refusal, 3),
            "recommended": (
                n >= 3
                and pass_rate >= 0.8
                and _mean(axis_delta) >= 3
                and _mean(off) <= 2
                and _mean(style_max) <= 2
                and echo == 0
                and refusal == 0
            ),
        })
    out.sort(key=lambda r: (
        r["recommended"],
        r["strict_pass_rate"],
        r["mean_axis_delta"],
        -r["mean_off_axis_problem"],
        -r["mean_max_style_abs_delta"],
    ), reverse=True)
    return out


def axis_score_distribution(results: list[dict]) -> list[dict]:
    counts: dict[tuple[str, str, float], int] = defaultdict(int)
    for r in results:
        if "error" in r:
            continue
        for judgment in r["axis_judgments"]:
            judge_model = judgment["judge_model"]
            for key in (
                "positive_axis_forward_judgment",
                "positive_axis_reverse_judgment",
                "negative_axis_forward_judgment",
                "negative_axis_reverse_judgment",
            ):
                score = _bounded_score(judgment[key], "A_more_target_than_B", 1.0, 5.0, step=0.1)
                counts[(judge_model, key.removesuffix("_judgment"), score)] += 1
    rows = [
        {"judge_model": model, "call": call, "score": score, "n": n}
        for (model, call, score), n in counts.items()
    ]
    rows.sort(key=lambda r: (r["judge_model"], r["call"], r["score"]))
    return rows


def _print_text_block(title: str, text: str) -> None:
    print(f"\n--- {title} ---")
    print(text)


def print_judge_audit_samples(results: list[dict]) -> None:
    if not results:
        return
    sample_indices = [0] if len(results) == 1 else [0, len(results) - 1]
    print("\n=== judge audit samples: first and last planned eval ===")
    for sample_name, idx in zip(("FIRST", "LAST"), sample_indices):
        rec = results[idx]
        print(f"\n### {sample_name} idx={idx} eval_id={rec.get('eval_id')} error={rec.get('error')}")
        _print_text_block("prompt", str(rec.get("prompt", "")))
        _print_text_block("pos_generation_prompt", str(rec.get("pos_generation_prompt", "")))
        _print_text_block("neg_generation_prompt", str(rec.get("neg_generation_prompt", "")))
        _print_text_block("cho_pos_response", str(rec.get("pos_response", "")))
        _print_text_block("rej_neg_response", str(rec.get("neg_response", "")))
        _print_text_block(
            "deterministic_audit_hits",
            json.dumps({
                "pos_refusal": rec.get("pos_refusal_phrase_hits", []),
                "neg_refusal": rec.get("neg_refusal_phrase_hits", []),
                "pos_persona_echo": rec.get("pos_persona_echo_hits", []),
                "neg_persona_echo": rec.get("neg_persona_echo_hits", []),
                "persona_echo": rec.get("persona_echo"),
                "refusal_or_ai_break": rec.get("refusal_or_ai_break"),
                "response_token_jaccard": rec.get("response_token_jaccard"),
                "pos_repeated_token_frac": rec.get("pos_repeated_token_frac"),
                "neg_repeated_token_frac": rec.get("neg_repeated_token_frac"),
                "pos_persona_overlap_tokens": rec.get("pos_persona_overlap_tokens", []),
                "neg_persona_overlap_tokens": rec.get("neg_persona_overlap_tokens", []),
            }, indent=2),
        )
        _print_text_block(
            "full_judge_output",
            json.dumps(rec.get("raw_judge_outputs", {}), indent=2, ensure_ascii=False),
        )


async def amain(args) -> None:
    load_dotenv(ROOT / ".env")
    axes = _select_axes(args.axes)
    templates = _select_templates(args.templates)
    rows = _select_rows(args.family, args.n, args.seed)
    axis_judge_models = tuple(
        model.strip() for model in args.axis_judge_models.split(",") if model.strip()
    )
    if not axis_judge_models:
        raise ValueError("--axis-judge-models selected zero models")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        results = []
        for row_i, row in enumerate(rows, start=1):
            prompt_text = _scenario_text(row)
            for axis in axes:
                for template in templates:
                    pos_label, neg_label, order = _labels_for(
                        args.seed, axis.id, template, str(row_i), prompt_text)
                    results.append({
                        "eval_id": _eval_id(
                            seed=args.seed,
                            row=row,
                            row_i=row_i,
                            scenario=prompt_text,
                            axis_id=axis.id,
                            template=template,
                            generator_model=args.generator_model,
                            judge_model=",".join(axis_judge_models) + "|" + args.judge_model,
                            gen_temperature=args.gen_temperature,
                        ),
                        "row": row_i,
                        "scenario_id": _scenario_id(row, row_i),
                        "source": row.get("source"),
                        "config": row.get("config"),
                        "tags": row.get("tags", []),
                        "selected_family": row.get("selected_family"),
                        "axis": asdict(axis),
                        "template": template,
                        "prompt": prompt_text,
                        "blind_order": order,
                        "pos_label": pos_label,
                        "neg_label": neg_label,
                        "dry_run": True,
                    })
        artifact = {
            "dry_run": True,
            "generator_model": args.generator_model,
            "judge_model": args.judge_model,
            "axis_judge_models": list(axis_judge_models),
            "style_judge_model": args.judge_model,
            "gen_temperature": args.gen_temperature,
            "judge_temperature": 0.0,
            "seed": args.seed,
            "max_word_delta_frac": args.max_word_delta_frac,
            "n_prompts": len(rows),
            "axes": [asdict(a) for a in axes],
            "templates": list(templates),
            "results": results,
            "summary": [],
        }
        out.write_text(json.dumps(artifact, indent=2))
        print(f"dry-run wrote {out}")
        print(f"axes: {', '.join(a.id for a in axes)}")
        print(f"templates: {len(templates)}; planned pairs: {len(results)}")
        return

    if not os.environ.get("OPENROUTER_API_KEY"):
        logger.error("OPENROUTER_API_KEY not set")
        sys.exit(1)

    router = OpenRouter(Path(args.cache_dir), args.concurrency)
    tasks = []
    for row_i, row in enumerate(rows, start=1):
        for axis in axes:
            for template in templates:
                tasks.append(_evaluate_one(
                    router,
                    generator_model=args.generator_model,
                    style_judge_model=args.judge_model,
                    axis_judge_models=axis_judge_models,
                    axis=axis,
                    template=template,
                    row=row,
                    row_i=row_i,
                    seed=args.seed,
                    gen_temperature=args.gen_temperature,
                    max_word_delta_frac=args.max_word_delta_frac,
                ))
    logger.info(
        f"{len(rows)} prompts × {len(axes)} axes × {len(templates)} templates "
        f"= {len(tasks)} pairs; generator={args.generator_model}; "
        f"axis_judges={','.join(axis_judge_models)}; style_judge={args.judge_model}; "
        f"gen_temperature={args.gen_temperature}; judge_temperature=0.0"
    )
    tasks = [asyncio.create_task(task) for task in tasks]
    results = []
    for task in atqdm(tasks, total=len(tasks), desc="persona-axes"):
        rec = await task
        results.append(rec)
        artifact = {
            "dry_run": False,
            "generator_model": args.generator_model,
            "judge_model": args.judge_model,
            "axis_judge_models": list(axis_judge_models),
            "style_judge_model": args.judge_model,
            "gen_temperature": args.gen_temperature,
            "judge_temperature": 0.0,
            "family": args.family,
            "seed": args.seed,
            "max_word_delta_frac": args.max_word_delta_frac,
            "n_prompts": len(rows),
            "axes": [asdict(a) for a in axes],
            "templates": list(templates),
            "n_results": len(results),
            "n_success": sum("error" not in r for r in results),
            "n_errors": sum("error" in r for r in results),
            "summary": summarize(results),
            "axis_score_distribution": axis_score_distribution(results),
            "results": results,
        }
        out.write_text(json.dumps(artifact, indent=2))

    summary = summarize(results)
    artifact = {
        "dry_run": False,
        "generator_model": args.generator_model,
        "judge_model": args.judge_model,
        "axis_judge_models": list(axis_judge_models),
        "style_judge_model": args.judge_model,
        "gen_temperature": args.gen_temperature,
        "judge_temperature": 0.0,
        "family": args.family,
        "seed": args.seed,
        "max_word_delta_frac": args.max_word_delta_frac,
        "n_prompts": len(rows),
        "axes": [asdict(a) for a in axes],
        "templates": list(templates),
        "n_results": len(results),
        "n_success": sum("error" not in r for r in results),
        "n_errors": sum("error" in r for r in results),
        "summary": summary,
        "axis_score_distribution": axis_score_distribution(results),
        "results": results,
    }
    out.write_text(json.dumps(artifact, indent=2))
    print(f"wrote {out}")
    print(tabulate(summary, headers="keys", tablefmt="pipe", floatfmt=".3f"))
    print("\naxis judge raw score distribution:")
    print(tabulate(
        axis_score_distribution(results),
        headers="keys",
        tablefmt="pipe",
        floatfmt=".1f",
    ))
    print_judge_audit_samples(results)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--generator-model", default="qwen/qwen3.5-27b")
    ap.add_argument("--judge-model", default="google/gemini-3.1-flash-lite-preview")
    ap.add_argument(
        "--axis-judge-models",
        default="google/gemini-3.1-flash-lite-preview,deepseek/deepseek-v4-flash",
    )
    ap.add_argument("--gen-temperature", type=float, default=0.0,
                    help="generation temperature; default 0 to avoid sampling-diff confounds")
    ap.add_argument("--family", default="character",
                    help="comma-separated scenario families; default avoids sycophancy")
    ap.add_argument("--n", type=int, default=6, help="number of scenario prompts")
    ap.add_argument("--axes", default=str(ROOT / "data/persona_pairs_pilot_two.jsonl"),
                    help="persona-pair JSONL path")
    ap.add_argument("--templates", default=str(ROOT / "data/template_catalog.yaml"),
                    help="'skill', 'controls', catalog path, text file path, or templates separated by ||")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--max-word-delta-frac", type=float, default=0.0,
                    help="optional hard length gate; 0 means report-only")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--cache-dir", default="out/cache/persona_axes_openrouter")
    ap.add_argument("--out", default="out/persona_axes_openrouter.json")
    ap.add_argument("--dry-run", action="store_true",
                    help="write planned randomized A/B jobs without network calls")
    args = ap.parse_args()
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()

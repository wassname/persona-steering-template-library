import argparse
import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from inspect_ai import Task, eval_async
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import CachePolicy, ModelOutput, ModelUsage, get_model

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import bounded_thinking_judge as bounded
import validate_persona_axes as validator


AXIS = validator.Axis(
    id="careful_hasty",
    pos_descriptor="careful",
    neg_descriptor="hasty",
    positive_behavior="checks the evidence before deciding",
    negative_behavior="commits without checking the evidence",
)
ROW = {
    "id": "test_scenario",
    "prompt": "A report contains one surprising number. What do you do?",
    "source": "test",
    "selected_family": "test",
    "self_contained": True,
}


def _args(*, method: str = "json") -> argparse.Namespace:
    return argparse.Namespace(
        generator_model="qwen/qwen3-14b",
        judge_model="mock/style",
        axis_judge_models=("mock/axis",),
        gen_temperature=0.0,
        seed=13,
        max_word_delta_frac=0.0,
        generator_provider_only=("DeepInfra",),
        axis_judge_method=method,
        axis_judge_n=1,
        axis_judge_budget=64,
        concurrency=4,
        exclude_confound_dims=set(),
        axis_delta_threshold=3.0,
        min_side_threshold=0.5,
        off_axis_threshold=2.0,
        family="test",
        n=1,
        n_per_source=None,
        axes="unused",
        templates="unused",
        out="unused",
        log_dir="unused",
        dry_run=False,
    )


def _output(model: str, content: str) -> ModelOutput:
    output = ModelOutput.from_content(model=model, content=content)
    output.usage = ModelUsage(input_tokens=10, output_tokens=5, total_tokens=15)
    return output


def _style_json() -> str:
    obj = {"style_reason": "matched"}
    for dim in validator.STYLE_DIMS:
        obj[f"{dim}_A"] = 1.0
        obj[f"{dim}_B"] = 1.0
    obj.update({
        "persona_echo_A": False,
        "persona_echo_B": False,
        "refusal_or_ai_break_A": False,
        "refusal_or_ai_break_B": False,
    })
    return json.dumps(obj)


def _confound_json() -> str:
    obj = {"confound_reason": "none"}
    obj.update({f"{dim}_likert": 1.0 for dim in validator.OFF_AXIS_DIMS})
    obj.update({
        "off_axis_problem_likert": 1.0,
        "likely_spurious_axis": "none",
        "usable_for_training": True,
    })
    return json.dumps(obj)


def _samples(templates: tuple[str, ...]) -> list[Sample]:
    return [
        Sample(
            id=f"sample-{i}",
            input=ROW["prompt"],
            metadata={
                "axis": validator.asdict(AXIS),
                "template": template,
                "row": ROW,
                "row_i": 1,
            },
        )
        for i, template in enumerate(templates)
    ]


def _mock_task(
    args: argparse.Namespace,
    templates: tuple[str, ...],
    *,
    generator_output,
    style_output,
    axis_output,
    force_output=None,
) -> Task:
    generator = get_model(
        "mockllm/generator", custom_outputs=generator_output, memoize=False
    )
    style = get_model("mockllm/style", custom_outputs=style_output, memoize=False)
    axis = get_model("mockllm/axis", custom_outputs=axis_output, memoize=False)
    roles = {"generator": generator, "style_judge": style, "axis_judge_0": axis}
    if force_output is not None:
        roles["axis_judge_force_0"] = get_model(
            "mockllm/force", custom_outputs=force_output, memoize=False
        )
    return Task(
        name="persona_axis_mock",
        dataset=MemoryDataset(_samples(templates)),
        solver=validator.persona_axis_solver(validator._solver_config(args)),
        scorer=validator.persona_axis_score(),
        model=generator,
        model_roles=roles,
        fail_on_error=True,
    )


def _run_eval(task: Task, tmp_path: Path):
    os.environ["INSPECT_DISPLAY"] = "none"
    os.environ["INSPECT_CACHE_DIR"] = str(tmp_path / "cache")
    return asyncio.run(eval_async(
        task,
        log_dir=str(tmp_path / "logs"),
        max_samples=4,
        fail_on_error=True,
        debug_errors=True,
        log_model_api=True,
    ))[0]


def test_inspect_log_baseline_reuse_scores_and_export(tmp_path: Path) -> None:
    args = _args()
    baseline_prompt = validator._generation_prompt(
        "", ROW["prompt"], self_contained=True, qwen_no_think=True
    )
    generator_calls = []

    def generator_output(input, tools, tool_choice, config):
        prompt = input[-1].text
        generator_calls.append((prompt, config.model_dump(mode="json")))
        if prompt == baseline_prompt:
            return _output("generator", "baseline answer")
        if "careful" in prompt:
            return _output("generator", "careful answer")
        return _output("generator", "hasty answer")

    def style_output(input, tools, tool_choice, config):
        prompt = input[-1].text
        content = _style_json() if "fresh-eyes style rater" in prompt else _confound_json()
        return _output("style", content)

    def axis_output(input, tools, tool_choice, config):
        prompt = input[-1].text
        response_a = prompt.split("Response A:\n", 1)[1].split("\n\nResponse B:", 1)[0]
        rating = 2.0 if response_a == "baseline answer" else 4.0
        return _output("axis", json.dumps({
            "target_reason": "target response moves away from baseline",
            "A_more_target_than_B": rating,
        }))

    templates = (
        "Act as a {persona} person.",
        "Use the priorities of a {persona} person.",
    )
    log = _run_eval(
        _mock_task(
            args,
            templates,
            generator_output=generator_output,
            style_output=style_output,
            axis_output=axis_output,
        ),
        tmp_path,
    )

    assert log.status == "success"
    assert Path(log.location).suffix == ".eval"
    assert len([prompt for prompt, _ in generator_calls if prompt == baseline_prompt]) == 1
    baseline_configs = [config for prompt, config in generator_calls if prompt == baseline_prompt]
    assert baseline_configs[0]["temperature"] == 0.0
    assert baseline_configs[0]["seed"] == 13
    assert baseline_configs[0]["max_connections"] == 4
    assert log.stats.model_usage

    results = [sample.store[validator.RESULT_STORE_KEY] for sample in log.samples]
    assert {result["base_generation_prompt"] for result in results} == {baseline_prompt}
    for sample, result in zip(log.samples, results):
        assert result["delta_pos_vs_base"] == 1.0
        assert result["delta_base_vs_neg"] == 1.0
        assert result["min_side_delta"] == 1.0
        assert result["axis_delta"] == 4.0
        assert result["overall_score"] == 3.0
        assert result["strict_pass"] is True
        assert result["self_contained"] is True
        assert sample.scores["persona_axis_score"].metadata["strict_pass"] is True
        assert any(type(event).__name__ == "ModelEvent" for event in sample.events)

    artifact = validator._artifact(
        args,
        axes=[AXIS],
        templates=templates,
        rows=[ROW],
        results=results,
        dry_run=False,
        inspect_log=log.location,
    )
    artifact_path = tmp_path / "screen.json"
    artifact_path.write_text(json.dumps(artifact))
    export_path = tmp_path / "selection.jsonl"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "export_selections.py"),
            str(artifact_path),
            "--strict-only",
            "--top-n",
            "1",
            "--out",
            str(export_path),
        ],
        check=True,
    )
    exported = json.loads(export_path.read_text().splitlines()[0])
    assert exported["strict_pass"] is True
    assert exported["overall_score"] == 3.0
    assert exported["self_contained"] is True


def test_skill_template_selects_verbatim_personas() -> None:
    assert validator._select_templates("skill") == (validator.VERBATIM_TEMPLATE,)


def test_dead_negative_side_fails_despite_total_separation(tmp_path: Path) -> None:
    args = _args()

    def generator_output(input, tools, tool_choice, config):
        prompt = input[-1].text
        if "careful" in prompt:
            content = "careful answer"
        elif "hasty" in prompt:
            content = "hasty answer"
        else:
            content = "baseline answer"
        return _output("generator", content)

    def style_output(input, tools, tool_choice, config):
        content = _style_json() if "fresh-eyes style rater" in input[-1].text else _confound_json()
        return _output("style", content)

    def axis_output(input, tools, tool_choice, config):
        prompt = input[-1].text
        if AXIS.negative_behavior in prompt:
            rating = 3.0
        else:
            response_a = prompt.split("Response A:\n", 1)[1].split("\n\nResponse B:", 1)[0]
            rating = 1.0 if response_a == "baseline answer" else 5.0
        return _output("axis", json.dumps({
            "target_reason": "fixture",
            "A_more_target_than_B": rating,
        }))

    log = _run_eval(
        _mock_task(
            args,
            ("Act as a {persona} person.",),
            generator_output=generator_output,
            style_output=style_output,
            axis_output=axis_output,
        ),
        tmp_path,
    )
    result = log.samples[0].store[validator.RESULT_STORE_KEY]
    assert result["axis_delta"] == 4.0
    assert result["min_side_delta"] == 0.0
    assert result["strict_pass"] is False


def test_malformed_json_fails_inspect_eval(tmp_path: Path) -> None:
    args = _args()

    def generator_output(input, tools, tool_choice, config):
        return _output("generator", "answer")

    def malformed(input, tools, tool_choice, config):
        return _output("judge", "not json")

    task = _mock_task(
        args,
        ("Act as a {persona} person.",),
        generator_output=generator_output,
        style_output=malformed,
        axis_output=malformed,
    )
    with pytest.raises(Exception, match="Expecting value"):
        _run_eval(task, tmp_path)


def test_bounded_nonverdict_is_recorded_as_failure(tmp_path: Path) -> None:
    args = _args(method="bounded_thinking")

    def generator_output(input, tools, tool_choice, config):
        prompt = input[-1].text
        if "careful" in prompt:
            content = "careful answer"
        elif "hasty" in prompt:
            content = "hasty answer"
        else:
            content = "baseline answer"
        return _output("generator", content)

    def style_output(input, tools, tool_choice, config):
        content = _style_json() if "fresh-eyes style rater" in input[-1].text else _confound_json()
        return _output("style", content)

    def no_verdict(input, tools, tool_choice, config):
        return _output("axis", "I remain uncertain.")

    log = _run_eval(
        _mock_task(
            args,
            ("Act as a {persona} person.",),
            generator_output=generator_output,
            style_output=style_output,
            axis_output=no_verdict,
            force_output=no_verdict,
        ),
        tmp_path,
    )
    result = log.samples[0].store[validator.RESULT_STORE_KEY]
    assert result["judge_did_not_commit"] is True
    assert result["strict_pass"] is False
    assert "error" not in result


def test_bounded_thinking_configs_and_openrouter_routing(tmp_path: Path) -> None:
    phase1_configs = []
    phase2_configs = []

    def phase1(input, tools, tool_choice, config):
        phase1_configs.append(config)
        return _output("thinking", "still thinking")

    def phase2(input, tools, tool_choice, config):
        phase2_configs.append(config)
        return _output("force", "SCORE: 4")

    thinking_model = get_model("mockllm/thinking", custom_outputs=phase1, memoize=False)
    force_model = get_model("mockllm/force", custom_outputs=phase2, memoize=False)
    score, found, forced = asyncio.run(bounded.judge_once(
        model=thinking_model,
        force_model=force_model,
        prompt="judge this",
        budget=123,
        seed=17,
        max_connections=6,
        cache=False,
    ))
    assert (score, found, forced) == (4, True, True)
    assert phase1_configs[0].model_dump()["max_tokens"] == 123
    assert phase1_configs[0].temperature == 1.0
    assert phase1_configs[0].top_p == 0.95
    assert phase1_configs[0].presence_penalty == 1.5
    assert phase1_configs[0].extra_body == {"top_k": 20}
    assert phase1_configs[0].seed == 17
    assert phase1_configs[0].max_connections == 6
    assert phase2_configs[0].max_tokens == 256
    assert phase2_configs[0].temperature == 0.7
    assert phase2_configs[0].reasoning_effort == "none"
    assert phase2_configs[0].extra_body == {"top_k": 20, "include_reasoning": False}

    os.environ["INSPECT_CACHE_DIR"] = str(tmp_path / "bounded-cache")
    calls = []

    def sampled(input, tools, tool_choice, config):
        calls.append(len(calls) + 1)
        return _output("sampled", f"SCORE: {calls[-1]}")

    sampled_model = get_model("mockllm/sampled", custom_outputs=sampled, memoize=False)
    sample_cache = CachePolicy(expiry=None, per_epoch=False)

    async def run_twice():
        first = await bounded.judge(
            model=sampled_model,
            force_model=force_model,
            prompt="sample twice",
            n=2,
            seed=1,
            cache=sample_cache,
        )
        second = await bounded.judge(
            model=sampled_model,
            force_model=force_model,
            prompt="sample twice",
            n=2,
            seed=1,
            cache=sample_cache,
        )
        return first, second

    first, second = asyncio.run(run_twice())
    assert calls == [1, 2]
    assert first["samples"] == second["samples"]
    assert {sample["score"] for sample in second["samples"]} == {1, 2}

    os.environ["OPENROUTER_API_KEY"] = "test-key"
    routed = validator._openrouter_model(
        "qwen/qwen3-14b",
        max_connections=7,
        provider_only=("DeepInfra",),
        reasoning_enabled=False,
    )
    assert routed.config.max_connections == 7
    assert routed.api.provider == {
        "only": ["DeepInfra"],
        "allow_fallbacks": False,
    }
    assert routed.api.reasoning_enabled is False

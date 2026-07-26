from __future__ import annotations

import json
import sys
from pathlib import Path


SCENARIO_SOURCE_DIR = Path(__file__).resolve().parents[1] / "scripts" / "corpus" / "scenario_sources"
sys.path.insert(0, str(SCENARIO_SOURCE_DIR))

import loaders  # noqa: E402


def test_bullshit_benchmark_flattens_questions_without_answer_leak(monkeypatch) -> None:
    upstream = {
        "techniques": [
            {
                "technique": "fabricated_framework",
                "questions": [
                    {
                        "id": "fake_01",
                        "question": "How should I calibrate the fictional mechanism?",
                        "nonsensical_element": "The mechanism is fictional.",
                        "domain": "testing",
                    },
                    {
                        "id": "fake_02",
                        "question": "What setting should the invented method use?",
                        "nonsensical_element": "The method is invented.",
                        "domain": "testing",
                    },
                ],
            }
        ]
    }
    monkeypatch.setattr(loaders, "_url_text", lambda _: json.dumps(upstream))

    rows = loaders.load_bullshit_benchmark(limit=1)

    assert rows == [{
        "text": "How should I calibrate the fictional mechanism?",
        "axes": ["credulity", "skepticism", "nonsense_detection"],
        "source": "bullshit_benchmark",
        "source_id": "bullshit_benchmark_fabricated_framework_fake_01",
    }]
    assert upstream["techniques"][0]["questions"][0]["nonsensical_element"] not in rows[0]["text"]

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CATALOG_PATH = DATA / "template_catalog.yaml"
CATALOG_JSONL_PATH = DATA / "template_catalog.jsonl"
TEMPLATES_TXT_PATH = DATA / "templates_v2_candidates.txt"
TEMPLATE_SOURCES_PATH = DATA / "template_sources.jsonl"

JINJA_VAR_RE = re.compile(r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def read_yaml(path: Path) -> list[dict[str, Any]]:
    rows = yaml.safe_load(path.read_text())
    if not isinstance(rows, list):
        raise ValueError(f"{path} must contain a YAML list")
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def extract_jinja_placeholders(template_jinja: str) -> list[str]:
    return [match.group(1) for match in JINJA_VAR_RE.finditer(template_jinja)]


def jinja_to_runtime(template_jinja: str) -> str:
    return JINJA_VAR_RE.sub(lambda match: "{" + match.group(1) + "}", template_jinja)


def load_template_catalog(path: Path = CATALOG_PATH) -> list[dict[str, Any]]:
    rows = read_yaml(path) if path.suffix in {".yaml", ".yml"} else read_jsonl(path)
    for row in rows:
        row.setdefault("status", "active")
        row.setdefault("kind", "persona_template")
        row.setdefault("other_sources", [])
        row.setdefault("example_bindings", {})
        placeholders = extract_jinja_placeholders(row["template_jinja"])
        row["placeholders"] = placeholders
        row["template_runtime"] = jinja_to_runtime(row["template_jinja"])
    return rows


def active_template_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row["status"] == "active"]


def validate_template_catalog(rows: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    seen_jinja: set[str] = set()
    seen_runtime_active: set[str] = set()
    for i, row in enumerate(rows, start=1):
        label = f"row {i} {row.get('template_jinja', '')!r}"
        template_jinja = row.get("template_jinja")
        if not template_jinja:
            errors.append(f"{label}: missing template_jinja")
            continue
        if template_jinja in seen_jinja:
            errors.append(f"{label}: duplicate template_jinja")
        seen_jinja.add(template_jinja)
        if row.get("status") not in {"active", "catalog_only", "excluded"}:
            errors.append(f"{label}: status must be active, catalog_only, or excluded")
        if not row.get("primary_source_id"):
            errors.append(f"{label}: missing primary_source_id")
        if not row.get("primary_source_type"):
            errors.append(f"{label}: missing primary_source_type")
        if not row.get("primary_source_url"):
            errors.append(f"{label}: missing primary_source_url")
        placeholders = row.get("placeholders", [])
        example_bindings = row.get("example_bindings", {})
        if example_bindings and set(example_bindings) != set(placeholders):
            errors.append(
                f"{label}: example_bindings keys {sorted(example_bindings)} do not match placeholders {placeholders}")
        if row.get("status") == "active":
            if row.get("kind") != "persona_template":
                errors.append(f"{label}: active row must have kind=persona_template")
            if placeholders != ["persona"]:
                errors.append(
                    f"{label}: active row must use exactly one {{ persona }} slot, got {placeholders}")
            runtime = row["template_runtime"]
            if runtime in seen_runtime_active:
                errors.append(f"{label}: duplicate active runtime template")
            seen_runtime_active.add(runtime)
        elif row.get("status") == "catalog_only":
            if not row.get("catalog_reason"):
                errors.append(f"{label}: catalog_only row missing catalog_reason")
            if len(placeholders) > 1 and not example_bindings:
                errors.append(f"{label}: catalog_only multi-slot row missing example_bindings")
        elif not row.get("exclusion_reason"):
            errors.append(f"{label}: excluded row missing exclusion_reason")
    return errors


def generated_template_source_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in active_template_rows(rows):
        out.append({
            "template": row["template_runtime"],
            "source_id": row["primary_source_id"],
            "source_type": row["primary_source_type"],
            "source_url": row["primary_source_url"],
            "note": row.get("note", ""),
            "other_sources": row.get("other_sources", []),
        })
    return out


def write_generated_runtime_files(rows: list[dict[str, Any]]) -> None:
    active_rows = active_template_rows(rows)
    generated_rows = [
        {
            key: value
            for key, value in row.items()
            if key not in {"placeholders", "template_runtime"}
        }
        for row in rows
    ]
    write_jsonl(CATALOG_JSONL_PATH, generated_rows)
    TEMPLATES_TXT_PATH.write_text(
        "\n".join(row["template_runtime"].replace("\n", "\\n") for row in active_rows) + "\n")
    write_jsonl(TEMPLATE_SOURCES_PATH, generated_template_source_rows(rows))

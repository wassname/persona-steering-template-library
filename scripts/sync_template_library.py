from __future__ import annotations

import argparse
import sys

from template_catalog import (
    CATALOG_PATH,
    load_template_catalog,
    validate_template_catalog,
    write_generated_runtime_files,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    rows = load_template_catalog(CATALOG_PATH)
    errors = validate_template_catalog(rows)
    if errors:
        print("template catalog check failed:")
        for err in errors:
            print(f"- {err}")
        raise SystemExit(1)

    active = sum(row["status"] == "active" for row in rows)
    catalog_only = sum(row["status"] == "catalog_only" for row in rows)
    excluded = sum(row["status"] == "excluded" for row in rows)
    print(f"catalog={CATALOG_PATH}")
    print(f"active={active} catalog_only={catalog_only} excluded={excluded}")

    if args.check:
        return

    write_generated_runtime_files(rows)
    print("wrote data/template_catalog.jsonl")
    print("wrote data/templates_v2_candidates.txt")
    print("wrote data/template_sources.jsonl")


if __name__ == "__main__":
    main()

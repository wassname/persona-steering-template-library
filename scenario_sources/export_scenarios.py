"""Write scenario source loaders to data/scenarios JSONL files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from loaders import LOADERS

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "scenarios"


def _lean_row(row: dict[str, Any]) -> dict[str, Any]:
    source_id = str(row["source_id"])
    return {
        "id": source_id,
        "text": row["text"],
        "axes": row["axes"],
        "source": row["source"],
        "source_id": source_id,
        "self_contained": True,
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def _source_names(raw: str) -> list[str]:
    if raw == "all":
        return sorted(LOADERS)
    names = [name.strip() for name in raw.split(",") if name.strip()]
    unknown = sorted(set(names) - set(LOADERS))
    if unknown:
        raise ValueError(f"unknown sources {unknown}; choices={sorted(LOADERS)}")
    return names


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default="all", help="comma-separated loader names, or all")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    for source in _source_names(args.sources):
        rows = [_lean_row(row) for row in LOADERS[source](limit=args.limit)]
        if not rows:
            raise RuntimeError(f"{source} emitted zero rows")
        out = args.out_dir / f"scenarios_{source}.jsonl"
        _write_jsonl(out, rows)
        print(f"{source}\t{len(rows)}\t{out}")


if __name__ == "__main__":
    main()

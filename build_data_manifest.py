from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None


ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "Data"
OUT_PATH = DATA_ROOT / "manifests" / "data_manifest.json"


@dataclass
class DataItem:
    name: str
    relative_path: str
    category: str
    purpose: str
    size_bytes: int
    target_column: str | None
    module: str
    public: bool
    columns: list[str]


def preview_columns(path: Path) -> list[str]:
    if pd is None or path.suffix.lower() not in {".xlsx", ".csv"}:
        return []
    try:
        if path.suffix.lower() == ".csv":
            return [str(c) for c in pd.read_csv(path, nrows=0).columns]
        return [str(c) for c in pd.read_excel(path, nrows=0).columns]
    except Exception:
        return []


def classify(path: Path) -> tuple[str, str, str | None, str, bool]:
    text = str(path).lower()
    if "3dfrac" in text:
        return "3Dfrac", "DAS, construction pressure and well trajectory for fracture inversion", None, "DT-Crack", False
    if "multimodal" in text:
        return "multimodal", "book text and images for expert knowledge graph", None, "FSL-Expert", True
    if "raw_frac" in text:
        target = "WORKING_TYPE" if path.suffix.lower() in {".xlsx", ".csv"} else None
        return "raw_frac", "fracturing operation modeling and label recognition", target, "FSL-Expert", False
    return "other", "project data asset", None, "shared", False


def build_manifest() -> dict:
    items: list[DataItem] = []
    for path in DATA_ROOT.rglob("*"):
        if not path.is_file():
            continue
        category, purpose, target, module, public = classify(path)
        items.append(
            DataItem(
                name=path.name,
                relative_path=str(path.relative_to(ROOT)).replace("\\", "/"),
                category=category,
                purpose=purpose,
                size_bytes=path.stat().st_size,
                target_column=target,
                module=module,
                public=public,
                columns=preview_columns(path),
            )
        )
    totals: dict[str, int] = {}
    for item in items:
        totals[item.category] = totals.get(item.category, 0) + 1
    return {
        "project_root": str(ROOT),
        "data_root": str(DATA_ROOT),
        "item_count": len(items),
        "category_counts": totals,
        "items": [asdict(item) for item in sorted(items, key=lambda x: (x.category, x.name.lower()))],
    }


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest()
    OUT_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(OUT_PATH), "item_count": manifest["item_count"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

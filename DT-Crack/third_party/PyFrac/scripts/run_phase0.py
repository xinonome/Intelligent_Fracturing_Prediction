from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def _run(script: Path) -> dict:
    completed = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, check=True)
    lines = [line for line in completed.stdout.splitlines() if line.strip().startswith("{")]
    return json.loads(lines[-1])


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    output = root / "outputs" / "baseline"; output.mkdir(parents=True, exist_ok=True)
    summary = {
        "phase": "0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "smoke": _run(root / "scripts" / "smoke_pyfrac.py"),
        "radial": _run(root / "scripts" / "reproduce_radial.py"),
        "height_contained": _run(root / "scripts" / "reproduce_height_contained.py"),
    }
    (output / "phase0_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "PASS", "summary": str(output / "phase0_summary.json")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

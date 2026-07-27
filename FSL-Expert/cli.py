from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
raise SystemExit(subprocess.run([sys.executable, str(ROOT / "run_project.py"), "fsl", *sys.argv[1:]], cwd=ROOT).returncode)

"""Allow the reviewed scripts to run directly as well as through run_project.py."""

from pathlib import Path
import sys

FSL_ROOT = Path(__file__).resolve().parent.parent
if str(FSL_ROOT) not in sys.path:
    sys.path.insert(0, str(FSL_ROOT))

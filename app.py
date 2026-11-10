import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "karta hai"))
runpy.run_path(str(ROOT / "careerpath_ai" / "app.py"), run_name="__main__")

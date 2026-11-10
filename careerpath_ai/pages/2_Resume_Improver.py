from pathlib import Path
import runpy


runpy.run_path(str(Path(__file__).resolve().parents[2] / "pages" / "2_Resume_Improver.py"))

from pathlib import Path
import runpy


runpy.run_path(str(Path(__file__).resolve().parents[2] / "pages" / "4_Career_Map.py"))

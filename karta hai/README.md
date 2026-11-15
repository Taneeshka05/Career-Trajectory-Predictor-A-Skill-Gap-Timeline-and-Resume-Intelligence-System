# CareerPath AI: Core Architecture & Engine

This folder contains the core processing modules, data tables, artifacts, and database operations for the **Career Trajectory Predictor & Skill Gap Timeline Analyzer**.

For complete academic documentation, system architecture, viva defense preparation, and execution guides, please refer to the primary project [README.md](../README.md).

## Core Modules Overview
- `core/engine.py`: Central coordination engine for multi-dimensional compatibility, career screening, and trajectory evaluation.
- `core/taxonomy.py`: Structured skill taxonomy across 6 technical categories, 50+ aliases, soft skills, and cross-domain transferability graph.
- `core/parser.py`: Multi-format resume parser supporting `.txt`, `.pdf` (via `pypdf`), and native pure-Python `.docx` extraction (via `zipfile` + `xml.etree.ElementTree`); validates uploaded resume content and derives experience from resume text and employment dates.
- `core/ml_pipeline.py`: Real-time machine learning pipeline: DataValidator, EDA, SimpleImputer, StandardScaler, OneHotEncoder, regression/classification training, unseen test metrics, and live inference.
- `core/nlp_feedback.py`: Passive verb replacement, vocabulary analysis (TTR), quantifiable metric detection, and STAR bullet point generation.
- `core/prerequisites.py`: DAG prerequisite dependencies and topological sorting for ordered learning roadmaps.
- `core/trajectory.py`: What-If simulator, trajectory path evaluation, and skill acquisition ROI calculator.
- `core/models.py`: Pre-trained model artifact management, 5-fold cross-validation, learning curves, and noise stress tests.
- `core/charts.py`: Plotly charts (Gauges, Radar, Gantt roadmaps, Trajectory projections, Confusion Matrices, EDA heatmaps).
- `core/database.py`: SQLite persistence for analysis history and milestone tracking (requires `CAREERPATH_ENABLE_HISTORY=1`).
- `core/ui.py`: Unified dark professional theme, glassmorphic cards, badges, and responsive UI components.

## Local Execution
From the project root:
```powershell
python -m pip install -r .\requirements.txt
python -m streamlit run .\app.py
```

Run test suite:
```powershell
python -m pytest tests/ -v
```

# CareerPath AI

Resume job matching with a skill gap timeline and career trajectory prediction. Built with Python, scikit-learn, Keras and Streamlit.

## What it does
- Match score between a resume and a job (TF-IDF + cosine similarity)
- Skill gap: matched and missing skills for the target job
- Time to job ready (Linear and Ridge regression) with a week-by-week roadmap that changes with study hours
- Readiness level Junior / Mid / Senior (Logistic Regression, KNN, Random Forest, SVM compared)
- Resume feedback with NLP and a neural network language score, plus a one-click resume improver
- Alternate careers with readiness level and weeks, on a PCA / K-Means skill map
- Job recommender across 420 postings
- Saved history in SQLite with a roadmap progress tracker
- Model Lab page with metrics, confusion matrices, normalization details and live logs

## Run locally
From the project root in PowerShell:

```powershell
python -m pip install -r ".\requirements.txt"
python -m streamlit run ".\app.py"
```

Run the test suite from the project root with `python -m pytest -q`.

The checked-in model artifacts avoid training on the first launch. If they are deleted, the app
trains replacement models automatically. Select Python 3.11 when running locally to match the
tested dependencies.

## Public demo deployment

1. Publish this project to GitHub. Include the root `app.py`, `requirements.txt`, `pages/`, `karta hai/core/`,
   and the two files in `karta hai/artifacts/`. The root `.gitignore` excludes local environments, logs,
   databases, and generated datasets.
2. In Streamlit Community Cloud, create an app from the repository and set its main file to `app.py`.
3. In Advanced settings, choose Python 3.11. The dependency versions are pinned for this runtime.
4. Leave `CAREERPATH_ENABLE_HISTORY` unset for a public demo. History storage is disabled by default, so
   uploaded resumes and analyses are not written to the shared SQLite database. Do not enable history on
   a public, unauthenticated deployment: profile names are not authentication.

The Home page includes a synthetic sample resume. Users can replace it with their own resume for analysis.
The analysis runs in memory for the active session; do not submit sensitive personal data to a public demo.
For local trusted development only, history can be explicitly enabled in PowerShell before starting Streamlit:

```powershell
$env:CAREERPATH_ENABLE_HISTORY = "1"
python -m streamlit run ".\app.py"
```

Training labels and the sample data are synthetic, so output is guidance—not a hiring decision or guarantee.

## Limitations
The dataset is synthetic (5,500 resumes, 420 jobs) and labels for readiness level, time to ready and language quality are created by documented rules. High scores reflect those rules, not real hiring outcomes.
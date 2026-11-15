import sys
from pathlib import Path
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "karta hai") not in sys.path:
    sys.path.insert(0, str(ROOT / "karta hai"))

from core import config as cfg
from core import charts
from core.parser import extract_resume_metadata, is_valid_resume, parse_document
from core.ui import setup, chips, show_invalid_resume_dialog

engine = setup("Recruiter Mode", "🧑‍💼")
st.title("🧑‍💼 Recruiter Screening & Candidate Ranking Mode")
st.caption("Batch screen and rank multiple applicant resumes against a target job specification with explainable scoring and gap audits.")

c1, c2 = st.columns(2)
with c1:
    domain = st.selectbox("Target job domain", cfg.DOMAIN_NAMES, key="domain_recruiter")
with c2:
    ids = [i for i, d in enumerate(engine.job_domains) if d == domain]
    job_id = st.selectbox("Target position", ids, key="job_recruiter", format_func=lambda j: f"{engine.job_titles[j]} (#{j})")

st.markdown("**Required Skills for this Role:** " + chips(sorted(engine.job_skills[job_id]), "info"), unsafe_allow_html=True)

files = st.file_uploader(
    "Upload Candidate Resumes (.txt, .pdf, or .docx)",
    type=["txt", "pdf", "docx"],
    accept_multiple_files=True,
    key="recruiter_files"
)
pasted = st.text_area(
    "Or paste multiple candidate resumes (separate candidates with a line containing only '---')",
    height=160,
    key="recruiter_text"
)

if st.button("Screen & Rank Applicants", type="primary"):
    candidates = []
    invalid_candidates = []
    if files:
        for f in files:
            txt = parse_document(f, f.name)
            if is_valid_resume(txt):
                candidates.append((f.name, txt))
            else:
                invalid_candidates.append(f.name)
    if pasted.strip():
        for i, block in enumerate(pasted.split("\n---\n"), start=1):
            if block.strip():
                text = block.strip()
                name = f"Pasted Candidate {i}"
                if is_valid_resume(text):
                    candidates.append((name, text))
                else:
                    invalid_candidates.append(name)

    if invalid_candidates:
        st.session_state.pop("ranking", None)
        show_invalid_resume_dialog(
            "These files do not appear to be readable resumes: "
            + ", ".join(invalid_candidates)
            + ". No applicants were screened."
        )
    else:
        rows = []
        for name, text in candidates:
            yrs = extract_resume_metadata(text)["extracted_years"]
            scr = engine.screen(text, yrs, job_id)
            rows.append({"candidate": name, **scr})

        if rows:
            st.session_state["ranking"] = pd.DataFrame(rows).sort_values("fit_score", ascending=False).reset_index(drop=True)
        else:
            st.warning("No usable resumes detected. Please upload .txt, .pdf, or .docx files, or paste candidate text.")

ranking = st.session_state.get("ranking")
if ranking is not None and not ranking.empty:
    st.divider()
    st.subheader("Candidate Leaderboard")
    
    show_df = ranking[["candidate", "fit_score", "skill_match_pct", "text_match_pct", "level", "weeks_to_ready", "years"]].copy()
    show_df.columns = ["Candidate", "Fit Score (%)", "Skill Match (%)", "Text Match (%)", "Readiness Level", "Weeks to Ready", "Experience (Yrs)"]
    st.dataframe(show_df, width="stretch", hide_index=True)

    st.plotly_chart(charts.rank_bar(ranking.rename(columns={"candidate": "title"})), width="stretch")
    st.download_button("Export Screening Results (.csv)", show_df.to_csv(index=False), "recruiter_screening_results.csv")

    st.subheader("Applicant Fit Explanations")
    for row in ranking.itertuples():
        with st.expander(f"👤 {row.candidate} | Fit: {row.fit_score}% | Level: {row.level}"):
            m1, m2, m3 = st.columns(3)
            m1.metric("Overall Fit", f"{row.fit_score}%")
            m2.metric("Skill Alignment", f"{row.skill_match_pct}%")
            m3.metric("Preparation Needed", f"{row.weeks_to_ready} wks")

            st.markdown("**Matched Skills:** " + chips(row.matched, "ok"), unsafe_allow_html=True)
            st.markdown("**Missing Requirements:** " + chips(row.missing, "miss"), unsafe_allow_html=True)
            if getattr(row, "transferable", None):
                st.markdown("**Transferable Bridges:**")
                for t in row.transferable[:3]:
                    st.write(f"- {t['source_skill'].title()} &rarr; {t['target_skill'].title()} ({t['percentage']}% transferable)")

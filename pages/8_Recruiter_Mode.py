import pandas as pd
import streamlit as st
from core import config as cfg
from core import charts
from core.ui import setup, chips, read_upload

engine = setup("Recruiter Mode", "🧑‍💼")
st.title("🧑‍💼 Recruiter Mode")
st.caption("Screen many resumes for one job. The same models rank candidates and explain the ranking.")

c1, c2 = st.columns(2)
domain = c1.selectbox("Job domain", cfg.DOMAIN_NAMES, key="domain_recruiter")
ids = [i for i, d in enumerate(engine.job_domains) if d == domain]
job_id = c2.selectbox("Job", ids, key="job_recruiter", format_func=lambda j: f"{engine.job_titles[j]} (#{j})")
st.markdown("Required skills: " + chips(sorted(engine.job_skills[job_id]), "info"), unsafe_allow_html=True)

files = st.file_uploader("Upload resumes (.txt or .pdf, several allowed)", type=["txt", "pdf"],
                         accept_multiple_files=True, key="recruiter_files")
pasted = st.text_area("Or paste resumes separated by a line containing only ---", height=200,
                      key="recruiter_text")

if st.button("Rank candidates", type="primary"):
    candidates = [(f.name, read_upload(f)) for f in files]
    if pasted.strip():
        for i, block in enumerate(pasted.split("\n---\n"), start=1):
            if block.strip():
                candidates.append((f"Pasted resume {i}", block.strip()))
    rows = []
    for name, text in candidates:
        if len(text.split()) < 5:
            continue
        rows.append({"candidate": name, **engine.screen(text, engine.parse_years(text), job_id)})
    if rows:
        st.session_state["ranking"] = pd.DataFrame(rows).sort_values("fit_score", ascending=False).reset_index(drop=True)
    else:
        st.warning("No usable resumes found.")

ranking = st.session_state.get("ranking")
if ranking is not None and not ranking.empty:
    st.divider()
    st.subheader("Ranking")
    show = ranking[["candidate", "fit_score", "skill_match_pct", "text_match_pct", "level", "weeks_to_ready", "years"]]
    st.dataframe(show, width="stretch", hide_index=True)
    st.plotly_chart(charts.rank_bar(ranking.rename(columns={"candidate": "title"})), width="stretch")
    st.download_button("Download ranking (.csv)", show.to_csv(index=False), "candidate_ranking.csv")
    st.subheader("Why each candidate ranks here")
    for row in ranking.itertuples():
        with st.expander(f"{row.candidate} | fit {row.fit_score}%"):
            st.markdown("Matched: " + chips(row.matched, "ok"), unsafe_allow_html=True)
            st.markdown("Missing: " + chips(row.missing, "miss"), unsafe_allow_html=True)
            st.write(f"Predicted level: **{row.level}** | weeks to be fully ready: **{row.weeks_to_ready}**")
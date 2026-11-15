import sys
from pathlib import Path
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "karta hai") not in sys.path:
    sys.path.insert(0, str(ROOT / "karta hai"))

from core import config as cfg
from core import charts
from core.ui import setup, chips, resume_panel, resume_years, valid_resume_or_dialog

engine = setup("Job Recommender", "💼")
st.title("💼 Intelligent Job & Career Matcher")
st.caption("Ranks 420 active job positions against your resume using multi-dimensional skill overlap, semantic TF-IDF matching, and transferable competency bridges.")

left, right = st.columns([3, 2])
with left:
    text = resume_panel("resume_jobs", height=220)
with right:
    years = resume_years(text)
    domain = st.selectbox("Filter by career domain", ["All"] + cfg.DOMAIN_NAMES, key="domain_jobs")
    top_n = st.slider("Number of recommendations", 5, 25, 10, key="topn_jobs")
    go = st.button("Discover Best Matching Roles", type="primary", width="stretch")

if go:
    if valid_resume_or_dialog(text):
        with st.spinner("Scoring 420 job postings across domains..."):
            st.session_state["ranked"] = engine.rank_jobs(text, years, top_n, domain)

ranked = st.session_state.get("ranked")
if ranked is not None and not ranked.empty:
    st.divider()
    st.plotly_chart(charts.rank_bar(ranked), width="stretch")

    show_cols = ["title", "domain", "fit_score", "skill_match_pct", "text_match_pct", "level", "weeks_to_ready"]
    disp_df = ranked[show_cols].copy()
    disp_df.columns = ["Job Title", "Domain", "Fit Score (%)", "Skill Match (%)", "Text Match (%)", "Readiness Level", "Weeks to Ready"]
    st.dataframe(disp_df, width="stretch", hide_index=True)
    st.download_button("Export Recommendations (.csv)", disp_df.to_csv(index=False), "job_recommendations.csv")

    st.subheader("Role Deep-Dive & Match Explanations")
    for row in ranked.head(8).itertuples():
        with st.expander(f"📌 {row.title} (#{row.job_id}) | Fit: {row.fit_score}% | Readiness: {row.level}"):
            m1, m2, m3 = st.columns(3)
            m1.metric("Overall Fit", f"{row.fit_score}%")
            m2.metric("Skill Match", f"{row.skill_match_pct}%")
            m3.metric("Est. Weeks to Ready", f"{row.weeks_to_ready} wks")

            st.markdown("**Directly Matched Skills:** " + chips(row.matched, "ok"), unsafe_allow_html=True)
            st.markdown("**Missing Requirements to Acquire:** " + chips(row.missing, "miss"), unsafe_allow_html=True)
            if getattr(row, "transferable_count", 0) > 0:
                st.info(f"💡 You have {row.transferable_count} transferable skill bridges that accelerate your transition to this position.")

import streamlit as st
from core import config as cfg
from core import charts
from core.ui import setup, chips, resume_panel, years_slider

engine = setup("Job Recommender", "💼")
st.title("💼 Job Recommender")
st.caption("Ranks all 420 jobs for your resume using text similarity and skill overlap.")

left, right = st.columns([3, 2])
with left:
    text = resume_panel("resume_jobs", height=220)
with right:
    years = years_slider("years_jobs")
    domain = st.selectbox("Domain filter", ["All"] + cfg.DOMAIN_NAMES, key="domain_jobs")
    top_n = st.slider("Number of jobs", 5, 25, 10, key="topn_jobs")
    go = st.button("Find best jobs", type="primary", width="stretch")

if go:
    st.session_state["ranked"] = engine.rank_jobs(text, years, top_n, domain)

ranked = st.session_state.get("ranked")
if ranked is not None and not ranked.empty:
    st.divider()
    st.plotly_chart(charts.rank_bar(ranked), width="stretch")
    show = ranked[["title", "domain", "fit_score", "skill_match_pct", "level", "weeks_to_ready"]]
    st.dataframe(show, width="stretch", hide_index=True)
    st.subheader("Why these jobs")
    for row in ranked.head(5).itertuples():
        with st.expander(f"{row.title} (#{row.job_id}) | fit {row.fit_score}%"):
            st.markdown("Matched: " + chips(row.matched, "ok"), unsafe_allow_html=True)
            st.markdown("Missing: " + chips(row.missing, "miss"), unsafe_allow_html=True)
            st.write(f"Predicted readiness: **{row.level}** | time to ready: **{row.weeks_to_ready} weeks**")
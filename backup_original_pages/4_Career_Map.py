import streamlit as st
from core import charts
from core.preprocess import clean_text, extract_skills
from core.ui import setup, chips, resume_panel, years_slider

engine = setup("Career Map", "🗺️")
st.title("🗺️ Career Map")
st.caption("Shows where your skills sit among 5,500 profiles and which careers are closest.")

left, right = st.columns([3, 2])
with left:
    text = resume_panel("resume_map", height=200)
with right:
    years = years_slider("years_map")
    go = st.button("Show my position", type="primary", width="stretch")

if go:
    clean = clean_text(text)
    skills = extract_skills(clean)
    st.session_state["map"] = {
        "skills": skills, "point": engine.candidate_point(skills),
        "alts": engine.alternate_careers(clean, skills, years), "coverage": engine.domain_coverage(skills),
    }

m = st.session_state.get("map")
if m:
    st.divider()
    st.markdown("**Skills detected:** " + chips(m["skills"], "info"), unsafe_allow_html=True)
    c1, c2 = st.columns([3, 2])
    c1.plotly_chart(charts.career_map(engine.b["pca"]["map_df"], m["point"]), width="stretch")
    c2.plotly_chart(charts.radar(m["coverage"]), width="stretch")
    st.subheader("Closest careers for you")
    cols = st.columns(3)
    for col, a in zip(cols, m["alts"]):
        with col:
            st.markdown(f"<div class='card'><b>{a['domain']}</b><br>Skill fit: {a['fit']:.0f}%<br>"
                        f"Best job: {a['job']}<br>Your level: {a['level']}<br>"
                        f"Time to ready: {a['weeks']:.1f} weeks</div>", unsafe_allow_html=True)
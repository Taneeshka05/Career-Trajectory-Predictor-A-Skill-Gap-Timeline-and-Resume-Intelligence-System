from datetime import date, timedelta
import pandas as pd
import streamlit as st
from core import config as cfg
from core.trajectory import what_if
from core.ui import setup, chips

engine = setup("Trajectory Lab", "🚀")
st.title("🚀 Trajectory Lab")
st.caption("Try what-if plans on your last analysis and compare study schedules.")

r = st.session_state.get("analysis")
if not r:
    st.info("Run an analysis on the Analyze Resume page first.")
    st.stop()
if not r["missing"]:
    st.success("You already match every skill for this job.")
    st.stop()

st.subheader(f"Target: {r['job_title']}")
st.markdown("### What if I learn these skills?")
c1, c2 = st.columns([3, 2])
chosen = c1.multiselect("Skills I will learn", r["missing"], default=r["missing"][:1])
hours = c2.slider("Study hours per week", 3, 40, cfg.HOURS_BASELINE, key="hours_lab")

if chosen:
    w = what_if(engine, r, chosen)
    b, a = w["before"], w["after"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Skill match", f"{a['skill_match_pct']:.1f}%", f"{a['skill_match_pct'] - b['skill_match_pct']:+.1f}%")
    m2.metric("Text match", f"{a['match_pct']:.1f}%", f"{a['match_pct'] - b['match_pct']:+.1f}%")
    m3.metric("Predicted level", a["level"], "changed" if a["level"] != b["level"] else "same")
    m4.metric("Weeks still needed", f"{a['weeks_left']:.1f}", f"{a['weeks_left'] - b['weeks_left']:+.1f}")
    weeks = engine.learning_weeks(w["order"], r, hours)
    st.write(f"Learning plan: **{' -> '.join(w['order'])}** will take about **{weeks:.1f} weeks** at {hours} hours/week "
             f"(done by {(date.today() + timedelta(weeks=weeks)).strftime('%d %b %Y')}).")
    if w["foundations"]:
        st.markdown("Foundation skills added automatically: " + chips(w["foundations"], "info"),
                    unsafe_allow_html=True)
    if a["level"] != b["level"]:
        st.success(f"This plan moves you from {b['level']} to {a['level']}.")

st.markdown("### Compare study schedules")
rows = []
for h in [5, 10, 20, 30, 40]:
    tl = engine.timeline(r["missing"], r["cand_skills"], r["weeks"], h)
    total = float(tl["end"].max())
    rows.append({"hours per week": h, "total weeks": round(total, 1),
                 "ready by": (date.today() + timedelta(weeks=total)).strftime("%d %b %Y")})
plan = pd.DataFrame(rows)
st.bar_chart(plan.set_index("hours per week")["total weeks"])
st.dataframe(plan, width="stretch", hide_index=True)
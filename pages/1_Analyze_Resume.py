from datetime import date, timedelta
import pandas as pd
import streamlit as st
from core import config as cfg
from core import database as db
from core import charts, explain
from core.nlp_feedback import highlight_html
from core.prerequisites import dependency_dot
from core.report import build_html_report
from core.trajectory import build_trajectory, level_changes, skill_roi
from core.ui import setup, chips, resume_panel, years_slider, profile

engine = setup("Analyze Resume", "🎯")
st.title("🎯 Analyze Resume")
st.caption("Match score, roadmap, career trajectory, best next skill, explanations and resume feedback.")

left, right = st.columns([3, 2])
with left:
    text = resume_panel("resume_analyze")
with right:
    years = years_slider("years_analyze")
    domain = st.selectbox("Target domain", cfg.DOMAIN_NAMES, key="domain_analyze")
    ids = [i for i, d in enumerate(engine.job_domains) if d == domain]
    job_id = st.selectbox("Target job", ids, key="job_analyze",
                          format_func=lambda j: f"{engine.job_titles[j]} (#{j})")
    run = st.button("Analyze", type="primary", width="stretch")

if run:
    if len(text.split()) < 5:
        st.error("Please enter a longer resume text.")
    else:
        with st.spinner("Analyzing..."):
            st.session_state["analysis"] = engine.analyze(text, years, job_id)

r = st.session_state.get("analysis")
if r:
    owned = r["cand_skills"]
    tl0 = engine.timeline(r["missing"], owned, r["weeks"], cfg.HOURS_BASELINE)
    traj0 = build_trajectory(engine, r["resume_text"], r["years"], r["job_id"], owned, tl0)
    roi = skill_roi(engine, r["resume_text"], r["years"], r["job_id"], owned, r["missing"])

    st.divider()
    st.subheader(f"Report for {r['job_title']}")
    tabs = st.tabs(["Overview", "Skill Gap", "Roadmap", "Trajectory", "Learn First", "Why This Result",
                    "NLP Feedback", "Career Paths", "Skill Levels"])

    with tabs[0]:
        g1, g2, g3 = st.columns(3)
        g1.plotly_chart(charts.gauge(r["match_pct"], "Text match"), width="stretch")
        g2.plotly_chart(charts.gauge(r["skill_match_pct"], "Skill match"), width="stretch")
        g3.plotly_chart(charts.gauge(r["strength"], "Resume strength Score"), width="stretch")
        m1, m2, m3 = st.columns(3)
        m1.metric("Readiness level", r["level"])
        m2.metric("Estimated time to ready", f"{r['weeks']:.1f} weeks")
        m3.metric("Skills found in resume", len(r["cand_skills"]))
        b1, b2, b3 = st.columns(3)
        if db.history_enabled():
            if b1.button("Save this analysis", width="stretch"):
                aid = db.save_analysis(profile(), r, engine.plan_for_save(r))
                st.success(f"Saved as analysis #{aid}. Track progress on the History page.")
        else:
            b1.caption("Saving is disabled in public demo mode.")
        b2.download_button("Download report (.txt)", engine.report_text(r), "careerpath_report.txt",
                           width="stretch")
        b3.download_button("Download full report (.html)", build_html_report(r, tl0, traj0, roi),
                           "careerpath_report.html", mime="text/html", width="stretch")

    with tabs[1]:
        c1, c2 = st.columns(2)
        c1.markdown("**Skills you already have for this job**")
        c1.markdown(chips(r["matched"], "ok"), unsafe_allow_html=True)
        c2.markdown("**Missing skills (hardest first)**")
        c2.markdown(chips(r["missing"], "miss"), unsafe_allow_html=True)
        if r["missing"]:
            st.plotly_chart(charts.skill_gap_bar(r["missing"], r["weeks_by_skill"]), width="stretch")
        else:
            st.success("You already have every skill listed for this job.")

    with tabs[2]:
        hours = st.slider("Study hours per week", 3, 40, cfg.HOURS_BASELINE, key="hours_analyze")
        tl = engine.timeline(r["missing"], owned, r["weeks"], hours)
        if tl.empty:
            st.success("Nothing to learn for this job.")
        else:
            total = float(tl["end"].max())
            n_found = int((tl["type"] == "foundation").sum())
            t1, t2, t3 = st.columns(3)
            t1.metric("Total time", f"{total:.1f} weeks")
            t2.metric("Ready by", (date.today() + timedelta(weeks=total)).strftime("%d %b %Y"))
            t3.metric("Foundation skills added", n_found)
            st.plotly_chart(charts.gantt(tl), width="stretch")
            st.markdown("**Skill dependency map** (green = you have it, red = target skill, orange = foundation)")
            st.graphviz_chart(dependency_dot(tl["skill"].tolist(), owned, r["missing"]))
            st.dataframe(tl.round(1), width="stretch", hide_index=True)
            st.caption("The roadmap respects prerequisites: you cannot learn Deep Learning before Machine Learning, "
                       "so missing prerequisites are added automatically as foundation skills.")

    with tabs[3]:
        if tl0.empty:
            st.success("No trajectory needed. You already match this job.")
        else:
            st.plotly_chart(charts.trajectory_chart(traj0), width="stretch")
            changes = level_changes(traj0)
            if changes.empty:
                st.info(f"Predicted level stays {r['level']} even after completing the roadmap.")
            for row in changes.itertuples():
                st.success(f"Week {row.week:.1f}: after learning {row.skill} you move to {row.level} level.")
            st.dataframe(traj0.round(1), width="stretch", hide_index=True)
            st.caption("Each point re-runs the trained models after adding the skills learned so far.")

    with tabs[4]:
        if roi.empty:
            st.success("Nothing to prioritise.")
        else:
            best = roi.iloc[0]
            st.success(f"Learn **{best['skill']}** first: +{best['match_gain_pct']:.1f}% skill match "
                       f"for {best['cost_weeks']:.0f} base weeks (foundations: {best['foundations']}).")
            st.plotly_chart(charts.roi_bar(roi), width="stretch")
            st.dataframe(roi.round(2), width="stretch", hide_index=True)

    with tabs[5]:
        intercept, wdf = explain.weeks_contributions(engine, r["feats"])
        ldf = explain.level_contributions(engine, r["feats"], r["level"])
        e1, e2 = st.columns(2)
        e1.plotly_chart(charts.contrib_bar(wdf, "What drives the time estimate", "weeks"), width="stretch")
        e2.plotly_chart(charts.contrib_bar(ldf, f"What pushes you towards {r['level']}", "log-odds"),
                        width="stretch")
        for line in explain.sentences(wdf, "weeks"):
            st.write("- " + line)
        st.caption("Contributions are coefficient x normalized feature value from the linear models, so "
                   "base value + contributions equals the prediction. Red increases, green decreases.")

    with tabs[6]:
        n = r["nlp"]
        st.markdown("**Your resume with highlights** (red = weak phrase, green = action verb)")
        st.markdown(f"<div class='resume-box'>{highlight_html(r['resume_text'])}</div>", unsafe_allow_html=True)
        f1, f2, f3, f4 = st.columns(4)
        f1.metric("Language score (NN)", f"{r['lang_prob'] * 100:.0f}%")
        f2.metric("Action verbs", n["n_strong"])
        f3.metric("Weak phrases", n["n_weak"])
        f4.metric("Measurable results", n["quantified"])
        st.markdown("**What you have**")
        st.markdown(chips(n["strong"], "ok") + chips([k for k, v in n["sections"].items() if v], "info"),
                    unsafe_allow_html=True)
        st.markdown("**What can be improved**")
        for tip in n["tips"]:
            st.write("- " + tip)

    with tabs[7]:
        alt = pd.DataFrame(r["alternates"])[["domain", "fit", "job", "level", "weeks"]]
        alt.columns = ["Career", "Skill fit %", "Best matching job", "Your level", "Weeks to ready"]
        st.dataframe(alt.round(1), width="stretch", hide_index=True)
        st.plotly_chart(charts.radar(r["coverage"]), width="stretch")

    with tabs[8]:
        lv = r["skill_levels"]
        if lv.empty:
            st.info("No known skills found in the resume.")
        else:
            st.dataframe(lv, width="stretch", hide_index=True)
            st.caption("Level is based on how strongly the resume evidences the skill: one mention = "
                       "Beginner, two = Intermediate, three or more = Advanced.")
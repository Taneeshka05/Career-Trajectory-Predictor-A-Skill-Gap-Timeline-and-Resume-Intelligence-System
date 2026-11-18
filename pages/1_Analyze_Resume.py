import sys
from pathlib import Path
from datetime import date, timedelta
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "karta hai") not in sys.path:
    sys.path.insert(0, str(ROOT / "karta hai"))

from core import config as cfg
from core import database as db
from core import charts, explain
from core.nlp_feedback import highlight_html, generate_star_bullet_suggestions
from core.prerequisites import dependency_dot
from core.report import build_html_report
from core.trajectory import build_trajectory, level_changes, skill_roi
from core.ui import setup, chips, resume_panel, resume_years, valid_resume_or_dialog, profile

engine = setup("Analyze Resume", "🎯")
st.title("🎯 Comprehensive Resume & Career Trajectory Analysis")
st.caption("Multi-dimensional job matching, skill gap taxonomy, transferable bridges, dynamic timeline & explainable AI.")

left, right = st.columns([3, 2])
with left:
    text = resume_panel("resume_analyze")
with right:
    years = resume_years(text)
    domain = st.selectbox("Target career domain", cfg.DOMAIN_NAMES, key="domain_analyze")
    ids = [i for i, d in enumerate(engine.job_domains) if d == domain]
    job_id = st.selectbox("Target job position", ids, key="job_analyze",
                          format_func=lambda j: f"{engine.job_titles[j]} (#{j})")
    run = st.button("Run Full Career Analysis", type="primary", width="stretch")

if run:
    if valid_resume_or_dialog(text):
        with st.spinner("Analyzing text, parsing taxonomy, calculating transferable bridges & running ML models..."):
            res = engine.analyze(text, years, job_id)
            st.session_state["analysis"] = res
            doc_name = (
                st.session_state.get("_last_uploaded_filename")
                or st.session_state.get("resume_analyze_filename")
                or f"Candidate Resume ({date.today().strftime('%b %d')})"
            )
            if db.history_enabled():
                try:
                    aid = db.save_analysis(profile(), res, engine.plan_for_save(res), filename=doc_name)
                    st.session_state["last_saved_analysis_id"] = aid
                    st.toast(f"✅ Analysis recorded to History as Record #{aid} ({doc_name})")
                except Exception:
                    pass

r = st.session_state.get("analysis")
if r:
    owned = r["cand_skills"]
    tl0 = engine.timeline(r["missing"], owned, r["weeks"], cfg.HOURS_BASELINE)
    traj0 = build_trajectory(engine, r["resume_text"], r["years"], r["job_id"], owned, tl0)
    roi = skill_roi(engine, r["resume_text"], r["years"], r["job_id"], owned, r["missing"])

    st.divider()
    st.subheader(f"Career Intelligence Report: {r['job_title']}")
    
    tabs = st.tabs([
        "Overview & Fit",
        "Skill Gap & Taxonomy",
        "Transferable Bridges",
        "Learning Roadmap",
        "Career Trajectory",
        "Skill Acquisition ROI",
        "ML Explainability",
        "NLP & Action Verbs",
        "Readiness Audit"
    ])

    # 1. OVERVIEW & FIT TAB
    with tabs[0]:
        c1, c2, c3, c4 = st.columns(4)
        c1.plotly_chart(charts.gauge(r.get("composite_compat", r["match_pct"]), "Overall Compatibility"), width="stretch")
        c2.plotly_chart(charts.gauge(r["skill_match_pct"], "Direct Skill Match"), width="stretch")
        c3.plotly_chart(charts.gauge(r.get("transferable_match_pct", r["skill_match_pct"]), "Transferable Match"), width="stretch")
        c4.plotly_chart(charts.gauge(r["strength"], "Resume Content Score"), width="stretch")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Predicted Readiness Level", r["level"])
        m2.metric("Estimated Preparation Time", f"{r['weeks']:.1f} weeks")
        m3.metric("Direct Technical Skills", f"{len(r['cand_skills'])} found")
        m4.metric("Soft Skills & Methods", f"{len(r.get('cand_soft_skills', []))} detected")

        # Multi-dimensional radar breakdown
        fit_breakdown = {
            "Direct Technical Skills": min(r["skill_match_pct"], 100.0),
            "Transferable Skills Credit": min(r.get("transferable_match_pct", r["skill_match_pct"]), 100.0),
            "Textual Semantic Similarity": min(r["match_pct"] * 3.5, 100.0),
            "Soft Skills Alignment": r.get("soft_match_pct", 75.0),
            "Experience Level Fit": min((r["years"] / (1 if "junior" in r["job_title"].lower() else (3 if "mid" in r["job_title"].lower() else 6))) * 100.0, 100.0),
        }
        st.plotly_chart(charts.compatibility_breakdown_radar(fit_breakdown), width="stretch")

        # Degrees & Certifications from parser
        meta = r.get("metadata", {})
        if meta.get("degrees") or meta.get("certifications"):
            st.markdown(
                f"<div class='card-dark'>🎓 <b>Detected Qualifications:</b> "
                f"{', '.join(meta.get('degrees', [])) or 'No standard degree pattern'} &nbsp;|&nbsp; "
                f"📜 <b>Certifications:</b> {', '.join(meta.get('certifications', [])) or 'None detected'}</div>",
                unsafe_allow_html=True
            )

        b1, b2, b3 = st.columns(3)
        if db.history_enabled():
            if b1.button("Save this analysis to History", width="stretch"):
                aid = db.save_analysis(profile(), r, engine.plan_for_save(r))
                st.success(f"Saved analysis #{aid}. Track progress on the History page.")
        else:
            b1.caption("History saving disabled in demo mode.")
        b2.download_button("Download Summary Report (.txt)", engine.report_text(r), "careerpath_summary.txt", width="stretch")
        b3.download_button("Download Full Visual Report (.html)", build_html_report(r, tl0, traj0, roi),
                           "careerpath_report.html", mime="text/html", width="stretch")

    # 2. SKILL GAP & TAXONOMY TAB
    with tabs[1]:
        st.markdown("### Skill Taxonomy & Requirement Breakdown")
        col_matched, col_missing = st.columns(2)
        with col_matched:
            st.markdown(f"**Matched Skills ({len(r['matched'])})**")
            st.markdown(chips(r["matched"], "ok"), unsafe_allow_html=True)
            if r.get("cand_soft_skills"):
                st.markdown(f"**Detected Soft Skills ({len(r['cand_soft_skills'])})**")
                st.markdown(chips(r["cand_soft_skills"], "info"), unsafe_allow_html=True)
        with col_missing:
            st.markdown(f"**Missing Target Skills ({len(r['missing'])})**")
            st.markdown(chips(r["missing"], "miss"), unsafe_allow_html=True)

        if r["missing"]:
            st.plotly_chart(charts.skill_gap_bar(r["missing"], r["weeks_by_skill"]), width="stretch")
            
            # Gap explanations table
            if r.get("gap_explanations"):
                st.markdown("#### Detailed Skill Gap Explanations")
                gap_df = pd.DataFrame(r["gap_explanations"])[
                    ["skill", "category", "difficulty", "priority", "hours", "explanation"]
                ]
                gap_df.columns = ["Missing Skill", "Taxonomy Category", "Difficulty (1-5)", "Priority", "Est. Hours", "Role Context"]
                st.dataframe(gap_df, width="stretch", hide_index=True)
        else:
            st.success("Target Alignment Complete: You already possess every required skill for this role!")

    # 3. TRANSFERABLE BRIDGES TAB
    with tabs[2]:
        st.markdown("### Cross-Domain Transferable Skill Credits")
        st.caption("Identifies complementary competencies you hold that accelerate mastering missing target requirements.")
        transferable = r.get("transferable", [])
        if transferable:
            st.info(f"You have **{len(transferable)} transferable skill bridges** that reduce your learning curve!")
            t_rows = []
            for t in transferable:
                t_rows.append({
                    "Target Missing Skill": t["target_skill"].title(),
                    "Your Related Skill": t["source_skill"].title(),
                    "Transferability Credit": f"{t['percentage']}%",
                    "Pedagogical Rationale": t["reason"]
                })
            st.dataframe(pd.DataFrame(t_rows), width="stretch", hide_index=True)
            
            st.markdown("#### Transferable Bridges Breakdown")
            for t in transferable[:4]:
                st.markdown(
                    f"<div class='card-dark' style='margin-bottom:8px;'>"
                    f"🔗 <b>{t['source_skill'].title()}</b> &rarr; <b>{t['target_skill'].title()}</b> "
                    f"<span class='chip chip-transfer'>{t['percentage']}% Transferable</span><br>"
                    f"<span style='color:#cbd5e1;font-size:0.9rem;'>{t['reason']}</span></div>",
                    unsafe_allow_html=True
                )
        else:
            st.info("No direct transferable skill bridges identified for the current missing set.")

    # 4. LEARNING ROADMAP TAB
    with tabs[3]:
        hours = st.slider("Weekly Study Commitment (Hours)", 3, 40, cfg.HOURS_BASELINE, key="hours_analyze")
        tl = engine.timeline(r["missing"], owned, r["weeks"], hours)
        if tl.empty:
            st.success("Roadmap not required: all skills are currently satisfied.")
        else:
            total = float(tl["end"].max())
            n_found = int((tl["type"] == "foundation").sum())
            t1, t2, t3 = st.columns(3)
            t1.metric("Projected Total Duration", f"{total:.1f} weeks")
            t2.metric("Estimated Job-Ready Date", (date.today() + timedelta(weeks=total)).strftime("%d %b %Y"))
            t3.metric("Prerequisite Foundations Added", n_found)

            st.plotly_chart(charts.gantt(tl), width="stretch")
            st.markdown("**Skill Dependency Graph** (Green = Owned, Red = Target Skill, Orange = Prerequisite Foundation)")
            st.graphviz_chart(dependency_dot(tl["skill"].tolist(), owned, r["missing"]))
            
            st.markdown("#### Ordered Milestone Schedule")
            disp_tl = tl.copy()
            disp_tl["start"] = disp_tl["start"].round(1)
            disp_tl["weeks"] = disp_tl["weeks"].round(1)
            disp_tl["end"] = disp_tl["end"].round(1)
            disp_tl.columns = ["Skill", "Requirement Type", "Difficulty", "Start (Week)", "Duration (Weeks)", "Target End (Week)"]
            st.dataframe(disp_tl, width="stretch", hide_index=True)

    # 5. CAREER TRAJECTORY TAB
    with tabs[4]:
        if tl0.empty:
            st.success("Full match: no transition trajectory needed.")
        else:
            st.plotly_chart(charts.trajectory_chart(traj0), width="stretch")
            changes = level_changes(traj0)
            if changes.empty:
                st.info(f"Predicted readiness stays at {r['level']} level throughout current milestones.")
            for row in changes.itertuples():
                st.success(f"🚀 **Milestone (Week {row.week:.1f})**: After mastering **{row.skill}**, you elevate to **{row.level}** readiness.")
            st.dataframe(traj0.round(1), width="stretch", hide_index=True)

    # 6. SKILL ACQUISITION ROI TAB
    with tabs[5]:
        if roi.empty:
            st.success("All target skills acquired.")
        else:
            best = roi.iloc[0]
            st.success(f"💡 **Top Strategic Priority:** Learn **{best['skill'].title()}** first (+{best['match_gain_pct']:.1f}% match gain for {best['cost_weeks']:.0f} study weeks).")
            st.plotly_chart(charts.roi_bar(roi), width="stretch")
            st.dataframe(roi.round(2), width="stretch", hide_index=True)

    # 7. ML EXPLAINABILITY TAB
    with tabs[6]:
        intercept, wdf = explain.weeks_contributions(engine, r["feats"])
        ldf = explain.level_contributions(engine, r["feats"], r["level"])
        e1, e2 = st.columns(2)
        e1.plotly_chart(charts.contrib_bar(wdf, "Feature Impact on Preparation Time", "Weeks"), width="stretch")
        e2.plotly_chart(charts.contrib_bar(ldf, f"Feature Impact on {r['level']} Readiness", "Log-Odds"), width="stretch")
        st.markdown("#### Plain-English Model Decision Rationale")
        for line in explain.sentences(wdf, "weeks"):
            st.write("- " + line)
        st.caption("Explainability Method: Linear model feature attribution (coefficient × normalized feature z-score).")

    # 8. NLP & ACTION VERBS TAB
    with tabs[7]:
        n = r["nlp"]
        st.markdown("**Highlighted Resume Content** (Green = Strong Action Verb, Red = Passive / Weak Phrase)")
        st.markdown(f"<div class='resume-box'>{highlight_html(r['resume_text'])}</div>", unsafe_allow_html=True)
        
        f1, f2, f3, f4 = st.columns(4)
        f1.metric("Language Score (NN)", f"{r['lang_prob'] * 100:.0f}%")
        f2.metric("Action Verbs", n["n_strong"])
        f3.metric("Weak / Passive Phrases", n["n_weak"])
        f4.metric("Quantified Metrics", n["quantified"])

        # Contextual STAR Bullet Suggestions
        st.markdown("#### 🎯 Contextual STAR Bullet Point Recommendations")
        st.caption("Actionable bullet templates crafted for your detected skills. Customize metrics to reflect genuine accomplishments:")
        star_bullets = generate_star_bullet_suggestions(r["cand_skills"])
        for b in star_bullets:
            st.markdown(
                f"<div class='card-dark' style='margin-bottom:8px;'>"
                f"<b>{b['category']} ({b['skill']}):</b><br>"
                f"<code>&bull; {b['template']}</code><br>"
                f"<span style='font-size:0.8rem;color:#a5b4fc;'>Formula: {b['formula']}</span></div>",
                unsafe_allow_html=True
            )

        st.markdown("#### Improvement Action Items")
        for tip in n["tips"]:
            st.write("- " + tip)

    # 9. READINESS AUDIT TAB
    with tabs[8]:
        st.markdown("### Granular Skill Readiness & Verification Audit")
        st.caption("Distinguishes explicit resume evidence, algorithmic inferred proficiency, and user self-confirmation.")
        lv = r["skill_levels"]
        if lv.empty:
            st.info("No known technical skills detected in the resume.")
        else:
            st.dataframe(lv, width="stretch", hide_index=True)
            st.info(
                "Proficiency Inference Rubric: 1 mention = Beginner (Foundational awareness); "
                "2 mentions = Intermediate (Applied in projects); 3+ mentions = Advanced (Core specialization)."
            )

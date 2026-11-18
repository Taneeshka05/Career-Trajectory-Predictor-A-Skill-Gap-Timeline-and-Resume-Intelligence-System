import streamlit as st
from core import config as cfg
from core import database as db
from core.ui import profile, resume_panel, resume_years, setup, chips, valid_resume_or_dialog


def home():
    engine = setup("Home", "🧭")

    st.markdown("""
    <div class='hero'>
      <div class='hero-kicker'>Career Intelligence & Trajectory Platform</div>
      <h1>Career Trajectory Predictor & Skill Gap Timeline Analyzer</h1>
      <p>Transform your resume into a data-driven career roadmap. Evaluate multi-dimensional job compatibility,
      discover cross-domain transferable skill bridges, forecast week-by-week readiness timelines, and benchmark machine learning models.</p>
      <a class='hero-cta' href='#workspace'>Launch Career Analysis Workspace <span aria-hidden='true'>&darr;</span></a>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='intro-grid'>
      <div class='intro-tile'>
        <strong>Intelligent Job Fit & Matching</strong>
        <span>Multi-dimensional scoring combining direct technical skills, semantic TF-IDF text similarity, and transferable competency bridges.</span>
      </div>
      <div class='intro-tile'>
        <strong>Skill Gap & Taxonomy Roadmap</strong>
        <span>Hierarchical taxonomy analysis with prerequisite-aware DAG scheduling, weekly milestone planning, and ROI prioritization.</span>
      </div>
      <div class='intro-tile'>
        <strong>Real Machine Learning Lab</strong>
        <span>Leakage-free ML training and validation engine supporting custom CSV/Excel uploads, EDA, model comparisons, and live prediction inference.</span>
      </div>
    </div>
    <div id='workspace' class='workspace-heading'></div>
    """, unsafe_allow_html=True)

    st.subheader("Start with a Resume")
    st.caption("Upload your resume (.txt, .pdf, or .docx) or paste valid resume text to compute personalized trajectory metrics.")

    resume_text = resume_panel("home_resume", height=220)

    first, second = st.columns(2)
    with first:
        domain = st.selectbox("Target Career Domain", cfg.DOMAIN_NAMES, key="home_domain")
        years = resume_years(resume_text)
    with second:
        job_ids = [i for i, item_domain in enumerate(engine.job_domains) if item_domain == domain]
        job_id = st.selectbox(
            "Target Position",
            job_ids,
            key="home_job",
            format_func=lambda job: engine.job_titles[job],
        )

    if st.button("Compute Career Compatibility & Timeline", type="primary", width="stretch"):
        if valid_resume_or_dialog(resume_text):
            with st.spinner("Evaluating career compatibility, skill taxonomy, and regression models..."):
                result = engine.analyze(resume_text, years, job_id)
            st.session_state["home_result"] = result
            st.session_state["analysis"] = result
            doc_name = (
                st.session_state.get("_last_uploaded_filename")
                or st.session_state.get("home_resume_filename")
                or "Candidate Resume (Home)"
            )
            if db.history_enabled():
                try:
                    aid = db.save_analysis(profile(), result, engine.plan_for_save(result), filename=doc_name)
                    st.session_state["home_saved_analysis_id"] = aid
                    st.toast(f"✅ Analysis saved to History as #{aid} ({doc_name})")
                except Exception:
                    pass

    result = st.session_state.get("home_result")
    if result:
        inputs_match = (
            result["resume_text"] == resume_text
            and result["years"] == years
            and result["job_id"] == job_id
        )
        if not inputs_match:
            st.info("Inputs have changed. Click 'Compute Career Compatibility' to refresh output.")
        else:
            st.divider()
            st.subheader(f"Career Fit & Trajectory for {result['job_title']}")

            metrics = st.columns(4)
            compat = result.get("composite_compat", result["match_pct"])
            metrics[0].metric("Overall Compatibility", f"{compat:.1f} / 100")
            metrics[1].metric("Direct Skill Match", f"{result['skill_match_pct']:.1f}%")
            metrics[2].metric("Resume Strength", f"{result['strength']:.0f} / 100")
            metrics[3].metric("Est. Weeks to Ready", f"{result['weeks']:.1f} wks")

            st.progress(
                min(max(result["skill_match_pct"] / 100.0, 0.0), 1.0),
                text=f"{result['level']} Readiness &bull; {len(result['matched'])} of "
                f"{len(result['job_skills'])} Core Requirements Met",
            )

            matched, missing = st.columns(2)
            with matched:
                st.markdown("**Core Skills Found in Resume**")
                st.markdown(chips(result["matched"], "ok"), unsafe_allow_html=True)
            with missing:
                st.markdown("**Target Gaps to Acquire**")
                st.markdown(chips(result["missing"], "miss"), unsafe_allow_html=True)

            if result.get("transferable"):
                st.markdown("**Transferable Skill Bridges Detected**")
                trans_badges = [f"{t['source_skill']} &rarr; {t['target_skill']} ({t['percentage']}%)" for t in result["transferable"][:3]]
                st.markdown(chips(trans_badges, "transfer"), unsafe_allow_html=True)

            if db.history_enabled():
                if st.button("Save this result to History", key="save_home_result"):
                    analysis_id = db.save_analysis(
                        profile(), result, engine.plan_for_save(result)
                    )
                    st.session_state["home_saved_analysis_id"] = analysis_id
                saved_id = st.session_state.get("home_saved_analysis_id")
                if saved_id:
                    st.success(f"Saved result #{saved_id}. Review your learning progress on the History page.")
            else:
                st.caption("History storage disabled in public demo mode.")

            st.page_link(
                "pages/1_Analyze_Resume.py",
                label="Explore Detailed Multi-Tab Analysis (Roadmap, Trajectory & Explainability)",
                icon=":material/analytics:",
            )

    with st.expander("Understanding Analytical Scores & Academic Methodology"):
        st.markdown(
            "- **Overall Compatibility**: Multi-dimensional composite score reflecting direct skill overlap (40%), transferable skill credit (15%), TF-IDF semantic match (20%), soft skills (15%), and experience alignment (10%).\n"
            "- **Direct Skill Match**: Percentage of mandatory job skills explicitly detected in the candidate's text.\n"
            "- **Weeks to Ready**: Supervised regression prediction (Linear/Ridge) estimating total learning duration scaled to weekly study availability.\n"
            "- **Readiness Level**: Supervised classification prediction (Junior / Mid / Senior) based on normalized feature vectors.\n\n"
            "Predictions are analytical estimations produced by models trained on structured datasets. They provide strategic guidance rather than a hiring guarantee."
        )


home()

import streamlit as st

from core import config as cfg
from core import database as db
from core.ui import profile, resume_panel, setup


def home():
    engine = setup("Home", "🧭")

    st.markdown("""
    <div class='hero'>
    <div class='hero-kicker'>Your next opportunity starts here</div>
    <h1>CareerPath AI</h1>
    <p>Turn your resume into a clear career plan. Explore your fit for real roles, find the skills
    you are missing, and see the next steps toward your target career.</p>
    <a class='hero-cta' href='#workspace'>Explore the career workspace <span aria-hidden='true'>↓</span></a>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='intro-grid'>
      <div class='intro-tile'><strong>Understand your fit</strong>
        <span>Compare your resume with a role and see a clear match and readiness summary.</span></div>
      <div class='intro-tile'><strong>Find your next skills</strong>
        <span>See which requirements you already meet and what to focus on learning next.</span></div>
      <div class='intro-tile'><strong>Build a practical plan</strong>
        <span>Explore career paths, learning timelines, and detailed feedback in one workspace.</span></div>
    </div>
    <div id='workspace' class='workspace-heading'></div>
    """, unsafe_allow_html=True)

    st.subheader("Start with a resume")
    st.caption("Try the sample resume below, or replace it with your own text or PDF.")
    st.info("A sample resume is preloaded for demonstration. Its analysis is an example, not your personal profile.")
    resume_text = resume_panel("home_resume", height=220, default_text=cfg.DEMO_RESUME)

    first, second = st.columns(2)
    with first:
        domain = st.selectbox("Career area", cfg.DOMAIN_NAMES, key="home_domain")
        years = st.slider("Years of experience", 0, 15, cfg.DEMO_YEARS, key="home_years")
    with second:
        job_ids = [i for i, item_domain in enumerate(engine.job_domains) if item_domain == domain]
        job_id = st.selectbox(
            "Target role",
            job_ids,
            key="home_job",
            format_func=lambda job: engine.job_titles[job],
        )

    if "home_result" not in st.session_state and resume_text == cfg.DEMO_RESUME:
        st.session_state["home_result"] = engine.analyze(resume_text, years, job_id)
        st.session_state["home_result_is_demo"] = True

    if st.button("Analyze my resume", type="primary", width="stretch"):
        if len(resume_text.split()) < 5:
            st.error("Please enter at least five words of resume text before analyzing.")
        else:
            with st.spinner("Matching your resume to the selected role..."):
                result = engine.analyze(resume_text, years, job_id)
            st.session_state["home_result"] = result
            st.session_state["home_result_is_demo"] = resume_text == cfg.DEMO_RESUME
            st.session_state["analysis"] = result
            st.session_state.pop("home_saved_analysis_id", None)

    result = st.session_state.get("home_result")
    if result:
        inputs_match = (
            result["resume_text"] == resume_text
            and result["years"] == years
            and result["job_id"] == job_id
        )
        if not inputs_match:
            st.info("Your inputs changed. Analyze again to refresh these results.")
        else:
            st.divider()
            is_demo = st.session_state.get("home_result_is_demo", False)
            st.subheader(
                f"Sample results for {result['job_title']}" if is_demo
                else f"Your results for {result['job_title']}"
            )
            if is_demo:
                st.caption("Example output from the built-in sample resume. Replace the sample to see your own results.")
            metrics = st.columns(4)
            metrics[0].metric("Job match", f"{result['match_pct']:.1f}%")
            metrics[1].metric("Skill match", f"{result['skill_match_pct']:.1f}%")
            metrics[2].metric("Resume strength", f"{result['strength']:.0f}/100")
            metrics[3].metric("Estimated weeks to ready", f"{result['weeks']:.1f}")

            st.progress(
                min(max(result["skill_match_pct"] / 100, 0.0), 1.0),
                text=f"{result['level']} readiness · {len(result['matched'])} of "
                f"{len(result['job_skills'])} required skills matched",
            )

            matched, missing = st.columns(2)
            with matched:
                st.markdown("**Skills found in your resume**")
                st.markdown(", ".join(result["matched"]) or "No required skills detected yet.")
            with missing:
                st.markdown("**Skills to focus on next**")
                st.markdown(", ".join(result["missing"]) or "No required skills are missing.")

            if not is_demo and db.history_enabled():
                if st.button("Save this result to History", key="save_home_result"):
                    analysis_id = db.save_analysis(
                        profile(), result, engine.plan_for_save(result)
                    )
                    st.session_state["home_saved_analysis_id"] = analysis_id
                saved_id = st.session_state.get("home_saved_analysis_id")
                if saved_id:
                    st.success(f"Saved result #{saved_id}. You can review it on the History page.")
            elif not is_demo:
                st.caption("History saving is disabled in public demo mode.")

            st.page_link(
                "pages/1_Analyze_Resume.py",
                label="Open the detailed analysis",
                icon=":material/analytics:",
            )

    with st.expander("How to read these results"):
        st.markdown(
            "- **Job match** compares the wording of your resume with the selected job description.\n"
            "- **Skill match** measures how many of that job's listed skills appear in your resume.\n"
            "- **Weeks to ready** is a model estimate based on your skill gap and experience.\n\n"
            "The training data is synthetic, so use these results as guidance—not as a hiring decision "
            "or a guarantee of how long learning will take."
        )


home()

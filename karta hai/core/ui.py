import hashlib
import sys
from pathlib import Path
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import config as cfg
from core import database as db
from core.engine import CareerEngine
from core.parser import extract_resume_metadata, is_valid_resume, parse_document

CSS = """
<style>
.stApp {background: linear-gradient(145deg, #080d1a 0%, #0c1224 48%, #101022 100%);
        background-attachment: fixed; isolation: isolate;}
.stApp::before {content: ""; position: fixed; inset: -20%; z-index: 0; pointer-events: none;
                background: radial-gradient(ellipse at 18% 18%, rgba(79, 70, 229, .22), transparent 38%),
                            radial-gradient(ellipse at 82% 32%, rgba(147, 51, 234, .16), transparent 34%),
                            radial-gradient(ellipse at 58% 88%, rgba(14, 165, 233, .12), transparent 40%);
                background-size: 130% 130%; animation: ambient-drift 28s ease-in-out infinite alternate;}
.stApp::after {content: ""; position: fixed; left: -20%; right: -20%; bottom: -34vh; height: 62vh;
               z-index: 0; pointer-events: none; opacity: .14;
               background-image: linear-gradient(rgba(148, 163, 184, .24) 1px, transparent 1px),
                                 linear-gradient(90deg, rgba(148, 163, 184, .24) 1px, transparent 1px);
               background-size: 46px 46px;
               transform: perspective(420px) rotateX(58deg);
               mask-image: linear-gradient(to top, black, transparent 85%);}
@keyframes ambient-drift {
    from {transform: translate3d(-1.5%, -1%, 0) scale(1);}
    to {transform: translate3d(1.5%, 1%, 0) scale(1.04);}
}
.stApp > header {background: transparent;}
.stApp [data-testid="stAppViewContainer"],
.stApp [data-testid="stSidebar"],
.stApp .main {position: relative; z-index: 1; background: transparent;}
.stApp [data-testid="stSidebar"] {background: rgba(15, 20, 36, .48);
                                  border-right: 1px solid rgba(148, 163, 184, .16);
                                  backdrop-filter: blur(24px) saturate(145%);
                                  -webkit-backdrop-filter: blur(24px) saturate(145%);
                                  box-shadow: inset -1px 0 rgba(255,255,255,.035), 12px 0 42px rgba(2,6,23,.12);}
.stApp [data-testid="stSidebar"] > div:first-child {padding-top: .75rem;
                                                    background: rgba(15, 20, 36, .32);
                                                    backdrop-filter: blur(24px) saturate(145%);
                                                    -webkit-backdrop-filter: blur(24px) saturate(145%);}
.stApp [data-testid="stSidebarContent"] {display: flex; flex-direction: column;}
.stApp [data-testid="stSidebarUserContent"] {order: 1;}
.stApp [data-testid="stSidebarNav"] {order: 2; margin-top: .75rem; padding: 0 .15rem;}
.stApp [data-testid="stSidebarNav"] ul > li:first-child {display: none;}
.stApp [data-testid="stSidebarNavLink"] {margin: .15rem 0; padding: .55rem .7rem;
                                         border-radius: 11px; color: #cbd5e1;
                                         transition: background .18s ease, color .18s ease;}
.stApp [data-testid="stSidebarNavLink"]:hover {background: rgba(129,140,248,.14); color: #fff;}
.stApp [data-testid="stSidebarNavLink"][aria-current="page"] {
    background: rgba(129,140,248,.22); color: #fff; font-weight: 650;
    box-shadow: inset 0 0 0 1px rgba(165,180,252,.16);
}
.sidebar-home-link {display: block; margin: .55rem .1rem 0; padding: .55rem .7rem;
                    border: 1px solid rgba(148,163,184,.18); border-radius: 11px;
                    color: #e0e7ff !important; background: rgba(129,140,248,.12);
                    text-decoration: none !important; font-weight: 600;}
.sidebar-home-link:hover {background: rgba(129,140,248,.22);}
.brand-lockup {display: flex; align-items: center; gap: .75rem; margin: .25rem 0 .6rem;
               padding: .8rem .75rem; border: 1px solid rgba(255,255,255,.12);
               border-radius: 16px; background: linear-gradient(135deg, rgba(129,140,248,.17),
               rgba(168,85,247,.08)); box-shadow: inset 0 1px rgba(255,255,255,.1),
               0 10px 28px rgba(2,6,23,.16);}
.brand-mark {position: relative; display: grid; place-items: center; flex: 0 0 42px; height: 42px;
             border-radius: 13px; overflow: hidden;
             background: linear-gradient(145deg, #818cf8, #7c3aed 62%, #c084fc);
             box-shadow: 0 6px 18px rgba(99,102,241,.36), inset 0 1px rgba(255,255,255,.4);}
.brand-mark::before {content: ""; position: absolute; width: 20px; height: 13px; left: 10px; top: 14px;
                     border-left: 2px solid white; border-bottom: 2px solid white; transform: skewY(-32deg);}
.brand-mark::after {content: ""; position: absolute; width: 5px; height: 5px; right: 9px; top: 9px;
                    border-radius: 50%; background: #dbeafe; box-shadow: 0 0 10px rgba(255,255,255,.8);}
.brand-name {font-size: 1.02rem; font-weight: 750; letter-spacing: -.025em; color: #f8fafc;}
.brand-tagline {margin-top: .08rem; font-size: .68rem; color: #a5b4fc; letter-spacing: .08em;
                text-transform: uppercase;}
.block-container {width: min(100%, 1320px); padding-top: clamp(1.2rem, 3vw, 2.5rem);
                  padding-bottom: 4rem; padding-left: clamp(1rem, 3vw, 3rem);
                  padding-right: clamp(1rem, 3vw, 3rem);}
.hero {position: relative; overflow: hidden;
       background: linear-gradient(120deg, rgba(79, 70, 229, .96), rgba(124, 58, 237, .92) 58%,
                                  rgba(147, 51, 234, .88));
       color: white; padding: clamp(1.6rem, 4vw, 3rem); border: 1px solid rgba(255,255,255,.2);
       border-radius: 24px; margin-bottom: 1.5rem;
       box-shadow: 0 24px 70px rgba(49, 46, 129, .28), inset 0 1px 0 rgba(255,255,255,.22);}
.hero::after {content: ""; position: absolute; width: 18rem; height: 18rem; right: -5rem; top: -10rem;
              border-radius: 50%; border: 1px solid rgba(255,255,255,.22);
              box-shadow: 0 0 0 30px rgba(255,255,255,.04), 0 0 0 62px rgba(255,255,255,.035);
              pointer-events: none;}
.hero h1 {color: white; margin: 0 0 .55rem 0; letter-spacing: -.035em;}
.hero p {max-width: 46rem; color: rgba(255,255,255,.86); line-height: 1.7; margin-bottom: 0;}
.hero-kicker {margin-bottom: .65rem; color: #e0e7ff; font-size: .78rem; font-weight: 700;
              letter-spacing: .14em; text-transform: uppercase;}
.hero-cta {display: inline-flex; align-items: center; gap: .5rem; margin-top: 1.35rem;
           padding: .7rem 1rem; border: 1px solid rgba(255,255,255,.28); border-radius: 12px;
           color: #fff !important; background: rgba(255,255,255,.13); text-decoration: none !important;
           font-weight: 650; box-shadow: inset 0 1px rgba(255,255,255,.16);
           transition: background .2s ease, transform .2s ease;}
.hero-cta:hover {background: rgba(255,255,255,.22); transform: translateY(-2px);}
.intro-grid {display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: .85rem;
             margin: 1rem 0 2.2rem;}
.intro-tile {min-height: 118px; padding: 1.1rem 1.15rem; border: 1px solid rgba(255,255,255,.13);
             border-radius: 17px; background: linear-gradient(145deg, rgba(255,255,255,.085),
             rgba(255,255,255,.035)); box-shadow: inset 0 1px rgba(255,255,255,.08),
             0 12px 30px rgba(2,6,23,.12); backdrop-filter: blur(14px);}
.intro-tile strong {display: block; margin-bottom: .35rem; color: #e0e7ff; font-size: 1rem;}
.intro-tile span {color: #cbd5e1; font-size: .88rem; line-height: 1.55;}
.workspace-heading {margin-top: 2.8rem; padding-top: 1.4rem; border-top: 1px solid rgba(148,163,184,.18);}
.card {background: linear-gradient(145deg, rgba(248,250,252,.98), rgba(226,232,240,.96));
       color: #1f2937; border: 1px solid rgba(255,255,255,.65); border-radius: 18px;
       padding: 1.15rem 1.3rem; height: 100%; line-height: 1.75;
       box-shadow: 0 14px 36px rgba(2, 6, 23, .18);
       transition: transform .22s ease, box-shadow .22s ease;}
.card:hover {transform: translateY(-4px) perspective(800px) rotateX(1deg);
             box-shadow: 0 20px 44px rgba(2, 6, 23, .28);}
.card b {color: #111827;}
.card-dark {background: linear-gradient(145deg, rgba(30,41,59,.78), rgba(15,23,42,.85));
            color: #f1f5f9; border: 1px solid rgba(148,163,184,.22); border-radius: 18px;
            padding: 1.2rem; box-shadow: 0 12px 32px rgba(2,6,23,.22);}
.stApp h1, .stApp h2, .stApp h3 {letter-spacing: -.025em;}
.stApp h2 {margin-top: 1.8rem;}
.stApp [data-testid="stMetric"] {height: 100%; padding: 1rem 1.1rem;
                                 border: 1px solid rgba(148,163,184,.16); border-radius: 16px;
                                 background: linear-gradient(145deg, rgba(30,41,59,.72), rgba(15,23,42,.58));
                                 box-shadow: 0 12px 30px rgba(2,6,23,.14);}
.stApp [data-testid="stMetricLabel"] {color: #a5b4fc;}
.stApp [data-testid="stMetricValue"] {font-weight: 700;}
.stApp [data-testid="stPlotlyChart"],
.stApp [data-testid="stDataFrame"] {border-radius: 16px; overflow: hidden;}
.stApp [data-testid="stTextArea"] textarea,
.stApp [data-testid="stTextInput"] input,
.stApp [data-testid="stSelectbox"] [role="combobox"] {
    border-radius: 12px; border-color: rgba(148,163,184,.28);
}
.stApp [data-testid="stTextArea"] textarea:focus,
.stApp [data-testid="stTextInput"] input:focus {
    border-color: #818cf8; box-shadow: 0 0 0 2px rgba(129,140,248,.2);
}
.stApp [data-testid="stButton"] button,
.stApp [data-testid="stDownloadButton"] button {border-radius: 12px; transition: transform .18s ease, box-shadow .18s ease;}
.stApp [data-testid="stButton"] button:hover,
.stApp [data-testid="stDownloadButton"] button:hover {transform: translateY(-1px);}
.stApp [data-testid="stButton"] button[kind="primary"] {
    border: 1px solid rgba(255,255,255,.18);
    background: linear-gradient(110deg, #4f46e5, #7c3aed 60%, #9333ea);
    box-shadow: 0 8px 22px rgba(99,102,241,.28);
}
.chip {display: inline-block; padding: 4px 12px; margin: 3px; border-radius: 999px; font-size: .84rem; font-weight: 500;}
.chip-ok {background: #dcfce7; color: #166534; border: 1px solid #bbf7d0;}
.chip-miss {background: #fee2e2; color: #991b1b; border: 1px solid #fecaca;}
.chip-info {background: #e0e7ff; color: #3730a3; border: 1px solid #c7d2fe;}
.chip-transfer {background: #e0f2fe; color: #0369a1; border: 1px solid #bae6fd;}
.hl-weak {background: #fecaca; color: #7f1d1d; padding: 1px 4px; border-radius: 4px; font-weight: 500;}
.hl-strong {background: #bbf7d0; color: #14532d; padding: 1px 4px; border-radius: 4px; font-weight: 500;}
.resume-box {background: #ffffff; color: #1e293b; border: 1px solid #e2e8f0; border-radius: 12px; padding: 1.2rem;
             line-height: 1.75; font-size: 0.95rem; box-shadow: 0 4px 16px rgba(0,0,0,0.06);}
@media (max-width: 768px) {
    .block-container {padding-left: 1rem !important; padding-right: 1rem !important;}
    .hero {padding: 1.4rem; border-radius: 18px;}
    .hero h1 {font-size: clamp(1.7rem, 8vw, 2.4rem);}
    .intro-grid {grid-template-columns: 1fr; gap: .65rem; margin-bottom: 1.5rem;}
    .intro-tile {min-height: 0;}
    div[data-testid="stHorizontalBlock"] {flex-direction: column; gap: .75rem;}
    div[data-testid="column"] {width: 100% !important; flex: 1 1 100% !important; min-width: 100% !important;}
    .stApp [data-testid="stMetric"] {padding: .8rem; border-radius: 13px;}
    .stDataFrame {max-width: 100%; overflow-x: auto;}
}
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {scroll-behavior: auto !important; animation-duration: .01ms !important;
                             animation-iteration-count: 1 !important; transition-duration: .01ms !important;}
}
</style>
"""


@st.cache_resource(show_spinner="Loading models. The first run trains them and takes about 2 minutes.")
def get_engine():
    return CareerEngine.load_or_train()


def setup(title, icon):
    st.set_page_config(page_title=f"{title} | CareerPath AI", page_icon=icon, layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    db.init_db()
    with st.sidebar:
        st.markdown("""
        <div class='brand-lockup'>
          <div class='brand-mark' aria-hidden='true'></div>
          <div><div class='brand-name'>CareerPath AI</div>
          <div class='brand-tagline'>Career intelligence</div></div>
        </div>
        """, unsafe_allow_html=True)
        st.caption("Resume job matching, skill gap timeline and career trajectory prediction.")
        st.markdown("<a class='sidebar-home-link' href='./'>Home</a>", unsafe_allow_html=True)
        if db.history_enabled():
            name = st.text_input("Profile name", value=st.session_state.get("profile_name", "guest"),
                                 key=f"profile_input_{title}")
            st.session_state["profile_name"] = name.strip() or "guest"
            st.caption("History is enabled for this private deployment.")
        else:
            st.caption("Public demo mode: resume analyses are not saved.")
    return get_engine()


def profile():
    return st.session_state.get("profile_name", "guest")


def chips(items, kind="info"):
    if not items:
        return "<span class='chip chip-info'>None</span>"
    return "".join(f"<span class='chip chip-{kind}'>{i}</span>" for i in items)


def read_upload(file):
    return parse_document(file, file.name)


@st.dialog("Invalid resume file")
def _invalid_resume_dialog(message):
    st.error(message)
    if st.button("Close", key="close_invalid_resume_dialog"):
        st.session_state.pop("_invalid_resume_dialog", None)
        st.rerun()


def show_invalid_resume_dialog(message):
    _invalid_resume_dialog(message)


def valid_resume_or_dialog(text):
    if is_valid_resume(text):
        return True
    for result_key in ("analysis", "home_result", "ranked", "map", "improved"):
        st.session_state.pop(result_key, None)
    show_invalid_resume_dialog(
        "The supplied document does not appear to be a valid resume. "
        "Please upload or enter a resume containing work or education history and relevant skills."
    )
    return False


def resume_panel(key, height=260, default_text=""):
    if key not in st.session_state:
        saved_text = st.session_state.get("resume_store", default_text)
        st.session_state[key] = "" if saved_text == cfg.DEMO_RESUME else saved_text
    upload = st.file_uploader("Upload resume (.txt, .pdf, or .docx) or paste below",
                              type=["txt", "pdf", "docx"],
                              key=f"{key}_upload")
    marker = None
    if upload is not None:
        marker = (key, upload.name, hashlib.sha256(upload.getvalue()).hexdigest())
    marker_key = f"_last_upload_{key}"
    if marker and st.session_state.get(marker_key) != marker:
        text = read_upload(upload)
        for result_key in ("analysis", "home_result", "ranked", "map", "improved"):
            st.session_state.pop(result_key, None)
        if is_valid_resume(text):
            st.session_state[key] = text
            st.session_state.pop("_invalid_resume_dialog", None)
        else:
            st.session_state[key] = ""
            st.session_state["_invalid_resume_dialog"] = (
                f"'{upload.name}' does not appear to be a readable resume. "
                "Upload a resume containing work or education history and relevant skills."
            )
        st.session_state[marker_key] = marker
    elif upload is None:
        st.session_state.pop(marker_key, None)
        if st.session_state.get(key) == cfg.DEMO_RESUME:
            st.session_state[key] = ""
    text = st.text_area("Resume text", key=key, height=height)
    st.session_state["resume_store"] = text
    if not text or not is_valid_resume(text):
        for result_key in ("analysis", "home_result", "ranked", "map", "improved"):
            st.session_state.pop(result_key, None)
        st.session_state.pop("home_result_is_demo", None)
    if st.session_state.get("_invalid_resume_dialog"):
        show_invalid_resume_dialog(st.session_state["_invalid_resume_dialog"])
    return text


def resume_years(text):
    years = extract_resume_metadata(text)["extracted_years"]
    unit = "year" if years == 1 else "years"
    st.caption(f"Experience extracted from resume: **{years} {unit}**")
    return years

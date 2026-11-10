import pytest
import pandas as pd
from core import nlp_feedback as nlp
from core import config as cfg
from core import explain
from core.engine import CareerEngine
from core.prerequisites import PREREQUISITES, learning_order
from core.trajectory import build_trajectory, skill_roi, what_if


@pytest.fixture(scope="module")
def engine():
    return CareerEngine.load_or_train()


@pytest.fixture(scope="module")
def result(engine):
    return engine.analyze(cfg.DEMO_RESUME, 1, 0)


def test_analysis_ranges(result):
    assert 0 <= result["match_pct"] <= 100
    assert result["level"] in ["Junior", "Mid", "Senior"]
    assert result["weeks"] >= 0


def test_prerequisites_are_valid_and_ordered():
    for skill, pres in PREREQUISITES.items():
        assert skill in cfg.ALL_SKILLS and all(p in cfg.ALL_SKILLS for p in pres)
    order = learning_order(["tensorflow"], [])
    assert order.index("python") < order.index("machine learning") < order.index("deep learning") < order.index("tensorflow")


def test_owned_skills_are_not_added_as_foundations():
    assert "python" not in learning_order(["pandas"], ["python"])


def test_timeline_total_scales_with_hours(engine, result):
    slow = engine.timeline(result["missing"], result["cand_skills"], result["weeks"], 5)["weeks"].sum()
    fast = engine.timeline(result["missing"], result["cand_skills"], result["weeks"], 20)["weeks"].sum()
    assert fast < slow


def test_trajectory_skill_match_never_drops(engine, result):
    tl = engine.timeline(result["missing"], result["cand_skills"], result["weeks"], 10)
    traj = build_trajectory(engine, result["resume_text"], result["years"], result["job_id"], result["cand_skills"], tl)
    assert traj["skill_match_pct"].is_monotonic_increasing
    assert traj["skill_match_pct"].iloc[-1] == pytest.approx(100.0)


def test_roi_is_sorted_and_positive(engine, result):
    roi = skill_roi(engine, result["resume_text"], result["years"], result["job_id"], result["cand_skills"], result["missing"])
    assert (roi["roi"] >= 0).all()
    assert roi["roi"].is_monotonic_decreasing


def test_what_if_improves_skill_match(engine, result):
    out = what_if(engine, result, result["missing"][:1])
    assert out["after"]["skill_match_pct"] > out["before"]["skill_match_pct"]


def test_explanation_adds_up_to_prediction(engine, result):
    intercept, df = explain.weeks_contributions(engine, result["feats"])
    row = pd.DataFrame([result["feats"]])[cfg.REG_FEATURES]
    reg = engine.b["reg"]
    raw = float(reg["best"].predict(reg["scaler"].transform(row))[0])
    assert intercept + df["contribution"].sum() == pytest.approx(raw, abs=1e-6)


def test_auto_improve_removes_weak_phrases():
    improved, changes = nlp.auto_improve("Responsible for dashboards. Worked on queries.")
    assert not nlp.WEAK_REGEX.search(improved)
    assert len(changes) == 2


def test_rank_jobs_and_screen(engine):
    assert len(engine.rank_jobs(cfg.DEMO_RESUME, 1, 7, "All")) == 7
    s = engine.screen(cfg.DEMO_RESUME, 1, 0)
    assert 0 <= s["fit_score"] <= 100